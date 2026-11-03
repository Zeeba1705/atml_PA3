import argparse
import json
import os

import numpy as np

from common.io import git_commit, load_yaml, write_json
from task3.answers import is_correct
from task3.load import N_ROLLOUTS, load_dataset
from task3.offline_eval import prefix_scores, question_pool
from task3.scores import calibrate
from task3.selectors import average_ranks, rwc, self_consistency

#rollout-level signals compared at N = 32 (neg_length = the length-only control: shortest first)
SCORES= {"math": ["logprob", "self_certainty", "neg_length"], "mathvista": ["logprob", "self_certainty", "neg_length", "vision", "visconf", "visconf_token"]}
#illustrative trajectories: this many questions by the fixed rule below
N_TRAJECTORY_QUESTIONS= 3


def spearman(a, b):
    #rank correlation with average ranks; None when either side is constant
    ra, rb= average_ranks(a), average_ranks(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None

    return float(np.corrcoef(ra, rb)[0, 1])


def quartile_gap(scores, correct):
    #correctness of the top quarter minus the bottom quarter by score (ties: earlier rollout id ranks higher)
    k= len(scores) // 4
    order= np.argsort(-np.where(np.isnan(scores), -np.inf, scores), kind="stable")

    return float(correct[order[:k]].mean() - correct[order[-k:]].mean())


def summary(values):
    v= np.array([x for x in values if x is not None], dtype=np.float64)
    if len(v) == 0:
        return {"n": 0}

    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)), "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)), "frac_positive": float((v > 0).mean())}


def main(args):
    cfg= load_yaml(args.config)
    out= {"info": {"git_commit": git_commit(), "split": "eval", "n": N_ROLLOUTS}, "datasets": {}}

    for name in args.datasets:
        data, ids= load_dataset(cfg["data_dir"], name)
        with open(os.path.join(args.answers_dir, f"answer_groups_{name}.json")) as f:
            groups= json.load(f)["questions"]
        with open(os.path.join(args.split_dir, f"{name}_split.json")) as f:
            ev= json.load(f)["eval"]
        with open(os.path.join(args.split_dir, f"{name}_orders.json")) as f:
            orders= json.load(f)

        row= {q: i * N_ROLLOUTS for i, q in enumerate(ids)}
        all_ids= np.arange(N_ROLLOUTS)
        res= {s: {"len_corr": [], "correct_corr": [], "gap_all": [], "gap_mixed": []} for s in SCORES[name]}
        len_correct, cv_corr, cov_check, trajectories= [], [], [], []

        for q in ev:
            pool= question_pool(data, row[q])
            correct= np.array([is_correct(g, groups[q]["gold_group"]) for g in groups[q]["groups"]], dtype=np.float64)
            mixed= 0 < correct.mean() < 1
            length= pool["n_tokens"].astype(np.float64)
            scores= prefix_scores(pool, all_ids, set(SCORES[name]))

            len_correct.append(spearman(length, correct) if mixed else None)
            for s in SCORES[name]:
                res[s]["len_corr"].append(spearman(length, scores[s]) if s != "neg_length" else None)
                res[s]["correct_corr"].append(spearman(scores[s], correct) if mixed else None)
                gap= quartile_gap(scores[s], correct)
                res[s]["gap_all"].append(gap)
                res[s]["gap_mixed"].append(gap if mixed else None)

            if name == "mathvista":
                #within-rollout correlation of calibrated confidence and vision share (calibrated on all 32)
                c= calibrate(pool["kl"])
                per= [float(np.corrcoef(ci, vi)[0, 1]) for ci, vi in zip(c, pool["v"]) if len(ci) > 2 and np.std(ci) > 0 and np.std(vi) > 0]
                cv_corr.append(float(np.mean(per)) if per else None)

                cov= np.array([np.mean(ci * vi) - np.mean(ci) * np.mean(vi) if len(ci) else 0.0 for ci, vi in zip(c, pool["v"])])
                cov_check.append(float(np.max(np.abs(scores["visconf_token"] - scores["visconf"] - cov))))

                #fixed rule: first eval questions (split order) where visconf-rwc and sc disagree on order 0 at N = 32;
                #keep the highest- and lowest-visconf rollouts' token series
                if len(trajectories) < N_TRAJECTORY_QUESTIONS:
                    order= np.array(orders[q][0])
                    answers= [groups[q]["groups"][i] for i in order]
                    #all 32 revealed: same calibration set as above, scores listed in this order
                    a_rwc= rwc(answers, prefix_scores(pool, order, {"visconf"})["visconf"])
                    a_sc= self_consistency(answers)
                    if a_rwc != a_sc:
                        hi, lo= int(np.nanargmax(scores["visconf"])), int(np.nanargmin(scores["visconf"]))
                        trajectories.append({
                            "sample_id": q,
                            "gold_group": groups[q]["gold_group"],
                            "visconf_rwc_answer": a_rwc,
                            "sc_answer": a_sc,
                            "rollouts": [
                                {"which": w, "rollout_id": i, "answer": groups[q]["groups"][i], "correct": bool(correct[i]), "visconf": float(scores["visconf"][i]), "c": c[i].tolist(), "v": pool["v"][i].tolist(), "kl": pool["kl"][i].tolist()}
                                for w, i in [("highest_visconf", hi), ("lowest_visconf", lo)]
                            ]
                        })

        d= {
            "n_questions": len(ev),
            "length_vs_correct": summary(len_correct),
            "scores": {s: {k: summary(v) for k, v in r.items()} for s, r in res.items()}
        }
        if name == "mathvista":
            d["confidence_vision_corr_per_question"]= summary(cv_corr)
            d["max_abs_ztok_minus_z_minus_cov"]= float(np.max(cov_check))
            d["trajectory_rule"]= "first eval questions in split order where VisConf-RWC and SC disagree (order 0, N=32); highest and lowest VisConf rollouts"
            d["trajectories"]= trajectories
        out["datasets"][name]= d

        for s, r in d["scores"].items():
            print(f"{name} {s:<15} gap(all) {r['gap_all']['mean']:+.4f} gap(mixed) {r['gap_mixed'].get('mean', float('nan')):+.4f} corr(score,correct) {r['correct_corr'].get('mean', float('nan')):+.4f} corr(len,score) {r['len_corr'].get('mean', float('nan')):+.4f}")
        if name == "mathvista":
            print(name, "conf-vision corr", d["confidence_vision_corr_per_question"], "cov identity max err", d["max_abs_ztok_minus_z_minus_cov"])

    write_json(os.path.join(args.out_dir, "signals.json"), out)

    return 0


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["math", "mathvista"]
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/task3/base.yaml"
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default="splits/task3"
    )
    parser.add_argument(
        "--answers_dir",
        type=str,
        default="results/task3"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/task3"
    )

    raise SystemExit(main(parser.parse_args()))
