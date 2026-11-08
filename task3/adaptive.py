import argparse
import json
import os

import numpy as np

from common.io import git_commit, load_yaml, write_json
from task3.answers import NO_ANSWER, is_correct
from task3.load import N_ROLLOUTS, load_dataset
from task3.offline_eval import prefix_scores, question_pool
from task3.selectors import rwc, rwc_margin, self_consistency, vote

#handout p. 13: stopping is checked after these many revealed candidates, otherwise all 32 are used
CHECKPOINTS= [4, 8, 16]


def agreement_stop(answers, tau):
    #unique, non-NO_ANSWER leading answer supported by at least tau * n of the n revealed candidates
    _, totals= vote(answers, np.ones(len(answers)))
    counts= sorted(totals.items(), key=lambda kv: -kv[1])

    leader, top= counts[0]
    unique= len(counts) == 1 or counts[1][1] < top

    return unique and leader != NO_ANSWER and top >= tau * len(answers)


def quality_stop(answers, scores, tau, gamma):
    #agreement AND self-consistency winner == rwc winner AND rwc margin >= gamma (scores from revealed only)
    if not agreement_stop(answers, tau):
        return False

    return self_consistency(answers) == rwc(answers, scores) and rwc_margin(answers, scores) >= gamma


def checkpoint_stats(groups, order, pool, score_name):
    #stopping evidence after n revealed candidates, for each check n and the final 32: agreement leader and its
    #support, self-consistency winner, rwc winner + margin (scores from the revealed candidates only), tokens used
    out= {}
    for n in CHECKPOINTS + [N_ROLLOUTS]:
        ids= np.array(order[:n])
        answers= [groups[i] for i in ids]
        scores= prefix_scores(pool, ids, {score_name})[score_name]

        _, totals= vote(answers, np.ones(n))
        top= max(totals.values())
        leaders= [a for a, c in totals.items() if c == top]

        out[n]= {
            "sc": self_consistency(answers),
            "leader": leaders[0] if len(leaders) == 1 else None,
            "support": top / n,
            "rwc": rwc(answers, scores),
            "margin": rwc_margin(answers, scores),
            "tokens": int(pool["n_tokens"][ids].sum())
        }

    return out


def stop_point(stats, family, tau, gamma):
    #first check where the controller stops (else 32); both controllers return the current self-consistency answer
    if family != "always_32":
        for n in CHECKPOINTS:
            s= stats[n]
            agree= s["leader"] is not None and s["leader"] != NO_ANSWER and s["support"] >= tau
            if agree and (family == "agreement" or (s["sc"] == s["rwc"] and s["margin"] >= gamma)):
                return n

    return N_ROLLOUTS


def simulate(stats_by_q, gold, family, tau, gamma, ids):
    #per question: mean over orders of correctness, tokens, rollouts; plus every (order, stop n, correct) record
    per_q, records= {}, []
    for q in ids:
        acc, tok, used= [], [], []
        for o, stats in enumerate(stats_by_q[q]):
            n= stop_point(stats, family, tau, gamma)
            ok= is_correct(stats[n]["sc"], gold[q])
            acc.append(ok)
            tok.append(stats[n]["tokens"])
            used.append(n)
            records.append({"sample_id": q, "order": o, "stop_n": n, "correct": ok, "correct_at_32": is_correct(stats[N_ROLLOUTS]["sc"], gold[q]), "answer": stats[n]["sc"], "answer_at_32": stats[N_ROLLOUTS]["sc"], "tokens": stats[n]["tokens"]})
        per_q[q]= {"correct": float(np.mean(acc)), "tokens": float(np.mean(tok)), "rollouts": float(np.mean(used))}

    mean= {k: float(np.mean([v[k] for v in per_q.values()])) for k in ["correct", "tokens", "rollouts"]}

    return mean, per_q, records


def grid(cfg):
    out= [("always_32", None, None)]
    out+= [("agreement", t, None) for t in cfg["adaptive_tau"]]
    out+= [("quality", t, g) for t in cfg["adaptive_tau"] for g in cfg["adaptive_gamma"]]

    return out


def select_on_dev(results, sc32_acc):
    #per family: lowest mean tokens with dev accuracy >= fixed-32 sc - 1 pp (always_32 as fallback);
    #ties -> higher accuracy, then smaller tau, then smaller gamma
    chosen= {}
    for family in ["agreement", "quality"]:
        ok= [r for r in results if r["family"] in (family, "always_32") and r["correct"] >= sc32_acc - 0.01 - 1e-12]
        best= min(ok, key=lambda r: (r["tokens"], -r["correct"], r["tau"] if r["tau"] is not None else 2.0, r["gamma"] if r["gamma"] is not None else 2.0))
        chosen[family]= best

    return chosen


def main(args):
    cfg= load_yaml(args.config)
    os.makedirs(args.out_dir, exist_ok=True)
    selected_path= os.path.join(args.out_dir, "adaptive_selected.json")

    data_all, splits, stats_all, gold_all= {}, {}, {}, {}
    for name in args.datasets:
        data, ids= load_dataset(cfg["data_dir"], name)
        with open(os.path.join(args.answers_dir, f"answer_groups_{name}.json")) as f:
            groups= json.load(f)["questions"]
        with open(os.path.join(args.split_dir, f"{name}_orders.json")) as f:
            orders= json.load(f)
        with open(os.path.join(args.split_dir, f"{name}_split.json")) as f:
            splits[name]= json.load(f)

        #quality signal: self-certainty for the llm, visconf (recalibrated on revealed candidates) for the vlm
        score_name= cfg["adaptive_score"][name]
        stats_all[name]= {}
        for i, q in enumerate(ids):
            pool= question_pool(data, i * N_ROLLOUTS)
            stats_all[name][q]= [checkpoint_stats(groups[q]["groups"], order, pool, score_name) for order in orders[q][:cfg["n_orders"]]]
        gold_all[name]= {q: groups[q]["gold_group"] for q in ids}
        print(name, "stopping evidence computed", flush=True)

    #phase 1: selection on dev only, frozen to json before any held-out question is simulated
    if os.path.exists(selected_path):
        with open(selected_path) as f:
            selected= json.load(f)
        print("using frozen selection", selected_path)
    else:
        selected= {"info": {"git_commit": git_commit(), "rule": "lowest mean tokens with dev acc >= fixed-32 SC - 1 pp; ties: higher acc, smaller tau, smaller gamma"}, "datasets": {}}
        for name in args.datasets:
            dev= splits[name]["dev"]
            assert not set(dev) & set(splits[name]["eval"])
            results= []
            for family, tau, gamma in grid(cfg):
                mean, _, _= simulate(stats_all[name], gold_all[name], family, tau, gamma, dev)
                results.append({"family": family, "tau": tau, "gamma": gamma, **mean})
            sc32= next(r for r in results if r["family"] == "always_32")["correct"]
            chosen= select_on_dev(results, sc32)
            selected["datasets"][name]= {"dev_sc32_accuracy": sc32, "score": cfg["adaptive_score"][name], "selected": chosen, "dev_grid": results}
            print(name, "selected:", {f: (c["family"], c["tau"], c["gamma"], round(c["correct"], 4), round(c["tokens"])) for f, c in chosen.items()})
        write_json(selected_path, selected)

    #phase 2: held-out evaluation with the frozen settings, alongside fixed-n self-consistency from offline_eval
    report= {}
    for name in args.datasets:
        ev= splits[name]["eval"]
        out= {}
        for family, c in selected["datasets"][name]["selected"].items():
            mean, per_q, records= simulate(stats_all[name], gold_all[name], c["family"], c["tau"], c["gamma"], ev)
            dist= {int(n): sum(r["stop_n"] == n for r in records) / len(records) for n in CHECKPOINTS + [N_ROLLOUTS]}
            #early stops that were wrong although fixed-32 self-consistency was right on the same order
            recoverable= [r for r in records if r["stop_n"] < N_ROLLOUTS and not r["correct"] and r["correct_at_32"]]
            wrong_early= [r for r in records if r["stop_n"] < N_ROLLOUTS and not r["correct"]]
            out[family]= {
                "setting": {"family": c["family"], "tau": c["tau"], "gamma": c["gamma"]},
                **mean,
                "stop_distribution": dist,
                "incorrect_early_stops": len(wrong_early) / len(records),
                "recoverable_by_32": len(recoverable) / len(records),
                "recoverable_cases": recoverable[:200],
                "per_question": per_q
            }
            print(f"{name} eval {family:<10} {c['family']} tau={c['tau']} gamma={c['gamma']}: acc {mean['correct']:.4f} tokens {mean['tokens']:.0f} rollouts {mean['rollouts']:.2f} stops {dist}")
        report[name]= out

    write_json(os.path.join(args.out_dir, "adaptive_eval.json"), {"info": {"git_commit": git_commit(), "selected_from": selected_path}, "datasets": report})

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
