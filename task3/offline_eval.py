import argparse
import json
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from common.io import git_commit, load_yaml, write_json
from task3.answers import is_correct
from task3.load import N_ROLLOUTS, load_dataset
from task3.scores import mean_logprob, self_certainty, vision_only, visconf
from task3.selectors import first_answer, highest_score, rwc, self_consistency

#method -> (score, aggregation). "first" and "sc" use no score; vlm-only methods need attention
METHODS= {
    "first_ans": (None, "first"),
    "self_consistency": (None, "vote"),
    "max_logprob": ("logprob", "highest"),
    "max_self_certainty": ("self_certainty", "highest"),
    "self_certainty_rwc": ("self_certainty", "rwc"),
    "max_vision": ("vision", "highest"),
    "vision_rwc": ("vision", "rwc"),
    "max_visconf": ("visconf", "highest"),
    "visconf_rwc": ("visconf", "rwc"),
    "max_visconf_token": ("visconf_token", "highest"),
    "visconf_token_rwc": ("visconf_token", "rwc"),
    #length-only control for the signal analysis: shortest retained response
    "shortest": ("neg_length", "highest")
}
VLM_SCORES= {"vision", "visconf", "visconf_token"}


def question_pool(data, start):
    #one question's 32 rollouts (rollout id = index) and the scores that do not depend on the candidate set
    rows= slice(start, start + N_ROLLOUTS)
    pool= {"logp": data["logp"][rows], "kl": data["kl"][rows], "n_tokens": data["n_tokens"][rows]}
    pool["fixed"]= {
        "logprob": mean_logprob(pool["logp"]),
        "self_certainty": self_certainty(pool["kl"]),
        "neg_length": -pool["n_tokens"].astype(np.float64)
    }

    if "v" in data:
        pool["v"]= data["v"][rows]
        pool["fixed"]["vision"]= vision_only(pool["v"])

    return pool


def prefix_scores(pool, ids, needed):
    #scores of the revealed candidates only: fixed per-rollout scores are indexed, visconf is recalibrated on them
    out= {s: pool["fixed"][s][ids] for s in needed if s in pool["fixed"]}

    if needed & {"visconf", "visconf_token"}:
        Z, Z_tok, _= visconf([pool["kl"][i] for i in ids], [pool["v"][i] for i in ids])
        out["visconf"], out["visconf_token"]= Z, Z_tok

    return out


def select(method, answers, scores):
    score, agg= METHODS[method]
    if agg == "first":
        return first_answer(answers)
    if agg == "vote":
        return self_consistency(answers)
    if agg == "highest":
        return highest_score(answers, scores[score])

    return rwc(answers, scores[score])


def evaluate_question(q, groups, gold_group, pool, orders, budgets, methods):
    #one row per (order, budget, method)
    needed= {METHODS[m][0] for m in methods} - {None}
    correct= np.array([is_correct(g, gold_group) for g in groups])

    rows= []
    for o, order in enumerate(orders):
        for n in budgets:
            ids= np.array(order[:n])
            answers= [groups[i] for i in ids]
            scores= prefix_scores(pool, ids, needed)

            cost_all= int(pool["n_tokens"][ids].sum())
            coverage= bool(correct[ids].any())
            distinct= len(set(answers))

            for m in methods:
                a= select(m, answers, scores)
                rows.append({
                    "sample_id": q,
                    "order": o,
                    "n": n,
                    "method": m,
                    "selected": a,
                    "correct": is_correct(a, gold_group),
                    #firstans consumes only its first candidate
                    "tokens": int(pool["n_tokens"][ids[0]]) if m == "first_ans" else cost_all,
                    "coverage": coverage,
                    "distinct": distinct
                })

    return rows


def summarize(rows, splits):
    #average over the ten orders within a question, then over the questions of each split
    t= {}
    for r in rows:
        key= (r["sample_id"], r["n"], r["method"])
        acc= t.setdefault(key, {"correct": 0.0, "tokens": 0.0, "coverage": 0.0, "distinct": 0.0, "missed": 0.0, "no_correct": 0.0, "k": 0})
        acc["correct"]+= r["correct"]
        acc["tokens"]+= r["tokens"]
        acc["coverage"]+= r["coverage"]
        acc["distinct"]+= r["distinct"]
        #failure decomposition: no correct candidate generated vs one was available but not selected
        acc["no_correct"]+= not r["coverage"]
        acc["missed"]+= r["coverage"] and not r["correct"]
        acc["k"]+= 1

    per_q= []
    for (q, n, m), acc in t.items():
        k= acc.pop("k")
        per_q.append({"sample_id": q, "n": n, "method": m, **{x: v / k for x, v in acc.items()}})

    summary= []
    for split, ids in splits.items():
        ids= set(ids)
        groups= {}
        for r in per_q:
            if r["sample_id"] in ids:
                groups.setdefault((r["n"], r["method"]), []).append(r)
        for (n, m), rs in sorted(groups.items()):
            summary.append({"split": split, "n": n, "method": m, "n_questions": len(rs), **{x: float(np.mean([r[x] for r in rs])) for x in ["correct", "coverage", "tokens", "distinct", "no_correct", "missed"]}})

    return per_q, summary


def main(args):
    cfg= load_yaml(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    for name in args.datasets:
        t0= time.time()
        data, ids= load_dataset(cfg["data_dir"], name)
        with open(os.path.join(args.answers_dir, f"answer_groups_{name}.json")) as f:
            groups= json.load(f)["questions"]
        with open(os.path.join(args.split_dir, f"{name}_orders.json")) as f:
            orders= json.load(f)
        with open(os.path.join(args.split_dir, f"{name}_split.json")) as f:
            split= json.load(f)

        methods= [m for m in METHODS if name == "mathvista" or METHODS[m][0] not in VLM_SCORES]
        qs= ids[:args.limit] if args.limit else ids

        rows= []
        for i, q in enumerate(qs):
            pool= question_pool(data, i * N_ROLLOUTS)
            rows+= evaluate_question(q, groups[q]["groups"], groups[q]["gold_group"], pool, orders[q][:cfg["n_orders"]], cfg["budgets"], methods)
            if (i + 1) % 100 == 0:
                print(f"{name}: {i + 1}/{len(qs)} questions ({time.time() - t0:.0f}s)", flush=True)

        pq.write_table(pa.Table.from_pylist(rows), os.path.join(args.out_dir, f"{name}_rows.parquet"))
        per_q, summary= summarize(rows, {"dev": split["dev"], "eval": split["eval"]})
        pq.write_table(pa.Table.from_pylist(per_q), os.path.join(args.out_dir, f"{name}_per_question.parquet"))
        write_json(os.path.join(args.out_dir, f"{name}_summary.json"), {
            "info": {"git_commit": git_commit(), "seed": cfg["seed"], "budgets": cfg["budgets"], "n_orders": cfg["n_orders"], "methods": methods, "seconds": time.time() - t0, "limit": args.limit},
            "summary": summary
        })

        for s in summary:
            if s["split"] == "eval" and s["n"] in (1, 32):
                print(f"{name} eval N={s['n']:>2} {s['method']:<20} acc {s['correct']:.4f} cov {s['coverage']:.4f} tok {s['tokens']:.0f}")

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
        default="results/task3/offline"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None
    )

    raise SystemExit(main(parser.parse_args()))
