import argparse
import csv
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from common.io import read_jsonl, write_json

DATASETS= ["math", "mathvista"]
#methods drawn in the figures (all methods are in the tables); fixed colours per method, never by rank
PLOT_METHODS= {
    "first_ans": ("FirstAns", "#2a78d6"),
    "self_consistency": ("Self-consistency", "#eb6834"),
    "max_logprob": ("Max log-prob", "#1baf7a"),
    "max_self_certainty": ("Max self-certainty", "#eda100"),
    "self_certainty_rwc": ("Self-certainty + RWC", "#e87ba4"),
    "max_visconf": ("Max VisConf", "#008300"),
    "visconf_rwc": ("VisConf + RWC", "#6250d6")
}
COVERAGE_COLOR= "#8c8b85"
INK= "#2f2e2b"
MUTED= "#7a7972"
GRID= "#e6e5e0"
SURFACE= "#fcfcfb"


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer= csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("saved", path)


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)


def save(fig, out_dir, name):
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print("saved", os.path.join(out_dir, name + ".png"))
    plt.close(fig)


def paired_ci(a, b, n_boot, seed):
    #question-level paired bootstrap of mean(a - b) (a, b already averaged over the ten orders per question)
    d= np.asarray(a) - np.asarray(b)
    idx= np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    means= d[idx].mean(axis=1)

    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def per_question(results_dir, name, split_ids):
    #(n, method) -> {sample_id: row} on one split
    t= pq.read_table(os.path.join(results_dir, "offline", f"{name}_per_question.parquet")).to_pylist()
    out= {}
    for r in t:
        if r["sample_id"] in split_ids:
            out.setdefault((r["n"], r["method"]), {})[r["sample_id"]]= r

    return out


def method_rows(pq_rows, name, cfg_boot):
    #every method x budget: accuracy, coverage, tokens, distinct groups, failure split, delta vs self-consistency
    rows= []
    for (n, m), by_q in sorted(pq_rows.items()):
        qs= sorted(by_q)
        sc= pq_rows[(n, "self_consistency")]
        d, lo, hi= paired_ci([by_q[q]["correct"] for q in qs], [sc[q]["correct"] for q in qs], cfg_boot["n"], cfg_boot["seed"])
        rows.append({
            "dataset": name,
            "n": n,
            "method": m,
            "accuracy": round(float(np.mean([by_q[q]["correct"] for q in qs])), 4),
            "delta_vs_sc_pp": round(100 * d, 2),
            "delta_ci_lo_pp": round(100 * lo, 2),
            "delta_ci_hi_pp": round(100 * hi, 2),
            "oracle_coverage": round(float(np.mean([by_q[q]["coverage"] for q in qs])), 4),
            "mean_tokens": round(float(np.mean([by_q[q]["tokens"] for q in qs])), 1),
            "distinct_groups": round(float(np.mean([by_q[q]["distinct"] for q in qs])), 3),
            "fail_no_correct_candidate": round(float(np.mean([by_q[q]["no_correct"] for q in qs])), 4),
            "fail_correct_available_missed": round(float(np.mean([by_q[q]["missed"] for q in qs])), 4),
            "n_questions": len(qs)
        })

    return rows


def plot_curves(rows, out_dir):
    #accuracy vs candidates and vs tokens, oracle coverage dashed; one panel per dataset (separate scales)
    for x_key, x_label, name in [("n", "candidates N (log scale)", "accuracy_vs_n"), ("mean_tokens", "mean generated tokens per question (log scale)", "accuracy_vs_tokens")]:
        fig, axes= plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURFACE)
        for ax, name_d in zip(axes, DATASETS):
            style(ax)
            sub= [r for r in rows if r["dataset"] == name_d]
            sc= sorted([r for r in sub if r["method"] == "self_consistency"], key=lambda r: r["n"])
            ax.plot([r[x_key] for r in sc], [r["oracle_coverage"] for r in sc], color=COVERAGE_COLOR, linestyle="--", linewidth=2, label="Oracle coverage")
            for m, (label, color) in PLOT_METHODS.items():
                pts= sorted([r for r in sub if r["method"] == m], key=lambda r: r["n"])
                if pts:
                    ax.plot([r[x_key] for r in pts], [r["accuracy"] for r in pts], color=color, linewidth=2, marker="o", markersize=4, label=label)
            ax.set_xscale("log", base=2 if x_key == "n" else 10)
            ax.set_xlabel(x_label, color=INK)
            ax.set_title({"math": "MATH (Qwen2.5-3B)", "mathvista": "MathVista (Qwen2.5-VL-3B)"}[name_d], color=INK, loc="left", fontsize=10)
        axes[0].set_ylabel("accuracy (held-out, 800 questions)", color=INK)
        #one shared legend under the panels, so it never covers the lines
        handles, labels= axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=INK, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
        fig.tight_layout()
        save(fig, out_dir, name)


def adaptive_rows(results_dir, method_table):
    with open(os.path.join(results_dir, "adaptive_eval.json")) as f:
        ad= json.load(f)["datasets"]
    with open(os.path.join(results_dir, "adaptive_selected.json")) as f:
        sel= json.load(f)["datasets"]

    rows= []
    for name in DATASETS:
        for r in sorted([r for r in method_table if r["dataset"] == name and r["method"] == "self_consistency"], key=lambda r: r["n"]):
            rows.append({"dataset": name, "policy": f"fixed SC N={r['n']}", "tau": None, "gamma": None, "accuracy": r["accuracy"], "mean_tokens": r["mean_tokens"], "mean_rollouts": r["n"], "stop_4": None, "stop_8": None, "stop_16": None, "stop_32": None, "incorrect_early_stops": None, "recoverable_by_32": None, "dev_accuracy": None})
        for family, d in ad[name].items():
            s= d["setting"]
            rows.append({
                "dataset": name, "policy": f"adaptive {family}", "tau": s["tau"], "gamma": s["gamma"],
                "accuracy": round(d["correct"], 4), "mean_tokens": round(d["tokens"], 1), "mean_rollouts": round(d["rollouts"], 2),
                **{f"stop_{n}": round(d["stop_distribution"][str(n)], 4) for n in [4, 8, 16, 32]},
                "incorrect_early_stops": round(d["incorrect_early_stops"], 4), "recoverable_by_32": round(d["recoverable_by_32"], 4),
                "dev_accuracy": round(sel[name]["selected"][family]["correct"], 4)
            })

    return rows, ad


def plot_adaptive(rows, out_dir):
    fig, axes= plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURFACE)
    for ax, name in zip(axes, DATASETS):
        style(ax)
        fixed= [r for r in rows if r["dataset"] == name and r["policy"].startswith("fixed")]
        ax.plot([r["mean_tokens"] for r in fixed], [r["accuracy"] for r in fixed], color=PLOT_METHODS["self_consistency"][1], linewidth=2, marker="o", markersize=4, label="Fixed-N self-consistency")
        for r, color, marker in zip([r for r in rows if r["dataset"] == name and r["policy"].startswith("adaptive")], ["#2a78d6", "#6250d6"], ["s", "D"]):
            ax.scatter(r["mean_tokens"], r["accuracy"], s=70, color=color, marker=marker, edgecolors=SURFACE, linewidths=2, zorder=3, label=r["policy"].replace("adaptive ", "Adaptive: "))
        ax.set_xscale("log")
        ax.set_xlabel("mean generated tokens per question (log scale)", color=INK)
        ax.set_title({"math": "MATH", "mathvista": "MathVista"}[name], color=INK, loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    axes[0].set_ylabel("accuracy (held-out)", color=INK)
    fig.tight_layout()
    save(fig, out_dir, "adaptive_vs_fixed")


def signal_rows(results_dir):
    with open(os.path.join(results_dir, "signals.json")) as f:
        sig= json.load(f)["datasets"]

    rows= []
    for name, d in sig.items():
        for s, r in d["scores"].items():
            rows.append({
                "dataset": name, "score": s,
                "top_minus_bottom_quartile_correct_all": round(r["gap_all"]["mean"], 4),
                "top_minus_bottom_quartile_correct_mixed": round(r["gap_mixed"]["mean"], 4),
                "n_mixed_questions": r["gap_mixed"]["n"],
                "spearman_score_correct_mean": round(r["correct_corr"]["mean"], 4),
                "spearman_length_score_mean": None if r["len_corr"]["n"] == 0 else round(r["len_corr"]["mean"], 4)
            })

    return rows, sig


def case_lists(results_dir, ad, split):
    #fixed rules: (a) selection disagreement: first held-out questions (split order) where SC and the best RWC
    #variant disagree on order 0 at N = 32; (b) premature stopping: first recoverable early stops per dataset
    cases= {}
    for name in DATASETS:
        rows= pq.read_table(os.path.join(results_dir, "offline", f"{name}_rows.parquet")).to_pylist()
        rwc_m= "visconf_rwc" if name == "mathvista" else "self_certainty_rwc"
        key= {(r["sample_id"], r["method"]): r for r in rows if r["order"] == 0 and r["n"] == 32}
        dis= []
        for q in split[name]["eval"]:
            a, b= key[(q, "self_consistency")], key[(q, rwc_m)]
            if a["selected"] != b["selected"]:
                dis.append({"sample_id": q, "sc": a["selected"], "sc_correct": a["correct"], rwc_m: b["selected"], "rwc_correct": b["correct"]})
        cases[name]= {
            "selection_disagreement_rule": f"held-out questions in split order where self-consistency and {rwc_m} disagree (order 0, N = 32)",
            "selection_disagreement": dis[:20],
            "premature_stopping_rule": "adaptive early stops that were wrong while fixed-32 self-consistency was right (first in held-out order)",
            "premature_stopping": {f: d["recoverable_cases"][:20] for f, d in ad[name].items()}
        }

    return cases


def live_rows(results_dir):
    #step 4 summary per strategy, if the live results exist; costs charge the shared initial response to refine
    files= sorted(glob.glob(os.path.join(results_dir, "live", "shard*of*.jsonl")))
    rows= [r for f in files for r in read_jsonl(f)]
    if not rows:
        return None, None

    out= []
    for strat in ["single", "parallel", "refine"]:
        calls= {"single": lambda r: [r["calls"]["single"]], "parallel": lambda r: r["calls"]["parallel"], "refine": lambda r: [r["calls"]["parallel"][0], r["calls"]["refine"]]}[strat]
        final= {"single": lambda r: r["calls"]["single"], "parallel": None, "refine": lambda r: r["calls"]["refine"]}[strat]
        out.append({
            "strategy": strat,
            "n_questions": len(rows),
            "accuracy": round(float(np.mean([r["grades"][strat]["correct"] for r in rows])), 4),
            "mean_input_tokens": round(float(np.mean([sum(c["input_tokens"] for c in calls(r)) for r in rows])), 1),
            "mean_output_tokens": round(float(np.mean([sum(c["output_tokens"] for c in calls(r)) for r in rows])), 1),
            "mean_seconds": round(float(np.mean([sum(c["seconds"] for c in calls(r)) for r in rows])), 2),
            "truncation_rate": round(float(np.mean([c["truncated"] for r in rows for c in calls(r)])), 4),
            "extraction_failure_rate": round(float(np.mean([(c["extracted"] is None) for r in rows for c in ([final(r)] if final else r["calls"]["parallel"])])), 4),
            "peak_gpu_gb": round(max((r["peak_mem_bytes"] or 0) for r in rows) / 1e9, 2)
        })

    trans= {
        "initial_accuracy": float(np.mean([r["grades"]["parallel"]["initial_correct"] for r in rows])),
        "wrong_to_correct": int(sum((not r["grades"]["parallel"]["initial_correct"]) and r["grades"]["refine"]["correct"] for r in rows)),
        "correct_to_wrong": int(sum(r["grades"]["parallel"]["initial_correct"] and not r["grades"]["refine"]["correct"] for r in rows)),
        "revision_cases": [{"sample_id": r["sample_id"], "initial": r["grades"]["parallel"]["initial_answer"], "revised": r["grades"]["refine"]["answer"], "gold": r["gold"]} for r in rows if r["grades"]["parallel"]["initial_correct"] != r["grades"]["refine"]["correct"]]
    }

    return out, trans


def main(args):
    out= args.results_dir
    fig_dir= os.path.join(out, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    split= {}
    for name in DATASETS:
        with open(os.path.join(args.split_dir, f"{name}_split.json")) as f:
            split[name]= json.load(f)

    boot= {"n": args.bootstrap, "seed": 6304}
    table= []
    for name in DATASETS:
        table+= method_rows(per_question(out, name, set(split[name]["eval"])), name, boot)
    write_csv(os.path.join(out, "summary.csv"), table)
    write_csv(os.path.join(out, "methods_n32.csv"), [r for r in table if r["n"] == 32])
    plot_curves(table, fig_dir)

    ad_rows, ad= adaptive_rows(out, table)
    write_csv(os.path.join(out, "adaptive_vs_fixed.csv"), ad_rows)
    plot_adaptive(ad_rows, fig_dir)

    sig_rows, sig= signal_rows(out)
    write_csv(os.path.join(out, "signals.csv"), sig_rows)

    cases= case_lists(out, ad, split)
    live, trans= live_rows(out)
    if live:
        write_csv(os.path.join(out, "live_summary.csv"), live)
        cases["revision"]= trans
    write_json(os.path.join(out, "cases.json"), cases)

    return 0


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/task3"
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default="splits/task3"
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000
    )

    raise SystemExit(main(parser.parse_args()))
