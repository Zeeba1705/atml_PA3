import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from common.io import load_yaml, read_jsonl
from common.sync import make_api, pull_file

BASE_CONFIG= "configs/task2/base.yaml"
RUN_FILES= ["train_log.jsonl", "dev_metrics.jsonl", "train_summary.json", "config.json", "eval/metrics.json", "eval/examples.json"]
CONDITIONS= ["correct", "complementary", "mismatched", "no_image"]
SUITES= {"vqa": "VQAv2 (soft accuracy)", "grounding": "Grounding (exact match)"}
#feature caching ran in a separate session before training; its wall-clock comes from that session's log
#(features step 451.6 s -> 3811.6 s, 9,226 images, cpu session, download-bound at 2.7 images/s)
FEATURE_CACHING= {"images": 9226, "seconds": 3360.0, "device": "cpu (kaggle session without accelerator)", "note": "image download bound, 2.7 img/s"}
#fixed colours per image condition (dataviz palette slots), the same in every figure
COLORS= {"correct": "#2a78d6", "complementary": "#eb6834", "mismatched": "#1baf7a", "no_image": "#8c8b85", "train": "#2a78d6", "vqa": "#e87ba4", "grounding": "#6250d6"}
INK= "#2f2e2b"
MUTED= "#7a7972"
GRID= "#e6e5e0"
SURFACE= "#fcfcfb"
SMOOTH_WINDOW= 25


def fetch(api, args, rel):
    path= os.path.join(args.results_dir, args.run_name, rel)
    if not os.path.exists(path):
        assert api is not None, f"{path} missing and no --sync_repo"
        ok= pull_file(api, args.sync_repo, f"task2-{args.run_name}", f"results/task2/{args.run_name}/{rel}", path, os.path.join(args.results_dir, ".staging"))
        assert ok, rel

    return path


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


def moving_average(x, window):
    c= np.cumsum(np.insert(np.asarray(x, dtype=float), 0, 0.0))
    n= np.arange(1, len(x) + 1)
    lo= np.maximum(n - window, 0)

    return (c[n] - c[lo]) / (n - lo)


def stage_lines(ax, ends):
    #stage boundaries (end of stage 1 and 2) as thin vertical rules, labelled once at the top
    for e, label in zip(ends[:-1], ["end of stage 1", "end of stage 2"]):
        ax.axvline(e, color=MUTED, linewidth=1, linestyle=":")
        ax.annotate(label, (e, 1), xycoords=("data", "axes fraction"), xytext=(3, -10), textcoords="offset points", fontsize=7, color=MUTED)


def plot_learning(log, dev, ends, out_dir):
    #five panels, one quantity each (different scales): train loss, dev caption loss, vqa nll, grounding nll, dNLL
    fig, axes= plt.subplots(1, 5, figsize=(20, 3.8), facecolor=SURFACE)
    u= [r["update"] for r in log]
    du= [r["update"] for r in dev]

    ax= axes[0]
    style(ax)
    ax.plot(u, moving_average([r["loss"] for r in log], SMOOTH_WINDOW), color=COLORS["train"], linewidth=1.6)
    ax.set_title(f"training loss ({SMOOTH_WINDOW}-update mov. avg.)", color=INK, loc="left", fontsize=10)

    ax= axes[1]
    style(ax)
    ax.plot(du, [r["caption_loss"] for r in dev], color=COLORS["train"], linewidth=2, marker="o", markersize=4)
    ax.set_title("dev caption reference loss", color=INK, loc="left", fontsize=10)

    for ax, suite in zip(axes[2:4], ["vqa", "grounding"]):
        style(ax)
        for c in ["correct", "complementary", "no_image"]:
            ax.plot(du, [r[f"{suite}_nll_{c}"] for r in dev], color=COLORS[c], linewidth=2, marker="o", markersize=4, label=c.replace("_", " "))
        ax.set_title(f"dev answer NLL: {suite}", color=INK, loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)

    ax= axes[4]
    style(ax)
    for suite in ["vqa", "grounding"]:
        ax.plot(du, [r[f"{suite}_delta_nll"] for r in dev], color=COLORS[suite], linewidth=2, marker="o", markersize=4, label=suite)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_title("image-swap penalty ΔNLL", color=INK, loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)

    for ax in axes:
        stage_lines(ax, ends)
        ax.set_xlabel("optimizer update", color=INK)
    fig.tight_layout()
    save(fig, out_dir, "learning_curves")


def plot_final(metrics, out_dir):
    #accuracy per image condition, both suites, by question type (dots; y from 0 so lengths are not implied)
    fig, axes= plt.subplots(1, 2, figsize=(12, 4), facecolor=SURFACE)
    for ax, suite in zip(axes, SUITES):
        style(ax)
        types= [t for t in metrics[suite]["correct"] if t != "all"]
        groups= ["all"] + types
        x= np.arange(len(groups))
        for k, c in enumerate(CONDITIONS):
            ax.scatter(x + (k - 1.5) * 0.16, [metrics[suite][c][g]["accuracy"] for g in groups], s=55, color=COLORS[c], edgecolors=SURFACE, linewidths=2, zorder=3, label=c.replace("_", " "))
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=9)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(SUITES[suite] + ", final test", color=INK, loc="left", fontsize=10)
    axes[0].set_ylabel("accuracy", color=INK)
    #shared legend under the panels: inside, it covered the presence/correct point
    handles, labels= axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=INK, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, out_dir, "final_accuracy_by_condition")


def main(args):
    cfg= load_yaml(BASE_CONFIG)
    api= make_api() if args.sync_repo else None
    paths= {f: fetch(api, args, f) for f in RUN_FILES}

    log= read_jsonl(paths["train_log.jsonl"])
    dev= read_jsonl(paths["dev_metrics.jsonl"])
    with open(paths["train_summary.json"]) as f:
        summary= json.load(f)
    with open(paths["eval/metrics.json"]) as f:
        ev= json.load(f)

    out= args.results_dir
    fig_dir= os.path.join(out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    ends= list(np.cumsum([s["updates"] for s in cfg["stages"]]))

    #dev rows are logged at the start of the next loop iteration: label each by the model's stage at that update
    def stage_of(u):
        return next(s["name"] for s, e in zip(cfg["stages"], ends) if u <= e)

    write_csv(os.path.join(out, "dev_metrics.csv"), [{"update": r["update"], "model_stage": "init" if r["update"] == 0 else stage_of(r["update"]), **{k: round(v, 4) for k, v in r.items() if k not in ("update", "stage")}} for r in dev])
    plot_learning(log, dev, ends, fig_dir)

    final= []
    for suite in SUITES:
        for c in CONDITIONS:
            for t, m in ev["metrics"][suite][c].items():
                final.append({"suite": suite, "condition": c, "type": t, **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}})
    write_csv(os.path.join(out, "final_results.csv"), final)
    plot_final(ev["metrics"], fig_dir)

    adv= []
    for suite in SUITES:
        for metric, comps in ev["advantages"][suite].items():
            for comp, v in comps.items():
                adv.append({"suite": suite, "metric": metric, "comparison": comp, "mean": round(v["mean"], 4), "ci95_lo": round(v["lo"], 4), "ci95_hi": round(v["hi"], 4), "bootstrap_resamples": cfg["bootstrap_resamples"]})
    write_csv(os.path.join(out, "advantages.csv"), adv)

    gen_time= {s: round(ev["metrics"][s]["correct"]["all"]["gen_seconds_per_answer"], 4) for s in SUITES}
    p= summary["progress"]
    resources= [
        {"item": "trainable params, stage 1 (connector)", "value": p["trainable_params"]["stage1"]},
        {"item": "trainable params, stages 2-3 (connector + LoRA)", "value": p["trainable_params"]["stage2"]},
        {"item": "training time (h), all 4,500 updates incl. dev evals", "value": round(p["train_seconds"] / 3600, 3)},
        {"item": "training peak GPU memory (GB)", "value": round(summary["peak_mem_bytes_last_session"] / 1e9, 2)},
        {"item": "training GPU", "value": summary["gpu"]},
        {"item": "eval peak GPU memory (GB)", "value": round((ev["info"]["peak_mem_bytes"] or 0) / 1e9, 2)},
        {"item": "generation time per answer, vqa / grounding (s, batch 8)", "value": f"{gen_time['vqa']} / {gen_time['grounding']}"},
        {"item": "feature caching (images / seconds / device)", "value": f"{FEATURE_CACHING['images']} / {FEATURE_CACHING['seconds']:.0f} / {FEATURE_CACHING['device']}"},
        {"item": "connector output scale (calibration)", "value": round(p["calibration"]["scale"], 5)},
        {"item": "draws per component", "value": json.dumps(p["counts"])}
    ]
    write_csv(os.path.join(out, "resources.csv"), resources)

    return 0


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--run_name",
        type=str,
        default="main"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/task2"
    )
    parser.add_argument(
        "--sync_repo",
        type=str,
        default=None
    )

    raise SystemExit(main(parser.parse_args()))
