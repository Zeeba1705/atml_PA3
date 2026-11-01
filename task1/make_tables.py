import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from common.io import load_yaml, read_jsonl
from common.sync import make_api, pull_file
from task1.diagnostics import diag_branch, diag_root
from task1.evaluate import EVAL_CONFIG, bench_paths, eval_root, sync_branch

RUNS= ["full_ft", "l2sp", "lora4", "lora16", "oplora16"]
CONDITIONS= ["baseline", "full_ft_mid", "full_ft", "l2sp", "lora4", "lora16", "oplora16"]
DIAG_FILES= ["weight_metrics.csv", "activation_summary.csv", "diagnostics_info.json"]

#fixed categorical order (dataviz reference palette slots 1-5): colour follows the run, never its rank.
#full_ft_mid = a lighter step of full_ft's blue, baseline = neutral gray
COLORS= {
    "baseline": "#8c8b85",
    "full_ft_mid": "#86b6ef",
    "full_ft": "#2a78d6",
    "l2sp": "#eb6834",
    "lora4": "#1baf7a",
    "lora16": "#eda100",
    "oplora16": "#e87ba4"
}
LABELS= {
    "baseline": "Baseline",
    "full_ft_mid": "Full FT (625)",
    "full_ft": "Full FT",
    "l2sp": "Full FT + L2-SP",
    "lora4": "LoRA-4",
    "lora16": "LoRA-16",
    "oplora16": "OPLoRA-16"
}
INK= "#2f2e2b"
MUTED= "#7a7972"
GRID= "#e6e5e0"
SURFACE= "#fcfcfb"

#loss curves: trailing moving average over this many updates (raw per-update losses are noisy)
SMOOTH_WINDOW= 25
#right panel of the loss figure starts here, after the steep initial drop
ZOOM_FROM= 100


def fetch(api, args, branch, rel):
    #local results first; otherwise pull the one file from its HF branch
    path= os.path.join(args.results_dir, rel)
    if os.path.exists(path):
        return path

    assert api is not None, f"{path} is missing and there is no --sync_repo to pull it from"
    ok= pull_file(api, args.sync_repo, branch, f"results/{rel}", path, os.path.join(args.results_dir, ".staging"))
    assert ok, f"download failed: results/{rel} on {branch}"

    return path


def load_json(path):
    with open(path) as f:
        return json.load(f)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer= csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("saved", path)


def write_markdown(path, rows):
    #same numbers as the csv, for pasting
    cols= list(rows[0])
    lines= ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join("" if r[c] is None else str(r[c]) for c in cols) + " |")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("saved", path)


def pct(x, digits=2):
    return None if x is None else round(100 * x, digits)


def summary_rows(api, args, eval_cfg):
    #one row per condition x benchmark; delta in percentage points vs the shared baseline
    root= os.path.relpath(eval_root(args.results_dir, False), args.results_dir)
    metrics= {}
    for condition in CONDITIONS:
        for b in eval_cfg["benchmarks"]:
            rel= os.path.relpath(bench_paths(os.path.join(root, condition), b)["metrics"])
            metrics[condition, b]= load_json(fetch(api, args, sync_branch(condition, False), rel))

    rows= []
    for condition in CONDITIONS:
        spec= eval_cfg["conditions"][condition]
        for b, bcfg in eval_cfg["benchmarks"].items():
            m= metrics[condition, b]
            base= metrics["baseline", b]
            raw= m.get("raw")

            rows.append({
                "condition": condition,
                "run": None if spec is None else spec["run"],
                "update": 0 if spec is None else spec["update"],
                "benchmark": b,
                "metric": m["main"]["name"],
                "score_pct": pct(m["main"]["value"]),
                "stderr_pct": pct(m["main"]["stderr"]),
                "delta_pp": pct(m["main"]["value"] - base["main"]["value"]),
                "raw_metric": None if raw is None else raw["name"],
                "raw_pct": None if raw is None else pct(raw["value"]),
                "raw_stderr_pct": None if raw is None else pct(raw["stderr"]),
                "raw_delta_pp": None if raw is None else pct(raw["value"] - base["raw"]["value"]),
                "n": m["n_samples"]["effective"],
                "n_truncated": m.get("n_truncated"),
                "n_extract_failed": m.get("n_extract_failed"),
                "eval_seconds": round(m["seconds"]),
                "peak_vram_gb": None if "peak_mem_bytes" not in m else round(m["peak_mem_bytes"] / 1e9, 2)
            })

    return rows


def wide_rows(rows, eval_cfg):
    #condition x (score, delta) per benchmark, for the report table
    out= []
    for condition in CONDITIONS:
        r= {"condition": LABELS[condition]}
        for b in eval_cfg["benchmarks"]:
            x= next(x for x in rows if x["condition"] == condition and x["benchmark"] == b)
            r[f"{b} (%)"]= f"{x['score_pct']:.2f} ± {x['stderr_pct']:.2f}"
            r[f"{b} Δpp"]= f"{x['delta_pp']:+.2f}"
        out.append(r)

    return out


def train_data(api, args):
    logs, summaries= {}, {}
    for run in RUNS:
        branch= f"task1-{run}"
        logs[run]= read_jsonl(fetch(api, args, branch, f"task1/{run}/train_log.jsonl"))
        summaries[run]= load_json(fetch(api, args, branch, f"task1/{run}/train_summary.json"))

        assert len(logs[run]) == summaries[run]["config"]["num_updates"], f"{run}: incomplete train log"

    return logs, summaries


def resource_rows(summaries):
    #total parameters = full ft's trainable count (every weight trains there)
    total= summaries["full_ft"]["trainable_params"]

    rows= []
    for run in RUNS:
        s= summaries[run]
        p= s["progress"]
        o= s["overhead"]
        final= s["checkpoint_sizes"][str(s["config"]["num_updates"])]
        export= s["exports"][f"update_{s['config']['num_updates']:05d}"]

        rows.append({
            "run": run,
            "method": s["config"]["method"],
            "trainable_params": s["trainable_params"],
            "trainable_pct": round(100 * s["trainable_params"] / total, 3),
            "train_hours": round(p["train_seconds"] / 3600, 3),
            "checkpoint_hours": round(p["ckpt_seconds"] / 3600, 3),
            "peak_vram_gb": round(p["peak_mem_bytes"] / 1e9, 2),
            "checkpoint_trainable_mb": round(final["trainable_bytes"] / 1e6, 1),
            "checkpoint_total_mb": round(final["total_bytes"] / 1e6, 1),
            "export_mb": round(export["bytes"] / 1e6, 1),
            "svd_seconds": None if "svd_seconds" not in o else round(o["svd_seconds"], 1),
            "bases_mb": None if "bases_bytes" not in o else round(o["bases_bytes"] / 1e6, 1),
            "theta0_mb": None if "theta0_bytes" not in o else round(o["theta0_bytes"] / 1e6, 1),
            "skipped_updates": p["skipped_updates"],
            "sessions": p["sessions"],
            "gpu": s["gpu"]
        })

    return rows


def diagnostic_rows(api, args):
    root= os.path.relpath(diag_root(args.results_dir, False), args.results_dir)
    paths= {f: fetch(api, args, diag_branch(False), os.path.join(root, f)) for f in DIAG_FILES}

    with open(paths["weight_metrics.csv"]) as f:
        weights= list(csv.DictReader(f))
    with open(paths["activation_summary.csv"]) as f:
        acts= list(csv.DictReader(f))

    rows= []
    for w in weights:
        r= {
            "condition": w["condition"],
            "layer": int(w["layer"]),
            "k": int(w["k"]),
            "rho_k": round(float(w["rho_k"]), 4),
            "eta": round(float(w["eta"]), 5),
            "fp16_noise_eta": None if w["fp16_noise_eta"] == "" else round(float(w["fp16_noise_eta"]), 5)
        }
        for a in acts:
            if a["condition"] == w["condition"] and a["layer"] == w["layer"]:
                r[f"R_{a['kind']}_mean"]= round(float(a["mean"]), 5)
                r[f"R_{a['kind']}_median"]= round(float(a["median"]), 5)
                r[f"R_{a['kind']}_std"]= round(float(a["std"]), 5)
        rows.append(r)

    order= {c: i for i, c in enumerate(CONDITIONS)}
    rows.sort(key=lambda r: (order[r["condition"]], r["layer"]))

    return rows


def style(ax):
    #recessive chrome: light grid behind the data, no top/right spines, muted tick labels
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)


def moving_average(x, window):
    #trailing mean; the first window-1 points average over what exists so far
    c= np.cumsum(np.insert(np.asarray(x, dtype=float), 0, 0.0))
    n= np.arange(1, len(x) + 1)
    lo= np.maximum(n - window, 0)

    return (c[n] - c[lo]) / (n - lo)


def save(fig, out_dir, name):
    for ext in ["png", "pdf"]:
        path= os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
        print("saved", path)

    plt.close(fig)


def plot_losses(logs, out_dir):
    #left: whole run; right: zoom after the initial drop, where the runs separate. five series, so the legend
    #carries identity (direct labels collided at the line ends)
    fig, axes= plt.subplots(1, 2, figsize=(11, 4), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1.4]})

    for ax, start in zip(axes, [1, ZOOM_FROM]):
        style(ax)
        lo, hi= [], []
        for run in RUNS:
            u= np.array([r["update"] for r in logs[run]])
            loss= moving_average([r["loss"] for r in logs[run]], SMOOTH_WINDOW)
            keep= u >= start
            ax.plot(u[keep], loss[keep], color=COLORS[run], linewidth=2 if ax is axes[0] else 1.6, label=LABELS[run])
            lo.append(loss[keep].min())
            hi.append(loss[keep].max())

        ax.set_xlabel("optimizer update", color=INK)
        if start > 1:
            pad= 0.05 * (max(hi) - min(lo))
            ax.set_ylim(min(lo) - pad, max(hi) + pad)

    axes[0].set_ylabel(f"SFT loss ({SMOOTH_WINDOW}-update moving average)", color=INK)
    axes[0].set_title("whole run", color=INK, loc="left", fontsize=10)
    axes[1].set_title(f"zoom: updates {ZOOM_FROM}-{len(logs[RUNS[0]])}", color=INK, loc="left", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    fig.suptitle("Training loss, all runs", color=INK, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()

    save(fig, out_dir, "sft_loss")


def plot_l2sp(log, out_dir):
    #three scales (loss ~0.2, lambda*omega ~0.02, omega ~20): one panel each, never a second y-axis
    u= [r["update"] for r in log]
    fig, axes= plt.subplots(1, 3, figsize=(11, 3.2), facecolor=SURFACE)

    panels= [
        ("SFT loss and total loss", [("SFT loss", "loss", COLORS["full_ft"]), ("total = SFT + λΩ", "total_loss", COLORS["l2sp"])]),
        ("penalty λΩ", [("λΩ", "lambda_omega", COLORS["l2sp"])]),
        ("Ω (unscaled)", [("Ω", "omega", COLORS["l2sp"])])
    ]
    for ax, (title, series) in zip(axes, panels):
        style(ax)
        for label, key, color in series:
            y= [r[key] for r in log]
            if key in ["loss", "total_loss"]:
                y= moving_average(y, SMOOTH_WINDOW)
            ax.plot(u, y, color=color, linewidth=2, label=label)

        ax.set_title(title, color=INK, loc="left", fontsize=10)
        ax.set_xlabel("optimizer update", color=INK)
        if len(series) > 1:
            ax.legend(frameon=False, fontsize=8, labelcolor=INK)

    axes[0].set_ylabel(f"loss ({SMOOTH_WINDOW}-update moving avg.)", color=INK)
    fig.suptitle("Full FT + L2-SP: objective terms", color=INK, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()

    save(fig, out_dir, "l2sp_terms")


def plot_benchmarks(rows, eval_cfg, out_dir):
    #one panel per benchmark (different scales). dots + 95% CI (1.96 stderr), not bars: the axes are zoomed to
    #the data, which bars (length from 0) must not be. dashed line = baseline score
    benchmarks= list(eval_cfg["benchmarks"])
    fig, axes= plt.subplots(1, len(benchmarks), figsize=(13, 3.8), facecolor=SURFACE)

    for ax, b in zip(axes, benchmarks):
        style(ax)
        ax.grid(axis="x", visible=False)
        sub= [next(r for r in rows if r["condition"] == c and r["benchmark"] == b) for c in CONDITIONS]
        x= np.arange(len(CONDITIONS))

        ax.axhline(sub[0]["score_pct"], color=MUTED, linewidth=1, linestyle="--", zorder=1)
        for xi, c, r in zip(x, CONDITIONS, sub):
            ax.errorbar(xi, r["score_pct"], yerr=1.96 * r["stderr_pct"], fmt="none", ecolor=COLORS[c], elinewidth=2, capsize=0, zorder=2)
            #surface ring keeps the marker readable on top of its interval
            ax.scatter(xi, r["score_pct"], s=60, color=COLORS[c], edgecolors=SURFACE, linewidths=2, zorder=3)

        ax.set_xlim(-0.6, len(CONDITIONS) - 0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[c] for c in CONDITIONS], rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{b} ({sub[0]['metric'].split(',')[0]})", color=INK, loc="left", fontsize=10)

    axes[0].set_ylabel("score (%), 95% CI", color=INK)
    fig.suptitle("Benchmark scores (dashed = baseline; y-axes differ per panel)", color=INK, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()

    save(fig, out_dir, "benchmark_scores")


def main(args):
    eval_cfg= load_yaml(EVAL_CONFIG)
    api= make_api() if args.sync_repo else None

    out= os.path.join(args.results_dir, "task1")
    fig_dir= os.path.join(out, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    rows= summary_rows(api, args, eval_cfg)
    write_csv(os.path.join(out, "summary.csv"), rows)
    wide= wide_rows(rows, eval_cfg)
    write_csv(os.path.join(out, "summary_wide.csv"), wide)
    write_markdown(os.path.join(out, "summary_wide.md"), wide)

    logs, summaries= train_data(api, args)
    resources= resource_rows(summaries)
    write_csv(os.path.join(out, "resources.csv"), resources)
    write_markdown(os.path.join(out, "resources.md"), resources)

    diag= diagnostic_rows(api, args)
    write_csv(os.path.join(out, "diagnostics.csv"), diag)
    write_markdown(os.path.join(out, "diagnostics.md"), diag)

    plot_losses(logs, fig_dir)
    plot_l2sp(logs["l2sp"], fig_dir)
    plot_benchmarks(rows, eval_cfg, fig_dir)

    return 0


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--results_dir",
        type=str,
        default="results"
    )
    parser.add_argument(
        "--sync_repo",
        type=str,
        default=None
    )

    raise SystemExit(main(parser.parse_args()))
