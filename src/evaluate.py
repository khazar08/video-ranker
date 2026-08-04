"""Aggregate per-experiment result JSONs into a table + plots.

    python -m src.evaluate --results-dir results

Produces:
    results/metrics.csv        headline metrics (test pool, random negatives)
    results/metrics_full.csv   every experiment x eval-slice, long form
    results/results_table.md   markdown table (pasted into the README)
    results/ndcg_by_k.png      NDCG@k vs k, one line per experiment
    results/ablation_losses.png listwise vs pairwise neural-ranker ablation
    results/retrieval_recall.png Recall@100/@200 per retriever (stage-1 ceiling)
    results/cold_vs_active.png  NDCG@10 on cold-start vs active user slices
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Pre-validated categorical palette (from the dataviz skill reference).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SLICES = ["random", "hard", "cold", "active"]


def load_results(results_dir: Path) -> List[dict]:
    rows = []
    for p in sorted(glob.glob(str(results_dir / "*.json"))):
        with open(p) as f:
            rows.append(json.load(f))
    if not rows:
        raise SystemExit(f"No result JSONs in {results_dir}. Run src.train first.")
    return rows


def _headline_frame(results: List[dict], ks=(5, 10, 20)) -> pd.DataFrame:
    """One row per experiment: retrieval recall + random-slice ranking metrics."""
    out = []
    for r in results:
        m = r["metrics_random"]
        row = {
            "experiment": r["experiment"],
            "retriever": r["retriever"],
            "ranker": r["ranker"] + (f":{r['loss']}" if r.get("loss") else ""),
            "loss_family": r.get("loss_family") or "-",
            "recall@100": r["retrieval"].get("recall@100"),
            "recall@200": r["retrieval"].get("recall@200"),
        }
        for k in ks:
            for name in ("ndcg", "mrr", "map", "recall"):
                row[f"{name}@{k}"] = m.get(f"{name}@{k}")
        out.append(row)
    return pd.DataFrame(out).sort_values("ndcg@10", ascending=False)


def _long_frame(results: List[dict], ks=(5, 10, 20)) -> pd.DataFrame:
    """Long form: (experiment, slice, metric@k) -> value, for all slices."""
    out = []
    for r in results:
        for sl in SLICES:
            m = r.get(f"metrics_{sl}", {})
            for k in ks:
                for name in ("ndcg", "mrr", "map", "recall"):
                    key = f"{name}@{k}"
                    if key in m:
                        out.append({
                            "experiment": r["experiment"], "slice": sl,
                            "metric": name, "k": k, "value": m[key],
                        })
    return pd.DataFrame(out)


def _markdown_table(df: pd.DataFrame) -> str:
    cols = ["experiment", "recall@200", "ndcg@5", "ndcg@10", "ndcg@20",
            "mrr@10", "map@10", "recall@10"]
    cols = [c for c in cols if c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --- plots ------------------------------------------------------------------

def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 130, "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "axes.axisbelow": True,
    })
    return plt


def plot_ndcg_by_k(results, out_path, ks=(5, 10, 20)):
    plt = _setup_mpl()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, r in enumerate(sorted(results, key=lambda r: -r["metrics_random"]["ndcg@10"])):
        m = r["metrics_random"]
        ys = [m[f"ndcg@{k}"] for k in ks]
        c = PALETTE[i % len(PALETTE)]
        ax.plot(ks, ys, marker="o", ms=6, lw=2, color=c, label=r["experiment"])
        ax.annotate(f"{ys[-1]:.3f}", (ks[-1], ys[-1]), textcoords="offset points",
                    xytext=(6, 0), va="center", fontsize=9, color=c)
    ax.set_xlabel("k"); ax.set_ylabel("NDCG@k")
    ax.set_title("Ranking quality vs cutoff (test pool, random negatives)")
    ax.set_xticks(list(ks))
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


def plot_loss_ablation(results, out_path):
    """NDCG@10 across neural-ranker losses, colored by listwise/pairwise family."""
    neu = [r for r in results if r["ranker"] == "neural" and r.get("loss")]
    if not neu:
        return False
    plt = _setup_mpl()
    order = {"listnet": 0, "listmle": 1, "ranknet": 2, "bpr": 3}
    neu = sorted(neu, key=lambda r: order.get(r["loss"], 9))
    labels = [r["loss"] for r in neu]
    vals = [r["metrics_random"]["ndcg@10"] for r in neu]
    fams = [r.get("loss_family") for r in neu]
    fam_color = {"listwise": PALETTE[0], "pairwise": PALETTE[1]}
    colors = [fam_color.get(f, PALETTE[2]) for f in fams]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, width=0.6, color=colors, zorder=3)
    for xi, v in zip(x, vals):
        ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("NDCG@10")
    ax.set_title("Neural ranker: listwise vs pairwise loss ablation")
    # legend by family
    from matplotlib.patches import Patch
    handles = [Patch(color=fam_color["listwise"], label="listwise"),
               Patch(color=fam_color["pairwise"], label="pairwise")]
    ax.legend(handles=handles, frameon=False, fontsize=9)
    ax.set_ylim(0, max(vals) * 1.18)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    return True


def plot_retrieval_recall(results, out_path):
    # unique retrievers -> recall@100/@200
    seen = {}
    for r in results:
        seen.setdefault(r["retriever"], r["retrieval"])
    plt = _setup_mpl()
    retrievers = list(seen)
    x = np.arange(len(retrievers))
    w = 0.36
    r100 = [seen[t].get("recall@100", 0) for t in retrievers]
    r200 = [seen[t].get("recall@200", 0) for t in retrievers]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.bar(x - w / 2, r100, w, color=PALETTE[0], label="Recall@100", zorder=3)
    ax.bar(x + w / 2, r200, w, color=PALETTE[2], label="Recall@200", zorder=3)
    for xi, v in zip(x - w / 2, r100):
        ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=8)
    for xi, v in zip(x + w / 2, r200):
        ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(retrievers)
    ax.set_ylabel("Recall"); ax.set_title("Stage-1 retrieval ceiling")
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0, max(r200 + [0.01]) * 1.2)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


def plot_cold_vs_active(results, out_path):
    plt = _setup_mpl()
    results = sorted(results, key=lambda r: -r["metrics_random"]["ndcg@10"])
    labels = [r["experiment"] for r in results]
    cold = [r.get("metrics_cold", {}).get("ndcg@10", 0) for r in results]
    active = [r.get("metrics_active", {}).get("ndcg@10", 0) for r in results]
    x = np.arange(len(labels)); w = 0.36
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.3), 4.4))
    ax.bar(x - w / 2, cold, w, color=PALETTE[3], label="cold-start users", zorder=3)
    ax.bar(x + w / 2, active, w, color=PALETTE[0], label="active users", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("NDCG@10")
    ax.set_title("Cold-start vs active users (NDCG@10)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()
    results_dir = Path(args.results_dir)
    results = load_results(results_dir)

    headline = _headline_frame(results)
    long = _long_frame(results)
    headline.to_csv(results_dir / "metrics.csv", index=False)
    long.to_csv(results_dir / "metrics_full.csv", index=False)

    md = _markdown_table(headline)
    (results_dir / "results_table.md").write_text(md + "\n")
    print(md)

    plot_ndcg_by_k(results, results_dir / "ndcg_by_k.png")
    plot_loss_ablation(results, results_dir / "ablation_losses.png")
    plot_retrieval_recall(results, results_dir / "retrieval_recall.png")
    plot_cold_vs_active(results, results_dir / "cold_vs_active.png")
    print(f"\nWrote metrics.csv, metrics_full.csv, results_table.md and 4 plots "
          f"to {results_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
