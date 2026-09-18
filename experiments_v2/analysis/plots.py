"""Seed-aware plots (v2): mean curves with seed-level 95% CI bands.

Replaces the original single-run overlay plots. For every
(dataset, model, condition) cell the per-step loss series across seeds is
resampled to a common step grid, plotted as mean +/- 95% CI across seeds.
Also plots per-seed summary scalars (final/min loss, test PPL) as box plots.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:  # matplotlib >= 3.9 renamed labels -> tick_labels
    _make_boxplot = lambda ax, data, conds: ax.boxplot(data, tick_labels=conds, patch_artist=True)
except TypeError:  # pragma: no cover
    _make_boxplot = lambda ax, data, conds: ax.boxplot(data, labels=conds, patch_artist=True)


def _boxplot_compat(ax, data, conds):
    try:
        return ax.boxplot(data, tick_labels=conds, patch_artist=True)
    except TypeError:
        return ax.boxplot(data, labels=conds, patch_artist=True)

COLORS = {
    "natural": "#1f77b4", "reversed": "#d62728", "parity_negation": "#2ca02c",
    "fixed_start_neg": "#9467bd", "fixed_end_neg": "#8c564b",
    "parity_negation_tok": "#e377c2", "parity_negation_negtok": "#7f7f7f",
    "word_shuffle": "#bcbd22",
}


def load_series(results_root: Path) -> dict:
    """-> {(dataset, model, condition): {seed: losses}}"""
    out = defaultdict(dict)
    for f in sorted(results_root.rglob("training_metrics.json")):
        with open(f, encoding="utf-8") as fh:
            rec = json.load(fh)
        if "summary" not in rec:
            continue
        key = (rec["dataset"], rec["model"], rec["condition"])
        out[key][rec["seed"]] = rec["losses"]
    return out


def resample(losses: list[float], grid: np.ndarray) -> np.ndarray:
    x = np.linspace(0, 1, len(losses))
    return np.interp(grid, x, losses)


def plot_curves(series: dict, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    grid = np.linspace(0, 1, 100)
    for condition, by_seed in sorted(series.items()):
        mats = np.array([resample(l, grid) for l in by_seed.values()])
        mean, sem = mats.mean(0), mats.std(0, ddof=1) / np.sqrt(mats.shape[0])
        ci = 1.96 * sem
        c = COLORS.get(condition)
        ax.plot(grid * 100, mean, label=condition, color=c, lw=2)
        ax.fill_between(grid * 100, mean - ci, mean + ci, color=c, alpha=0.18, lw=0)
    ax.set_xlabel("training progress (%)")
    ax.set_ylabel("loss")
    ax.set_title(title + "  (mean ± 95% CI over seeds)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_summaries(df: pd.DataFrame, out_dir: Path) -> None:
    for metric in ["final_loss", "test_ppl"]:
        for (dataset, model), grp in df.groupby(["dataset", "model"]):
            fig, ax = plt.subplots(figsize=(8, 5))
            conds = sorted(grp["condition"].unique())
            data = [grp.loc[grp["condition"] == c, metric].dropna().values for c in conds]
            bp = _boxplot_compat(ax, data, conds)
            for patch, c in zip(bp["boxes"], conds):
                patch.set_facecolor(COLORS.get(c, "#cccccc"))
                patch.set_alpha(0.7)
            ax.set_title(f"{dataset} / {model} — {metric} per seed")
            ax.grid(alpha=0.3, axis="y")
            fig.tight_layout()
            fig.savefig(out_dir / f"box_{dataset}_{model}_{metric}.png", dpi=200)
            plt.close(fig)


def main() -> None:
    results_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results"
    out_dir = results_root / "aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)

    series = load_series(results_root)
    for (dataset, model, condition), by_seed in series.items():
        if condition == "natural":
            continue  # plotted as reference with each group below
    groups = defaultdict(dict)
    for (dataset, model, condition), by_seed in series.items():
        groups[(dataset, model)][condition] = by_seed
    for (dataset, model), conds in groups.items():
        plot_curves(conds, out_dir / f"curves_{dataset}_{model}.png",
                    f"{dataset} / {model}")

    from aggregate_seeds import collect_runs

    df = collect_runs(results_root)
    plot_summaries(df, out_dir)
    print(f"plots -> {out_dir}")


if __name__ == "__main__":
    main()
