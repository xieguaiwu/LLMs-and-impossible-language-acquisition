"""Per-seed aggregation + corrected statistical tests (v2).

Fixes the two documented statistical flaws of the original analysis
(statistics/EXPDESIGN.md section 4.1):

1. time-series pseudo-replication: the unit of analysis becomes ONE SCALAR
   PER SEED (from the ``summary`` block of each run JSON), not each logged
   training step;
2. no multiple-comparison correction: Holm-Bonferroni is applied within each
   (experiment x metric) family of 3 pairwise comparisons.

Pipeline
--------
aggregate_seeds.py : experiments_v2/results/**/seed*/**/training_metrics.json
                     -> aggregated/all_runs.csv (one row per run)
                        + aggregated/per_seed_summary.csv (one row per seed)
stats_tests.py     : per_seed_summary.csv -> holm_corrected_tests.csv with
                     Shapiro-Wilk, Levene, Welch t / Mann-Whitney U, Holm
                     adjusted p, Cohen's d with bootstrap CI, and (for
                     architecture null claims) TOST equivalence bounds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

SUMMARY_KEYS = [
    "final_loss", "min_loss", "final_ppl", "min_ppl", "auc_loss",
    "convergence_step", "convergence_frac", "test_loss", "test_ppl",
]


def collect_runs(results_root: Path) -> pd.DataFrame:
    rows = []
    for record_file in sorted(results_root.rglob("training_metrics.json")):
        with open(record_file, encoding="utf-8") as f:
            rec = json.load(f)
        if "summary" not in rec:  # old-format file from the original n=1 runs
            continue
        row = {
            "run_id": rec.get("run_id", record_file.parent.name),
            "experiment": rec.get("experiment"),
            "model": rec.get("model"),
            "dataset": rec.get("dataset"),
            "condition": rec.get("condition"),
            "seed": rec.get("seed"),
        }
        for k in SUMMARY_KEYS:
            row[k] = rec["summary"].get(k)
        row["path"] = str(record_file)
        rows.append(row)
    return pd.DataFrame(rows)


def per_seed_summary(all_runs: pd.DataFrame) -> pd.DataFrame:
    return all_runs.copy()  # one row per run == one row per seed (seed is unique per cell)


def load_results(results_root: Path) -> pd.DataFrame:
    df = collect_runs(results_root)
    if df.empty:
        raise SystemExit(f"no v2 run JSONs with summary blocks under {results_root}")
    return df


if __name__ == "__main__":
    results_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results"
    out_dir = results_root / "aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = collect_runs(results_root)
    df.to_csv(out_dir / "all_runs.csv", index=False)
    per_seed_summary(df).to_csv(out_dir / "per_seed_summary.csv", index=False)
    print(f"runs: {len(df)}  cells: {df.groupby(['dataset','model','condition']).ngroups}")
    print(f"wrote {out_dir/'all_runs.csv'} and per_seed_summary.csv")
