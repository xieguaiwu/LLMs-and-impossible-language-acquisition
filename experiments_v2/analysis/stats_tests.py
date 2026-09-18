"""Corrected pairwise statistics on per-seed aggregates (v2).

For each (dataset, model, metric) cell, the three pairwise comparisons
(natural-vs-reversed, natural-vs-parity_negation, reversed-vs-parity_negation)
are tested with:

  step 1  Shapiro-Wilk per group (n>=5 per seed; note low power at n=5 --
          we therefore also always report the non-parametric fallback)
  step 2  Levene's test for variance homogeneity
  step 3  Welch's t (default) or Mann-Whitney U (any group non-normal)
  step 4  Holm-Bonferroni across the 3 comparisons in the family
  step 5  Cohen's d with 95% bootstrap CI over seeds
  step 6  TOST equivalence for every comparison (equivalence bound d=0.8,
          the conventional 'large' threshold) -- used to argue LACK of a
          bias (e.g. the LSTM null) with a proper acceptance criterion
          instead of 'p > 0.05, therefore no difference'.

Outputs one tidy CSV: one row per (dataset, model, metric, comparison).
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

METRICS = ["final_loss", "min_loss", "auc_loss", "convergence_frac", "test_loss", "test_ppl"]
EQUIV_BOUND_D = 0.8
N_BOOT = 10000
CONDITION_ORDER = ["natural", "reversed", "parity_negation",
                   "fixed_start_neg", "fixed_end_neg",
                   "parity_negation_tok", "parity_negation_negtok", "word_shuffle"]


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return (a.mean() - b.mean()) / pooled


def d_bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n_boot):
        aa = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        ds.append(cohens_d(aa, bb))
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def holm_bonferroni(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (monotonicity enforced)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running = max(running, adj)
        adjusted[idx] = min(1.0, running)
    return list(adjusted)


def tost_welch(a: np.ndarray, b: np.ndarray, bound_d: float = EQUIV_BOUND_D) -> dict:
    """Two one-sided tests via Welch SE against equivalence margin delta = bound_d * pooled SD."""
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    delta = bound_d * pooled
    diff = a.mean() - b.mean()
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)) if (va > 0 or vb > 0) else 1.0
    t_low = (diff + delta) / se if se > 0 else np.inf
    t_high = (diff - delta) / se if se > 0 else np.inf
    p_low = stats.t.sf(t_low, df)          # H0: diff <= -delta
    p_high = stats.t.cdf(t_high, df)       # H0: diff >= +delta
    return {
        "tost_p1": float(p_low), "tost_p2": float(p_high),
        "tost_p": float(max(p_low, p_high)),
        "equivalent_at_d0.8": bool(max(p_low, p_high) < 0.05),
    }


def test_cell(cell: pd.DataFrame, metric: str) -> list[dict]:
    conds = [c for c in CONDITION_ORDER if c in set(cell["condition"])]
    rows = []
    for c1, c2 in itertools.combinations(conds, 2):
        a = cell.loc[cell["condition"] == c1, metric].dropna().values.astype(float)
        b = cell.loc[cell["condition"] == c2, metric].dropna().values.astype(float)
        if len(a) < 2 or len(b) < 2:
            continue
        sw_a = stats.shapiro(a) if len(a) >= 3 else None
        sw_b = stats.shapiro(b) if len(b) >= 3 else None
        normal = (sw_a.pvalue > 0.05 if sw_a else True) and (sw_b.pvalue > 0.05 if sw_b else True)
        lev = stats.levene(a, b)
        if normal:
            t, p = stats.ttest_ind(a, b, equal_var=False)
            test_name = "welch_t"
            stat = t
        else:
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            test_name = "mann_whitney_u"
            stat = u
        d = cohens_d(a, b)
        lo, hi = d_bootstrap_ci(a, b)
        tost = tost_welch(a, b)
        rows.append({
            "metric": metric,
            "cond1": c1, "cond2": c2,
            "n1": len(a), "n2": len(b),
            "mean1": round(float(a.mean()), 5), "sd1": round(float(a.std(ddof=1)), 5),
            "mean2": round(float(b.mean()), 5), "sd2": round(float(b.std(ddof=1)), 5),
            "shapiro_p1": None if sw_a is None else round(float(sw_a.pvalue), 5),
            "shapiro_p2": None if sw_b is None else round(float(sw_b.pvalue), 5),
            "normal": bool(normal),
            "levene_p": round(float(lev.pvalue), 5),
            "test": test_name, "statistic": round(float(stat), 4),
            "p_raw": float(p),
            "cohens_d": round(d, 4), "d_ci95_low": round(lo, 4), "d_ci95_high": round(hi, 4),
            **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in tost.items()},
            "_pair": (c1, c2),
        })
    # Holm within this family
    if rows:
        adjusted = holm_bonferroni([r["p_raw"] for r in rows])
        for r, adj in zip(rows, adjusted):
            r["p_holm"] = float(adj)  # full precision: rounded values can violate p_holm >= p_raw
            r["significant_holm"] = bool(adj < 0.05)
    for r in rows:
        r.pop("_pair")
    return rows


def run_tests(per_seed_csv: Path, out_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(per_seed_csv)
    out_rows = []
    for (dataset, model), grp in df.groupby(["dataset", "model"]):
        for metric in METRICS:
            if metric not in grp.columns or grp[metric].dropna().empty:
                continue
            rows = test_cell(grp, metric)
            for r in rows:
                r.update({"dataset": dataset, "model": model})
            out_rows.extend(rows)
    out = pd.DataFrame(out_rows)
    out.to_csv(out_csv, index=False)
    return out


if __name__ == "__main__":
    import sys

    results_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results"
    agg = results_root / "aggregated"
    out = run_tests(agg / "per_seed_summary.csv", agg / "holm_corrected_tests.csv")
    print(f"{len(out)} comparisons -> {agg/'holm_corrected_tests.csv'}")
    if len(out):
        cols = ["dataset", "model", "metric", "cond1", "cond2", "test",
                "p_raw", "p_holm", "significant_holm", "cohens_d",
                "d_ci95_low", "d_ci95_high", "tost_p", "equivalent_at_d0.8"]
        print(out[cols].to_string(index=False))
