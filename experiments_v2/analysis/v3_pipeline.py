#!/usr/bin/env python3
"""v3_pipeline.py — pre-registered analysis for the v3 BabyLM grid.

Implements STATS_PLAN_V3 for the frozen families plus the amendments registered
on 2026-09-20 (design audit, pre-P-class-data):

  * primary metric = `ppl_gmean_final` on the **duplicate-free** subset of the
    evaluation draw. The draw itself is the frozen per-condition numpy draw over
    each condition's own test pool (so the CPU-LSTM cells, whose weights were
    discarded, stay comparable); duplicates are removed at metric time via the
    sentence-id vector recomputed from the condition file (audit B5). The
    unfiltered value and the content-token-only (marker-masked) value are
    reported as sensitivity columns.
  * H9 keeps its replication content but is evaluated **within class** (the
    cross-class raw-ppl rank vector contradicts DESIGN_V3 §A.2 / REDTEAM #2,
    which forbid raw ppl ordering across marker families); the original
    cross-class Kendall tau is still computed and published descriptively.
  * F1..F5 = the frozen Holm families, with the seed extensions of
    2026-09-20 (`shuffle_control`, `reverse_full`, `parity_word`, `fixed_start`,
    `parity_tok`, `negtok` at seeds 53/96; `fixed_end` at 0/14/41).

USAGE
    python3 experiments_v2/analysis/v3_pipeline.py --results DIR [--lstm-dir DIR]
        [--out DIR] [--selftest]

Outputs (all rows published, significant or not):
    aggregated/checkpoint_evals_v3.csv
    aggregated/per_seed_v3.csv
    aggregated/stats_tests_v3.csv
    aggregated/h9_replication.csv
    aggregated/marker_entropy.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments_v2" / "kallini_repro"))

GENRES = ["aochildes", "bnc_spoken", "cbt", "children_stories", "gutenberg",
          "open_subtitles", "qed", "simple_wikipedia", "switchboard", "wikipedia"]

# ---------------------------------------------------------------- statistics ---
# Ported/verified against STATS_PLAN_V3 §2/§5/§6. scipy is optional: without it
# the parametric tests fall back to a pure-python implementation and the
# rank-based tests are skipped (the CSV records `test=unavailable`).

def _mean(x):
    return float(np.mean(x))


def _sd(x):
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def paired_t(diffs) -> tuple[float, float]:
    """Two-sided paired t on the difference vector; returns (t, p)."""
    d = np.asarray(diffs, dtype=float)
    n = d.size
    sd = d.std(ddof=1)
    if sd == 0:
        return (math.inf if d.mean() != 0 else 0.0, 0.0 if d.mean() != 0 else 1.0)
    t = d.mean() / (sd / math.sqrt(n))
    try:
        from scipy import stats
        return float(t), float(2 * stats.t.sf(abs(t), n - 1))
    except Exception:
        # normal approximation fallback (n>=3, disclosed in the CSV)
        from math import erfc
        return float(t), float(erfc(abs(t) / math.sqrt(2)))


def paired_t_one_sided(diffs, direction: int = +1) -> tuple[float, float]:
    t, p2 = paired_t(diffs)
    if direction > 0:
        p = p2 / 2 if t > 0 else 1 - p2 / 2
    else:
        p = p2 / 2 if t < 0 else 1 - p2 / 2
    return t, float(min(max(p, 0.0), 1.0))


def welch(a, b) -> tuple[float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / a.size + vb / b.size)
    if se == 0:
        return (0.0, 1.0)
    t = (a.mean() - b.mean()) / se
    try:
        from scipy import stats
        df = (va / a.size + vb / b.size) ** 2 / (
            (va / a.size) ** 2 / (a.size - 1) + (vb / b.size) ** 2 / (b.size - 1))
        return float(t), float(2 * stats.t.sf(abs(t), df))
    except Exception:
        from math import erfc
        return float(t), float(erfc(abs(t) / math.sqrt(2)))


def cohen_dz(diffs) -> float:
    d = np.asarray(diffs, float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd else float("inf") if d.mean() else 0.0


def bootstrap_ci(diffs, n_boot: int = 10000, seed: int = 42, alpha: float = 0.05):
    d = np.asarray(diffs, float)
    if d.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def tost_paired(diffs, margin: float) -> tuple[float, float]:
    """TOST for equivalence within +/- margin (STATS_PLAN_V3 §5).

    Returns (p, delta_hat) where p is max(p_lower, p_upper) on the difference
    scale; equivalent iff p < .05 on BOTH one-sided tests.
    """
    d = np.asarray(diffs, float)
    n = d.size
    sd = d.std(ddof=1)
    if sd == 0:
        return (0.0 if abs(d.mean()) < margin else 1.0), float(d.mean())
    t = d.mean() / (sd / math.sqrt(n))
    try:
        from scipy import stats
        t_lower = (d.mean() + margin) / (sd / math.sqrt(n))
        t_upper = (d.mean() - margin) / (sd / math.sqrt(n))
        p_lower = float(stats.t.sf(t_lower, n - 1))       # H0: delta <= -margin
        p_upper = float(stats.t.cdf(t_upper, n - 1))      # H0: delta >= +margin
        return float(max(p_lower, p_upper)), float(t)
    except Exception:
        from math import erfc
        p_lower = erfc(((d.mean() + margin) / (sd / math.sqrt(n))) / math.sqrt(2)) / 2
        p_upper = 1 - erfc(((d.mean() - margin) / (sd / math.sqrt(n))) / math.sqrt(2)) / 2
        return float(max(p_lower, p_upper)), float(t)


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (monotonicity preserved, full precision)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [1.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        running = max(running, val)
        adj[i] = running
    return adj


def kendall_tau_a(ranks, values) -> float:
    """Kendall tau-b with tie correction (both vectors may contain ties)."""
    r = np.asarray(ranks, float)
    v = np.asarray(values, float)
    n = r.size
    if n < 2:
        return float("nan")
    conc = disc = 0
    tie_r = tie_v = 0
    for i in range(n):
        for j in range(i + 1, n):
            dr, dv = r[i] - r[j], v[i] - v[j]
            if dr == 0 and dv == 0:
                tie_r += 1
                tie_v += 1
            elif dr == 0:
                tie_r += 1
            elif dv == 0:
                tie_v += 1
            elif dr * dv > 0:
                conc += 1
            else:
                disc += 1
    d0 = math.sqrt((conc + disc + tie_r) * (conc + disc + tie_v))
    return float((conc - disc) / d0) if d0 else float("nan")


# ------------------------------------------------------------------- metrics ---

def load_eval_ids(condition: str, seed: int):
    """Sentence ids of the frozen evaluation draw + the duplicate-free positions.

    Uses the trainer's own function so the draw cannot drift from the one that
    produced the saved per-sentence ppl arrays.
    """
    import train_exp1 as G
    sample = G.load_eval_sentences(condition, seed)
    keep = G.dedup_positions(sample.ids)
    return sample.ids, keep


def gmean_on(ppls, positions=None) -> float:
    arr = np.asarray(ppls, float)
    if positions is not None:
        arr = arr[np.asarray(positions, int)]
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return float("nan")
    return float(math.exp(np.log(arr).mean()))


def read_cell(cell_dir: Path, arch: str, condition: str, seed: int, budget: str = "1x",
              eval_n: int | None = None):
    """Returns (final_ppls, final_ppls_content, ladder{step: ppls}, meta)."""
    if arch == "gpt2":
        res_path = cell_dir / "exp1_result.json"
        if not res_path.exists():
            return None
        meta = json.loads(res_path.read_text())
        ppls = None
        for step in sorted(meta.get("eval_gmean", {}), key=lambda s: int(s)):
            f = cell_dir / f"ppls_step{step}.pt"
            if f.exists():
                import torch
                arr = torch.load(f, weights_only=False)
                ppls = np.asarray(arr, float)
        if ppls is None:
            return None
        content = None
        f = cell_dir / "ppls_content_step3000.pt"
        if f.exists():
            import torch
            content = np.asarray(torch.load(f, weights_only=False), float)
        ladder = {}
        for step in sorted(meta.get("eval_gmean", {}), key=lambda s: int(s)):
            f = cell_dir / f"ppls_step{step}.pt"
            if f.exists():
                import torch
                ladder[int(step)] = np.asarray(torch.load(f, weights_only=False), float)
        return ppls, content, ladder, meta
    res_path = cell_dir / "lstm_result.json"
    if not res_path.exists():
        return None
    meta = json.loads(res_path.read_text())
    steps = sorted((int(s) for s in meta.get("eval_gmean", {})),
                   key=int)
    final_step = steps[-1] if steps else None
    ppls = content = None
    ladder = {}
    for s in steps:
        f = cell_dir / f"eval_step{s}.json"
        if f.exists():
            d = json.loads(f.read_text())
            ladder[s] = np.asarray(d.get("ppls", []), float)
    if final_step is not None and final_step in ladder:
        ppls = ladder[final_step]
    if eval_n:
        ppls = None if ppls is None else ppls[:eval_n]
        ladder = {s: v[:eval_n] for s, v in ladder.items()}
    return ppls, content, ladder, meta


def per_seed_rows(results: Path, lstm_dir: Path | None, out_rows: list, eval_n_lstm={}):
    for arm, root, kind, arch in (("gpt2", results, "exp1_result.json", "gpt2"),
                                  ("lstm_gpu", results.parent / "results_lstm_gpu",
                                   "lstm_result.json", "lstm"),
                                  ("lstm_cpu", lstm_dir, "lstm_result.json", "lstm")):
        if root is None or not Path(root).is_dir():
            continue
        for cell in sorted(Path(root).glob(f"babylm_*/**/{kind}")):
            cell_dir = cell.parent
            cond = cell_dir.parent.name.replace("babylm_", "").replace("_100M", "")
            seed_tag = cell_dir.name
            if seed_tag.startswith("steps"):
                budget = seed_tag.split("_")[0].replace("steps", "") + "_steps"
                seed = int(seed_tag.split("seed")[1])
            else:
                budget, seed = "1x", int(seed_tag.replace("seed", ""))
            got = read_cell(cell_dir, arch, cond, seed, budget)
            if got is None:
                continue
            ppls, content, ladder, meta = got
            ids, keep = load_eval_ids(cond, seed)
            n = len(ppls) if ppls is not None else 0
            keep = [i for i in keep if i < n]
            out_rows.append(dict(
                arm=arm, arch=arch, condition=cond, budget_tag=budget, seed=seed,
                n_draw=n, n_dedup=len(keep),
                ppl_final_all=gmean_on(ppls),
                ppl_final_dedup=gmean_on(ppls, keep),
                ppl_final_content=(gmean_on(content, keep) if content is not None else None),
                fp_draw=meta.get("eval_fingerprint"),
                pool_n=meta.get("eval_pool_n"),
            ))
    return out_rows


# ------------------------------------------------------------- families (H) ---

def _cell(rows, arm, cond, seed, budget="1x"):
    for r in rows:
        if (r["arm"] == arm and r["condition"] == cond and r["seed"] == seed
                and r["budget_tag"] == budget):
            return r
    return None


def _vec(rows, arm, cond, seeds, key="ppl_final_dedup"):
    out = []
    for s in seeds:
        r = _cell(rows, arm, cond, s)
        out.append(r[key] if r else float("nan"))
    return out


def _row(metric, cond1, cond2, arch, seeds, v1, v2, test, p_raw, d, ci, tost_p,
         verdict, note=""):
    return dict(dataset="babylm", model=arch, metric=metric,
                cond1=cond1, cond2=cond2, n1=len(v1), n2=len(v2),
                mean1=_mean(v1), sd1=_sd(v1), mean2=_mean(v2), sd2=_sd(v2),
                test=test, p_raw=p_raw, p_holm=None, d=d,
                d_ci_low=ci[0], d_ci_high=ci[1], tost_p=tost_p, verdict=verdict,
                seeds=",".join(str(s) for s in seeds), note=note)


def run_families(rows, out_dir: Path):
    """F1..F5 + TOST rows, Holm-adjusted inside each family."""
    S3 = [0, 14, 41]
    S5 = [0, 14, 41, 53, 96]
    families: dict[str, list[dict]] = {}

    # --- F1: parity_word vs fixed_start (H10, one-sided >) --------------------
    f1 = []
    for arch, arm, seeds in (("gpt2", "gpt2", S5), ("lstm_gpu", "lstm_gpu", S3)):
        v1, v2 = _vec(rows, arm, "parity_word", seeds), _vec(rows, arm, "fixed_start", seeds)
        if all(np.isfinite(v1)) and all(np.isfinite(v2)) and np.all(np.isfinite(v1)):
            diffs = np.asarray(v1, float) - np.asarray(v2, float)
            t, p = paired_t_one_sided(diffs, +1)
            fam = "F1_gpt2" if arch == "gpt2" else "F1_lstm"
            families.setdefault(fam, []).append(_row(
                "ppl_gmean_final_dedup", "parity_word", "fixed_start", arch, seeds,
                v1, v2, f"paired_t (n={len(seeds)})", p, cohen_dz(diffs),
                bootstrap_ci(diffs), None,
                "diff>0" if p < 0.05 else f"no_detectable_diff_n{len(seeds)}",
                note="H10 primary; one-sided"))
            f1 = families[fam]
    # marker-entropy-matched control (audit B1): also report vs not_random
    for arch, arm, seeds in (("gpt2", "gpt2", S3), ("lstm_gpu", "lstm_gpu", S3)):
        v1, v2 = _vec(rows, arm, "parity_word", seeds), _vec(rows, arm, "not_random", seeds)
        if all(np.isfinite(v1)) and all(np.isfinite(v2)):
            diffs = np.asarray(v1, float) - np.asarray(v2, float)
            t, p = paired_t_one_sided(diffs, +1)
            families.setdefault("F1_not_random", []).append(_row(
                "ppl_gmean_final_dedup", "parity_word", "not_random", arch, seeds,
                v1, v2, f"paired_t (n={len(seeds)})", p, cohen_dz(diffs),
                bootstrap_ci(diffs), None,
                "diff>0" if p < 0.05 else f"no_detectable_diff_n{len(seeds)}",
                note="audit B1: position-entropy-matched control"))

    # --- F2: parity_tok vs fixed_start / fixed_end ----------------------------
    for ctrl in ("fixed_start", "fixed_end"):
        seeds = S5 if ctrl == "fixed_start" else S3
        v1, v2 = _vec(rows, "gpt2", "parity_tok", seeds), _vec(rows, "gpt2", ctrl, seeds)
        if all(np.isfinite(v1)) and all(np.isfinite(v2)):
            diffs = np.asarray(v1, float) - np.asarray(v2, float)
            t, p = paired_t_one_sided(diffs, +1)
            families.setdefault("F2", []).append(_row(
                "ppl_gmean_final_dedup", "parity_tok", ctrl, "gpt2", seeds,
                v1, v2, f"paired_t (n={len(seeds)})", p, cohen_dz(diffs),
                bootstrap_ci(diffs), None,
                "diff>0" if p < 0.05 else f"no_detectable_diff_n{len(seeds)}"))

    # --- F3: negtok vs parity_word -------------------------------------------
    v1, v2 = _vec(rows, "gpt2", "negtok", S5), _vec(rows, "gpt2", "parity_word", S5)
    if all(np.isfinite(v1)) and all(np.isfinite(v2)):
        diffs = np.asarray(v1, float) - np.asarray(v2, float)
        t, p = paired_t_one_sided(diffs, +1)
        families.setdefault("F3", []).append(_row(
            "ppl_gmean_final_dedup", "negtok", "parity_word", "gpt2", S5,
            v1, v2, f"paired_t (n={len(S5)})", p, cohen_dz(diffs),
            bootstrap_ci(diffs), None,
            "diff>0" if p < 0.05 else f"no_detectable_diff_n{len(S5)}"))

    # --- F4: architecture deltas on shared seeds ------------------------------
    for cond, alt, direction in (("shuffle_control", None, 0),
                                 ("reverse_full", "shuffle_control", +1),
                                 ("parity_word", "shuffle_control", +1),
                                 ("not_random", "shuffle_control", +1)):
        g = _vec(rows, "gpt2", cond, S3)
        l = _vec(rows, "lstm_gpu", cond, S3)
        if alt:
            g0, l0 = _vec(rows, "gpt2", alt, S3), _vec(rows, "lstm_gpu", alt, S3)
            if not (all(np.isfinite(g)) and all(np.isfinite(l))
                    and all(np.isfinite(g0)) and all(np.isfinite(l0))):
                continue
            dj = np.asarray(g, float) - np.asarray(g0, float)
            dl = np.asarray(l, float) - np.asarray(l0, float)
            cond1, cond2 = f"delta({cond}−{alt})", "GPT2 − LSTM"
        else:
            if not (all(np.isfinite(g)) and all(np.isfinite(l))):
                continue
            dj, dl = np.asarray(g, float), np.asarray(l, float)
            cond1, cond2 = cond, "GPT2 vs LSTM"
        diffs = dj - dl
        t, p = (paired_t_one_sided(diffs, +1) if direction else paired_t(diffs))
        families.setdefault("F4", []).append(_row(
            "ppl_gmean_final_dedup", cond1, cond2, "gpt2_vs_lstm_gpu", S3,
            list(dj), list(dl), f"paired_t (n={len(S3)})", p, cohen_dz(diffs),
            bootstrap_ci(diffs), None,
            "diff>0" if p < 0.05 else f"no_detectable_diff_n{len(S3)}",
            note="equal-token-budget LSTM arm (audit B2)"))

    # --- F5: H7 2x vs 1x (within condition, seed-paired) ----------------------
    for cond in ("shuffle_control", "parity_word"):
        seeds = [0, 14, 41]
        v2x = _vec(rows, "gpt2", cond, seeds, key="ppl_final_dedup")  # placeholder
        # budget-tagged cells live in a different directory tag; read from rows
        v2x = [(_cell(rows, "gpt2", cond, s, "6000_steps") or {}).get("ppl_final_dedup", float("nan"))
               for s in seeds]
        v1x = [(_cell(rows, "gpt2", cond, s, "1x") or {}).get("ppl_final_dedup", float("nan"))
               for s in seeds]
        if all(np.isfinite(v2x)) and all(np.isfinite(v1x)):
            diffs = np.asarray(v2x, float) - np.asarray(v1x, float)
            t, p = paired_t_one_sided(diffs, +1)
            families.setdefault("F5", []).append(_row(
                "ppl_gmean_final_dedup", f"{cond}@6000", f"{cond}@3000", "gpt2", seeds,
                v2x, v1x, f"paired_t (n={len(seeds)})", p, cohen_dz(diffs),
                bootstrap_ci(diffs), None,
                "diff>0" if p < 0.05 else f"no_detectable_diff_n{len(seeds)}",
                note="H7 budget arm; blocked (<n3) if seeds missing"))

    # Holm inside every family, then write
    out_rows = []
    for fam, rws in families.items():
        pv = [r["p_raw"] for r in rws]
        adj = holm(pv)
        for r, a in zip(rws, adj):
            r["p_holm"] = a
            r["family"] = fam
            if a >= 0.05 and str(r["verdict"]).startswith("diff>"):
                r["verdict"] = f"no_detectable_diff_n{r['n1']}"
            out_rows.append(r)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["family", "dataset", "model", "metric", "cond1", "cond2", "n1", "n2",
              "mean1", "sd1", "mean2", "sd2", "test", "p_raw", "p_holm", "d",
              "d_ci_low", "d_ci_high", "tost_p", "verdict", "seeds", "note"]
    with open(out_dir / "stats_tests_v3.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(out_rows, key=lambda r: (r["family"], r["cond1"])):
            w.writerow(r)
    return out_rows


# ------------------------------------------------------------------ H9 --------

H9_RANKS = {  # frozen cross-class vector (kept for the descriptive report only)
    "shuffle_control": 0, "reverse_control": 1, "fixed_start": 1, "negtok": 2,
    "reverse_partial": 2, "reverse_full": 3, "parity_word": 3, "parity_tok": 4,
    "shuffle_even_odd": 4, "shuffle_local3": 5, "shuffle_local10": 6,
    "shuffle_deterministic21": 7, "shuffle_nondeterministic": 8,
}
# Amended within-class orderings (audit B3). Marker-free classes use raw ppl;
# class P uses the marker-matched within-family deltas.
H9_WITHIN_CLASS = {
    "S_class": ["shuffle_control", "shuffle_local3", "shuffle_local10",
                "shuffle_even_odd", "shuffle_deterministic21",
                "shuffle_nondeterministic"],
    "R_class": ["reverse_control", "reverse_partial", "reverse_full"],
    "P_class": ["fixed_start", "fixed_end", "negtok", "parity_word", "parity_tok"],
}


def run_h9(rows, out_dir: Path):
    seeds = [0, 14, 41]
    def seed_mean(cond, arm="gpt2"):
        vals = _vec(rows, arm, cond, seeds)
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")
    present = {c: seed_mean(c) for c in H9_RANKS if np.isfinite(seed_mean(c))}
    out = []
    if len(present) >= 3:
        r = [H9_RANKS[c] for c in present]
        v = list(present.values())
        out.append(dict(scope="cross_class_descriptive", kendall_tau_a=kendall_tau_a(r, v),
                        n=len(present), threshold=0.75,
                        verdict="criterion_pass" if kendall_tau_a(r, v) >= 0.75 else "criterion_fail",
                        note="frozen vector, descriptive only (RD#2 forbids cross-marker raw ppl orderings)"))
    for cls, conds in H9_WITHIN_CLASS.items():
        vals = {c: seed_mean(c) for c in conds}
        vals = {c: v for c, v in vals.items() if np.isfinite(v)}
        if len(vals) < 3:
            continue
        r = list(range(len(vals)))
        tau = kendall_tau_a(r, list(vals.values()))
        out.append(dict(scope=f"within_{cls}", kendall_tau_a=tau, n=len(vals),
                        threshold=0.75,
                        verdict="criterion_pass" if tau >= 0.75 else "criterion_fail",
                        note="amended criterion (audit B3): ordering inside the class only"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "h9_replication.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scope", "kendall_tau_a", "n", "threshold",
                                          "verdict", "note"])
        w.writeheader()
        for r in out:
            w.writerow(r)
    return out


# --------------------------------------------------------------- self-test ----

def selftest() -> int:
    ok = True
    # paired t vs a hand check: diffs = [1,2,3] -> mean 2, sd 1, t = 2/(1/sqrt3)=3.4641,
    # two-sided p (df=2) = 0.0742 -> the n=3 granularity caveat of STATS_PLAN_V3 §2
    t, p = paired_t([1, 2, 3])
    ok &= abs(t - 3.4641) < 1e-3 and abs(p - 0.0742) < 1e-3
    # one-sided direction handling
    t1, p1 = paired_t_one_sided([1, 2, 3], +1)
    t2, p2 = paired_t_one_sided([-1, -2, -3], +1)
    ok &= p1 < 0.05 and p2 > 0.5
    # Holm: [0.01, 0.04, 0.03] with m=3 -> [0.03, 0.06, 0.06]
    a = holm([0.01, 0.04, 0.03])
    ok &= abs(a[0] - 0.03) < 1e-9 and abs(a[1] - 0.06) < 1e-9
    # TOST: tight differences around 0 with margin 1 -> equivalent
    p_eq, _ = tost_paired([0.01, -0.01, 0.02, -0.02, 0.0], 1.0)
    ok &= p_eq < 0.05
    # kendall tau with perfect agreement / reversal
    ok &= abs(kendall_tau_a([0, 1, 2], [1, 2, 3]) - 1.0) < 1e-9
    ok &= abs(kendall_tau_a([0, 1, 2], [3, 2, 1]) + 1.0) < 1e-9
    # gmean + dedup mechanics
    import math as _m
    ok &= abs(gmean_on([1.0, _m.e ** 2]) - _m.e) < 1e-9
    ok &= gmean_on([4.0, 2.0, 8.0], [0]) == 4.0        # position subset respected
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(REPO / "experiments_v2" / "kallini_repro" / "results"))
    ap.add_argument("--lstm-dir", default="/root/llm-impossible-lstm/experiments_v2/kallini_repro/results_lstm")
    ap.add_argument("--out", default=str(REPO / "experiments_v2" / "kallini_repro" / "aggregated"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    rows: list[dict] = []
    per_seed_rows(Path(args.results), Path(args.lstm_dir) if args.lstm_dir else None, rows)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_seed_v3.csv", "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    fam = run_families(rows, out_dir)
    h9 = run_h9(rows, out_dir)
    print(f"per-seed rows: {len(rows)} | family rows: {len(fam)} | H9 rows: {len(h9)}")
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
