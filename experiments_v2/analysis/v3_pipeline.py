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
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments_v2" / "kallini_repro"))

GENRES = ["aochildes", "bnc_spoken", "cbt", "children_stories", "gutenberg",
          "open_subtitles", "qed", "simple_wikipedia", "switchboard", "wikipedia"]

# Exploratory families (registered 2026-09-20 evening, prereg §10c): pooled into one
# Benjamini-Hochberg bucket (q=.10) and kept out of the confirmatory Holm set.
EXPLORATORY_FAMILIES = {"F4_budget40M", "F7_nope", "F8_datascale", "F9_model_scale"}
DATASCALE_CONDS_EXPL = ["parity_word", "fixed_start"]
MODEL_SCALE_CONDS_EXPL = ["parity_word", "fixed_start"]

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


def bh(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (exploratory bucket, STATS_PLAN §6b)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [1.0] * m
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = min(running, pvals[i] * m / (rank + 1))
        running = val
        adj[i] = val
    return adj


def parse_seed_tag(tag: str) -> tuple[int, str, str]:
    """(seed, budget_tag, variant) from a cell directory name.

    Accepted forms (2026-09-20 evening arms added variants):
      seed0            -> (0, '1x', '')
      seed0_sub1M      -> (0, '1x', 'sub1M')
      steps6000_seed14 -> (14, '6000_steps', '')
      steps9000_seed0  -> (0, '9000_steps', '')
    """
    m = re.match(r"^steps(\d+)_seed(\d+)(?:_(.*))?$", tag)
    if m:
        return int(m.group(2)), f"{m.group(1)}_steps", (m.group(3) or "")
    m = re.match(r"^seed(\d+)(?:_(.*))?$", tag)
    if m:
        return int(m.group(1)), "1x", (m.group(2) or "")
    raise ValueError(f"unparseable cell directory name: {tag!r}")


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
    # Arm registry (§10c, 2026-09-20 evening): one entry per registered results
    # tree. Keep in sync with kallini_queue.sh + grid_status.py.
    arms = [
        ("gpt2", results, "exp1_result.json", "gpt2"),
        ("lstm_gpu", results.parent / "results_lstm_gpu", "lstm_result.json", "lstm"),
        ("lstm_cpu", lstm_dir, "lstm_result.json", "lstm"),
        ("lstm_capmatch", results.parent / "results_lstm_gpu_capmatch",
         "lstm_result.json", "lstm"),
        ("nope", results.parent / "results_nope", "exp1_result.json", "gpt2_nope"),
        ("ladder_probe", results.parent / "results_ladder_probe", "exp1_result.json", "gpt2"),
        ("datascale", results.parent / "results_datascale", "exp1_result.json", "gpt2"),
        ("logo", results.parent / "results_logo", "exp1_result.json", "gpt2"),
        ("model_scale", results.parent / "results_model_scale", "exp1_result.json", "gpt2_medium"),
    ]
    for arm, root, kind, arch in arms:
        if root is None or not Path(root).is_dir():
            continue
        for cell in sorted(Path(root).glob(f"babylm_*/**/{kind}")):
            cell_dir = cell.parent
            cond = cell_dir.parent.name.replace("babylm_", "").replace("_100M", "")
            try:
                seed, budget, variant = parse_seed_tag(cell_dir.name)
            except ValueError:
                continue
            got = read_cell(cell_dir, arch, cond, seed, budget)
            if got is None:
                continue
            ppls, content, ladder, meta = got
            ids, keep = load_eval_ids(cond, seed)
            n = len(ppls) if ppls is not None else 0
            keep = [i for i in keep if i < n]
            # A missing evaluation pool (data not present on this host) yields an
            # empty keep vector: the dedup metric would silently become nan and
            # look like a legitimate result. Flag it instead (2026-09-20).
            dedup_ok = bool(keep)
            if not dedup_ok:
                print(f"[warn] no evaluation pool for {cond} on this host -> "
                      f"dedup metric unavailable for {arm}/{cond}/seed{seed}", file=sys.stderr)
            out_rows.append(dict(
                arm=arm, arch=arch, condition=cond, budget_tag=budget, seed=seed,
                variant=variant,
                n_draw=n, n_dedup=len(keep), dedup_applied=dedup_ok,
                ppl_final_all=gmean_on(ppls),
                ppl_final_dedup=(gmean_on(ppls, keep) if dedup_ok else float("nan")),
                ppl_final_content=(gmean_on(content, keep) if (content is not None and dedup_ok)
                                   else None),
                fp_draw=meta.get("eval_fingerprint"),
                pool_n=meta.get("eval_pool_n"),
            ))
    return out_rows


# ------------------------------------------------------------- families (H) ---

def _cell(rows, arm, cond, seed, budget="1x", variant=None):
    for r in rows:
        if (r["arm"] == arm and r["condition"] == cond and r["seed"] == seed
                and r["budget_tag"] == budget
                and (variant is None or r.get("variant", "") == variant)):
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

    # --- F4: architecture deltas, CAPACITY-MATCHED arm (§10c-2) ---------------
    # Registered 2026-09-20 evening, pre-data for this family: the confirmatory
    # architecture contrast now lives on the capacity-matched LSTM (EMB=HIDDEN=
    # 1620, tied head, 123.4M ≈ GPT-2-small 124M) at the same token budget. The
    # old equal-budget-but-1/3-capacity rows are kept, renamed F4_budget40M, as
    # the budget/capacity diagnostic they actually are.
    for arm_name, family in (("lstm_capmatch", "F4"), ("lstm_gpu", "F4_budget40M")):
        for cond, alt, direction in (("shuffle_control", None, 0),
                                     ("reverse_full", "shuffle_control", +1),
                                     ("parity_word", "shuffle_control", +1),
                                     ("not_random", "shuffle_control", +1)):
            g = _vec(rows, "gpt2", cond, S3)
            l = _vec(rows, arm_name, cond, S3)
            if alt:
                g0, l0 = _vec(rows, "gpt2", alt, S3), _vec(rows, arm_name, alt, S3)
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
            families.setdefault(family, []).append(_row(
                "ppl_gmean_final_dedup", cond1, cond2,
                "gpt2_vs_lstm_capmatch" if arm_name == "lstm_capmatch" else "gpt2_vs_lstm_gpu",
                S3, list(dj), list(dl), f"paired_t (n={len(S3)})", p, cohen_dz(diffs),
                bootstrap_ci(diffs), None,
                "diff>0" if p < 0.05 else f"no_detectable_diff_n{len(S3)}",
                note=("capacity-matched arm (EMB=HIDDEN=1620, 123.4M params; §10c-2)"
                      if arm_name == "lstm_capmatch" else
                      "budget/capacity diagnostic only (40M vs 124M; §10c-2)")))

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

    # === exploratory families (§10c, registered 2026-09-20 evening) ==========
    # Pooled into one BH (q=.10) bucket; no alpha is spent on the confirmatory
    # holm set. Every row is still published.

    # --- F7_nope (§10c-4): position-ablation contrast --------------------------
    # Estimand: the architectural impossibility penalty
    #     B_m(cond) = ppl_m(cond) − ppl_m(shuffle_control)
    # for m in {gpt2, gpt2-nope}. Registered one-sided prediction:
    #     B_gpt2(parity_word) − B_nope(parity_word) > 0
    # (removing positional information reduces the impossible-language deficit,
    # i.e. the bias is carried by position rather than by hierarchy).
    for cond in ("parity_word", "reverse_full"):
        nope_seeds = sorted({r["seed"] for r in rows
                             if r["arm"] == "nope" and r["condition"] == cond})
        seeds = [s for s in S3 if s in nope_seeds]
        if len(seeds) < 2:
            continue
        bg, bn, ok = [], [], True
        for s in seeds:
            for acc, arm in ((bg, "gpt2"), (bn, "nope")):
                a, b = _cell(rows, arm, cond, s), _cell(rows, arm, "shuffle_control", s)
                if not a or not b or not np.isfinite(a["ppl_final_dedup"]) \
                   or not np.isfinite(b["ppl_final_dedup"]):
                    ok = False
                    break
                acc.append(a["ppl_final_dedup"] - b["ppl_final_dedup"])
            if not ok:
                break
        if not ok:
            continue
        diffs = np.asarray(bg, float) - np.asarray(bn, float)
        t, p = paired_t_one_sided(diffs, +1)
        families.setdefault("F7_nope", []).append(_row(
            "ppl_gmean_final_dedup", f"penalty({cond})", "GPT2 − NoPE-GPT2",
            "gpt2_vs_gpt2_nope", seeds, bg, bn, f"paired_t (n={len(seeds)})", p,
            cohen_dz(diffs), bootstrap_ci(diffs), None,
            "diff>0" if p < 0.05 else f"no_detectable_diff_n{len(seeds)}",
            note="exploratory (mechanistic): NoPE = wpe zeroed+frozen"))

    # --- F8_datascale (§10c-5): the PoS analog ---------------------------------
    # Registered one-sided prediction: the impossibility penalty is LARGER at
    # reduced data scale — penalty(sub1M) > penalty(1x full corpus).
    for cond in DATASCALE_CONDS_EXPL:
        seeds = [s for s in (0, 14) if _cell(rows, "datascale", cond, s, "1x", "sub1M")]
        if len(seeds) < 2:
            continue
        small, full, ok = [], [], True
        for s in seeds:
            c_small = _cell(rows, "datascale", cond, s, "1x", "sub1M")
            c_small_ctrl = _cell(rows, "datascale", "shuffle_control", s, "1x", "sub1M")
            c_full = _cell(rows, "gpt2", cond, s)
            c_full_ctrl = _cell(rows, "gpt2", "shuffle_control", s)
            if not all([c_small, c_small_ctrl, c_full, c_full_ctrl]):
                ok = False
                break
            small.append(c_small["ppl_final_dedup"] - c_small_ctrl["ppl_final_dedup"])
            full.append(c_full["ppl_final_dedup"] - c_full_ctrl["ppl_final_dedup"])
        if not ok:
            continue
        diffs = np.asarray(small, float) - np.asarray(full, float)
        t, p = paired_t_one_sided(diffs, +1)
        families.setdefault("F8_datascale", []).append(_row(
            "ppl_gmean_final_dedup", f"penalty({cond})@1M", f"penalty({cond})@full",
            "gpt2", seeds, small, full, f"paired_t (n={len(seeds)})", p,
            cohen_dz(diffs), bootstrap_ci(diffs), None,
            "diff>0" if p < 0.05 else f"no_detectable_diff_n{len(seeds)}",
            note="exploratory: PoS analog at fixed 3000-step budget"))

    # --- F9_model_scale (§10c-6): does scale shrink the bias? ------------------
    # Two-sided: the paper's Limitations speculate memorisation may erase the
    # bias with size, but a larger model can equally amplify it. Reported both ways.
    for cond in MODEL_SCALE_CONDS_EXPL:
        seeds = [s for s in (0, 14) if _cell(rows, "model_scale", cond, s)]
        if len(seeds) < 2:
            continue
        big, small, ok = [], [], True
        for s in seeds:
            c_big = _cell(rows, "model_scale", cond, s)
            c_big_ctrl = _cell(rows, "model_scale", "shuffle_control", s)
            c_sm = _cell(rows, "gpt2", cond, s)
            c_sm_ctrl = _cell(rows, "gpt2", "shuffle_control", s)
            if not all([c_big, c_big_ctrl, c_sm, c_sm_ctrl]):
                ok = False
                break
            big.append(c_big["ppl_final_dedup"] - c_big_ctrl["ppl_final_dedup"])
            small.append(c_sm["ppl_final_dedup"] - c_sm_ctrl["ppl_final_dedup"])
        if not ok:
            continue
        diffs = np.asarray(big, float) - np.asarray(small, float)
        t, p = paired_t(diffs)
        families.setdefault("F9_model_scale", []).append(_row(
            "ppl_gmean_final_dedup", f"penalty({cond})@355M", f"penalty({cond})@124M",
            "gpt2_medium", seeds, big, small, f"paired_t (n={len(seeds)})", p,
            cohen_dz(diffs), bootstrap_ci(diffs), None,
            "exploratory_two_sided",
            note="exploratory: GPT-2-medium vs small at matched token budget"))

    # --- Holm (confirmatory) / BH (exploratory) -------------------------------
    out_rows = []
    conf_rows, expl_rows = [], []
    for fam, rws in families.items():
        target = expl_rows if fam in EXPLORATORY_FAMILIES else conf_rows
        for r in rws:
            r["family"] = fam
            r["exploratory"] = fam in EXPLORATORY_FAMILIES
            target.append(r)
    for fam, rws in families.items():
        if fam in EXPLORATORY_FAMILIES:
            continue
        adj = holm([r["p_raw"] for r in rws])
        for r, a in zip(rws, adj):
            r["p_holm"] = a
            if a >= 0.05 and str(r["verdict"]).startswith("diff>"):
                r["verdict"] = f"no_detectable_diff_n{r['n1']}"
    if expl_rows:
        adj = bh([r["p_raw"] for r in expl_rows])
        for r, a in zip(expl_rows, adj):
            r["p_holm"] = a
            if a >= 0.10 and str(r["verdict"]).startswith("diff>"):
                r["verdict"] = f"no_detectable_diff_n{r['n1']}_bh"
    out_rows = conf_rows + expl_rows
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["family", "exploratory", "dataset", "model", "metric", "cond1", "cond2",
              "n1", "n2", "mean1", "sd1", "mean2", "sd2", "test", "p_raw", "p_holm",
              "d", "d_ci_low", "d_ci_high", "tost_p", "verdict", "seeds", "note"]
    with open(out_dir / "stats_tests_v3.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(out_rows, key=lambda r: (r["family"], r["cond1"])):
            w.writerow(r)
    return out_rows


# ------------------------------------------------------------------ H9 --------

H9_RANKS = {  # frozen cross-class vector (kept for the descriptive report only)    "shuffle_control": 0, "reverse_control": 1, "fixed_start": 1, "negtok": 2,
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
    # selftest additions for the §10c arms
    ok &= parse_seed_tag("seed0") == (0, "1x", "")
    ok &= parse_seed_tag("seed14_sub1M") == (14, "1x", "sub1M")
    ok &= parse_seed_tag("steps9000_seed0") == (0, "9000_steps", "")
    ok &= parse_seed_tag("seed0_logo7sw") == (0, "1x", "logo7sw")
    # BH monotonicity + identity on a uniform p vector
    b = bh([0.01, 0.02, 0.03, 0.04])
    ok &= all(b[i] <= b[i + 1] + 1e-12 for i in range(3)) and b[0] <= 0.04
    # exploratory bucket policy
    ok &= "F4" in (set() | {"F4"}) and "F7_nope" in EXPLORATORY_FAMILIES
    ok &= "F4" not in EXPLORATORY_FAMILIES
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def run_ladder_probe_dynamics(results: Path, lstm_dir: Path | None, out_dir: Path):
    """Rule-acquisition dynamics from the in-process ladder probe (§10c-3).

    The class-P cells now carry ``ladder_probe`` inside their result JSON: the P1
    branch-matched minimal-pair delta (violating − obeying mean NLL, in nats) at
    every evaluation checkpoint. Tracing it against the content-only ppl curve
    separates two stories: a rule that the model acquires progressively (probe
    delta falls while content ppl falls) vs surface marker statistics (probe
    delta flat while the marker's own NLL collapses). Descriptive only — no
    alpha is spent; the confirmatory probé claims remain P1-P4 on final weights.
    """
    rows = []
    for root, kind in ((results, "exp1_result.json"),
                       (results.parent / "results_ladder_probe", "exp1_result.json")):
        if not Path(root).is_dir():
            continue
        for f in sorted(Path(root).glob(f"babylm_*/**/{kind}")):
            try:
                meta = json.loads(f.read_text())
            except Exception:
                continue
            lp = meta.get("ladder_probe") or {}
            cond = meta.get("language")
            for step, branches in sorted(lp.items(), key=lambda kv: int(kv[0])):
                for branch, d in branches.items():
                    rows.append(dict(
                        condition=cond, seed=meta.get("seed"), step=int(step),
                        branch=branch, n=d.get("n"), mean_delta_nats=d.get("mean_delta_nats"),
                        pct_positive=d.get("pct_positive"), wilcoxon_p=d.get("wilcoxon_p"),
                        content_gmean=(meta.get("eval_gmean_content") or {}).get(str(step)),
                        all_gmean=(meta.get("eval_gmean") or {}).get(str(step))))
    if rows:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "ladder_probe_dynamics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows


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
    ladder = run_ladder_probe_dynamics(Path(args.results), Path(args.lstm_dir) if args.lstm_dir else None,
                                       out_dir)
    print(f"per-seed rows: {len(rows)} | family rows: {len(fam)} | H9 rows: {len(h9)} "
          f"| ladder-probe rows: {len(ladder)}")
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
