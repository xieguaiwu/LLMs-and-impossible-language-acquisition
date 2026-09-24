#!/usr/bin/env python3
"""ngram_baseline.py — the "mere pattern predictor" floor (preregistration §10c-1).

WHY
    Chomsky's critique (2023) says an LLM is "nothing but" a pattern predictor
    over surface statistics. Every experiment in this project measures an LLM;
    none measures the pattern-predictor baseline that claim implicitly invokes.
    This script supplies it: an interpolated absolute-discounting bigram model
    fitted on the SAME perturbed corpus a condition trains on, evaluated on the
    SAME frozen 10k-sentence draw with the SAME per-sentence geometric-mean
    perplexity (including the marker-masked content-only column).

    Reading rule (registered): if a condition's deficit against its control is
    already present in this bigram baseline, the deficit is explained by surface
    statistics and the "architecture-level inductive bias" wording must be
    weakened. If the transformer's deficit exceeds the bigram floor, the extra
    gap is evidence of a bias beyond surface statistics.

APPROXIMATION (disclosed)
    True Kneser-Ney needs continuation counts N1+(•, w), i.e. the set of distinct
    left contexts per word — that set does not fit this box's memory cap. We use
    interpolated absolute discounting over plain bigram counts instead:
        P(w|u) = max(c(u,w) - D, 0)/c(u) + lambda(u) * P_uni(w)
        lambda(u) = D * |{w: c(u,w) > 0}| / c(u),   D = n1/(n1 + 2*n2)
    The unigram backoff is add-delta smoothed. This is a LOWER bound on
    Kneser-Ney quality, i.e. a conservative floor for the comparison above.

MEMORY
    Fit is capped by --max-train-tokens (default 6e6) and the process aborts
    before fitting if available memory is below --min-free-mb, because this box
    also runs the cpu2 LSTM arm (an OOM there has already killed cells of other
    sessions).

ORDER 5 (§10c-13 C / G3_ags floor, registered 2026-09-24)
    --order 5 switches to a 5-gram model with the SAME interpolated
    absolute-discounting scheme applied recursively per level (D_k =
    n1/(n1+2n2) fitted per level, backoff chain 5→4→3→2→unigram, BOS-padded
    contexts, same add-0.1 unigram law). Level counts are derived from the
    highest-order table by aggregation, so only one n-gram table per level is
    materialised, and counting is numpy-sorted (np.unique on void-viewed rows)
    instead of Python dicts — a 6M-token fit stays well under the memory cap
    (the 100M pool is still out of scope for this box). Order 2 keeps the
    original dict-based path byte-for-byte (registered floor numbers must not
    move).

--save-per-sentence (§10c-13 C / G3_ags MSSC pairing)
    Writes the per-sentence ln-ppl arrays (all + content columns) as .npy plus
    a sidecar JSON carrying the sentence IDs and the eval_fingerprint, so
    ags_mssc.py can pair conditions sentence-for-sentence (P-class share
    pool-v2; S/R pools differ 0.4-0.5% and are intersected by sentence ID).
    --persentence-seeds scores additional eval draws (the trainer re-draws the
    10k per seed, so per-seed GPT-2 arrays need per-seed floor arrays for
    consistent pairing; the summary CSV stays on --seed as registered).

USAGE
    python3 ngram_baseline.py --conditions shuffle_control parity_word --out DIR
    python3 ngram_baseline.py --conditions parity_word --order 5 --save-per-sentence \
        --persentence-seeds 0 14 41
    python3 ngram_baseline.py --selftest
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

BOS = -1          # left-context symbol for sentence-initial tokens
BIGRAM_STRIDE = 60000     # > any token id (50258) -> unique int key per (prev, w)
NGRAM_MAX_ORDER = 5
NGRAM_VOCAB = 50259       # > any token id incl. BOS offset; ids stored as w+1 (>=0)


def available_mb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return float("inf")


def read_train_tokens(condition: str, max_tokens: int):
    """Yield sentences (lists of ints) from a condition's train pool, capped."""
    root = Path(os.environ.get("KALLINI_DATA_PATH", "/root/kallini_data")) / \
        "babylm_data_perturbed" / f"babylm_{condition}" / "babylm_100M"
    used = 0
    for f in sorted(root.glob("*.train")):
        for line in f.read_text().splitlines():
            toks = [int(t) for t in line.split()]
            if not toks:
                continue
            used += len(toks)
            yield toks
            if used >= max_tokens:
                return


# ------------------------------------------------------------- order-5 model ---
# Same interpolated absolute-discounting family as the bigram path, applied
# recursively. Counting is numpy-sorted: one lexsorted (n,k) int32 table per
# level, lower levels aggregated from it. BOS-padded contexts keep the bigram
# path's sentence-initial convention.

def _level_tables(stream: np.ndarray, k: int):
    """Lexsorted unique rows + counts for level k, plus per-context stats.

    stream = shifted token ids (w+1, BOS -> 0), all sentences concatenated
    WITH BOS padding already inserted, boundaries NOT removed: cross-sentence
    windows would mix sentences, so callers must pad per sentence (see
    fit_ngram). Rows are (ctx_{k-1}, ..., ctx_1, w); rows sharing a context are
    contiguous under the lexsort, which makes the reduceat aggregation valid.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(stream, k)                     # (N, k) view, no copy
    # BOS may appear only as a prefix of a window; a zero after a nonzero marks
    # a sentence boundary inside the window (cross-sentence), and a zero target
    # (window ending in BOS) is a separator artifact — drop both.
    z = win == 0
    valid = (~(z[:, :-1] & ~z[:, 1:]).any(axis=1)) & (~z[:, -1])
    win = np.ascontiguousarray(win[valid], dtype=np.int32)   # ids <= 50259: int32 void keys
    rows, counts = np.unique(win, axis=0, return_counts=True)  # lexsorted
    counts = counts.astype(np.int64)
    ctx_rows = rows[:, : k - 1]
    ctx_void = np.ascontiguousarray(ctx_rows).view(
        np.dtype((np.void, (k - 1) * 4))).ravel()
    ctx_uni, inv = np.unique(ctx_void, return_inverse=True)
    c_ctx = np.bincount(inv, weights=counts).astype(np.int64)   # c(context)
    d_ctx = np.bincount(inv).astype(np.int64)                   # distinct next
    row_void = np.ascontiguousarray(rows).view(
        np.dtype((np.void, k * 4))).ravel()
    return {"rows": row_void, "counts": counts, "n": len(counts),
            "ctx": ctx_uni, "c_ctx": c_ctx, "d_ctx": d_ctx, "k": k}


def _finalize_level(tab: dict) -> dict:
    n1 = int((tab["counts"] == 1).sum())
    n2 = int((tab["counts"] == 2).sum())
    tab["D"] = n1 / (n1 + 2 * n2) if (n1 + 2 * n2) > 0 else 0.5
    return tab


def fit_ngram5(sentences) -> dict:
    """Fit the order-5 interpolated absolute-discount model over a sentence stream.

    Sentences are separated AND led by (k-1) BOS symbols so (a) sentence-initial
    contexts match the bigram path's BOS convention and (b) no cross-sentence
    n-gram is ever counted. Windows where BOS appears anywhere but the prefix
    (i.e. spanning a boundary) are dropped by construction-check below.
    """
    sep = np.zeros(NGRAM_MAX_ORDER - 1, dtype=np.int64)
    chunks: list[np.ndarray] = []
    n_tokens = 0
    first = True
    for toks in sentences:
        arr = np.asarray(toks, dtype=np.int64)
        if arr.size == 0:
            continue
        n_tokens += int(arr.size)
        if not first:
            chunks.append(sep)
        first = False
        chunks.append(np.concatenate([sep, arr + 1]))      # BOS prefix, ids shifted +1
    stream = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int64)
    uni_counts = np.bincount(stream[stream > 0], minlength=NGRAM_VOCAB)
    levels = {k: _finalize_level(_level_tables(stream, k))
              for k in range(2, NGRAM_MAX_ORDER + 1)}
    return {"order": NGRAM_MAX_ORDER, "levels": levels,
            "D": levels[NGRAM_MAX_ORDER]["D"],
            "uni_counts": uni_counts, "total": int(uni_counts.sum()),
            "V": int((uni_counts > 0).sum()), "n_tokens": n_tokens,
            "n_types": int(sum(t["n"] for t in levels.values()))}


def _lookup(tab: dict, ctx_and_w) -> int:
    """Count of one level-k row via binary search on the lexsorted void table."""
    key = np.ascontiguousarray(np.asarray(ctx_and_w, dtype=np.int32)).view(
        np.dtype(("V", tab["k"] * 4)))[0]
    i = np.searchsorted(tab["rows"], key)
    if i < tab["n"] and tab["rows"][i] == key:
        return int(tab["counts"][i])
    return 0


def _ctx_stats(tab: dict, ctx) -> tuple[int, int]:
    key = np.ascontiguousarray(np.asarray(ctx, dtype=np.int32)).view(
        np.dtype(("V", (tab["k"] - 1) * 4)))[0]
    i = np.searchsorted(tab["ctx"], key)
    if i < len(tab["ctx"]) and tab["ctx"][i] == key:
        return int(tab["c_ctx"][i]), int(tab["d_ctx"][i])
    return 0, 0


def logp_ngram(w: int, hist: list[int], m: dict) -> float:
    """ln P(w | hist) under the recursive absolute-discount model (hist = full left history)."""
    h = [x + 1 for x in hist]
    for k in range(m["order"], 1, -1):
        tab = m["levels"][k]
        ctx = h[-(k - 1):]
        if len(ctx) < k - 1:
            ctx = [0] * (k - 1 - len(ctx)) + ctx            # sentence-start BOS padding
        cp, d = _ctx_stats(tab, ctx)
        if cp == 0:
            continue                                        # unseen context: back off
        c = _lookup(tab, ctx + [w + 1])
        lam = tab["D"] * d / cp
        p = max(c - tab["D"], 0.0) / cp + lam * math.exp(_lower_logp(w, h, m, k - 1))
        return math.log(max(p, 1e-12))
    return _lower_logp(w, h, m, 1)


def _lower_logp(w: int, h: list[int], m: dict, level: int) -> float:
    """ln P at `level` (recursive descent to the unigram law)."""
    if level <= 1:
        return _logp_uni(w, m)
    tab = m["levels"][level]
    ctx = h[-(level - 1):]
    if len(ctx) < level - 1:
        ctx = [0] * (level - 1 - len(ctx)) + ctx
    cp, d = _ctx_stats(tab, ctx)
    if cp == 0:
        return _lower_logp(w, h, m, level - 1)
    c = _lookup(tab, ctx + [w + 1])
    lam = tab["D"] * d / cp
    higher = max(c - tab["D"], 0.0) / cp
    p = higher + lam * math.exp(_lower_logp(w, h, m, level - 1))
    return math.log(max(p, 1e-12))


def _logp_uni(w: int, m: dict) -> float:
    # same add-0.1 law as the bigram path's p_uni
    c = int(m["uni_counts"][w + 1]) if 0 <= w + 1 < len(m["uni_counts"]) else 0
    return math.log(max((c + 0.1) / (m["total"] + 0.1 * m["V"]), 1e-12))


def fit_bigram(sentences) -> dict:
    """Count unigrams + bigrams (with BOS context) over a sentence stream."""
    uni: dict[int, int] = {}
    bi: dict[int, int] = {}
    n_tokens = 0
    for toks in sentences:
        prev = BOS
        for w in toks:
            uni[w] = uni.get(w, 0) + 1
            key = prev * BIGRAM_STRIDE + (w + 1)
            bi[key] = bi.get(key, 0) + 1
            prev = w
            n_tokens += 1
    return {"uni": uni, "bi": bi, "n_tokens": n_tokens,
            "n_sentences": None, "n_types": len(bi)}


def finalize(counts: dict) -> dict:
    """Derive c(prev), distinct-next counts, discounts and the unigram law."""
    bi, uni = counts["bi"], counts["uni"]
    n1 = sum(1 for v in bi.values() if v == 1)
    n2 = sum(1 for v in bi.values() if v == 2)
    D = n1 / (n1 + 2 * n2) if (n1 + 2 * n2) > 0 else 0.5
    c_prev: dict[int, int] = {}
    d_prev: dict[int, int] = {}
    for key, c in bi.items():
        prev = key // BIGRAM_STRIDE
        c_prev[prev] = c_prev.get(prev, 0) + c
        d_prev[prev] = d_prev.get(prev, 0) + 1
    total = max(1, counts["n_tokens"])
    V = max(1, len(uni))
    counts.update({"D": D, "c_prev": c_prev, "d_prev": d_prev,
                   "total": total, "V": V, "n1": n1, "n2": n2})
    return counts


def p_uni(w: int, m: dict) -> float:
    return (m["uni"].get(w, 0) + 0.1) / (m["total"] + 0.1 * m["V"])


def logp(w: int, prev: int, m: dict) -> float:
    c = m["bi"].get(prev * BIGRAM_STRIDE + (w + 1), 0)
    cp = m["c_prev"].get(prev, 0)
    if cp == 0:
        return math.log(p_uni(w, m))
    lam = m["D"] * m["d_prev"].get(prev, 0) / cp
    p = max(c - m["D"], 0.0) / cp + lam * p_uni(w, m)
    return math.log(max(p, 1e-12))


def score_sentences_full(m: dict, sents, marker_ids: set[int]) -> tuple[dict, list[float], list[float]]:
    """Per-sentence ppl, GPT-2 convention (position 0 unpredicted), + content-only.

    ``nll`` accumulates NEGATIVE log-probabilities (a sum of log p is negative;
    perplexity is exp(mean NLL)). Returns the summary dict plus the raw
    per-sentence ln-ppl lists (all / content) for --save-per-sentence; the
    summary math is identical to the registered path.
    """
    nll_all, nll_content = [], []
    scorer = logp_ngram if m.get("order", 2) >= 3 else None
    for toks in sents:
        if len(toks) < 2:
            continue
        s = 0.0
        sc = 0.0
        nc = 0
        if scorer is None:
            prev = toks[0]
            for w in toks[1:]:
                lp = logp(w, prev, m)
                s -= lp
                if w not in marker_ids:
                    sc -= lp
                    nc += 1
                prev = w
        else:
            for i in range(1, len(toks)):
                lp = scorer(toks[i], toks[max(0, i - (NGRAM_MAX_ORDER - 1)):i], m)
                s -= lp
                if toks[i] not in marker_ids:
                    sc -= lp
                    nc += 1
        n = len(toks) - 1
        nll_all.append(s / n)
        if nc:
            nll_content.append(sc / nc)
    def gmean(vals):
        return math.exp(float(np.mean(vals))) if vals else float("nan")
    summary = {"n": len(nll_all), "gmean_ppl": round(gmean(nll_all), 4),
               "gmean_ppl_content": round(gmean(nll_content), 4),
               "perplexity_median": round(math.exp(float(np.median(nll_all))), 4)}
    return summary, nll_all, nll_content


def score_sentences(m: dict, sents, marker_ids: set[int]) -> dict:
    """Summary-only wrapper (kept for the registered callers/selftest)."""
    summary, _, _ = score_sentences_full(m, sents, marker_ids)
    return summary


def selftest() -> int:
    ok = True
    counts = fit_bigram([[1, 2, 1, 2], [2, 3]])
    m = finalize(counts)
    # 6 tokens: (BOS,1)(1,2)(2,1)(1,2)(BOS,2)(2,3); key = prev*STRIDE + (w+1)
    ok &= m["n_tokens"] == 6
    ok &= m["bi"][1 * BIGRAM_STRIDE + 3] == 2      # (1,2) appears twice
    ok &= m["bi"][BOS * BIGRAM_STRIDE + 2] == 1    # sentence-initial 1
    ok &= m["n_types"] == 5
    ok &= 0.0 < m["D"] <= 1.0
    # observed bigram must be more probable than an unobserved one
    ok &= logp(2, 1, m) > logp(3, 1, m)
    # probability mass over the full vocabulary closes to ~1 for a seen context
    tot = sum(math.exp(logp(w, 1, m)) for w in range(1, 4))
    ok &= 0.9 <= tot <= 1.1
    # marker masking changes only the content column, not n
    s = score_sentences(m, [[1, 2, 1]], marker_ids={2})
    ok &= s["n"] == 1 and s["gmean_ppl_content"] != s["gmean_ppl"]
    # ppl must be a perplexity (>1 on real data), not a probability (<1)
    s2 = score_sentences(m, [[1, 2, 1, 2, 3]], marker_ids=set())
    ok &= s2["gmean_ppl"] > 1.0
    # ---- order-5 path: isolation, backoff, closure, recursion consistency ----
    m5 = fit_ngram5([[1, 2, 1, 2], [2, 3]])
    ok &= m5["n_tokens"] == 6 and m5["V"] == 3
    # the boundary pair (sent1-last=2 -> sent2-first=2) must NOT be counted:
    # (2,2) never occurs within a sentence in the fit data (sentences isolated)
    ok &= _lookup(m5["levels"][2], [2 + 1, 2 + 1]) == 0
    # within-sentence (1,2) counted twice
    ok &= _lookup(m5["levels"][2], [1 + 1, 2 + 1]) == 2
    # observed transition beats unobserved under the full 5-gram model
    ok &= logp_ngram(2, [1], m5) > logp_ngram(3, [1], m5)
    # probability mass closes to ~1 over the fitted vocab for seen contexts
    tot5 = sum(math.exp(logp_ngram(w, [1, 2], m5)) for w in range(1, 4))
    ok &= 0.9 <= tot5 <= 1.1
    tot5b = sum(math.exp(logp_ngram(w, [1, 2, 3, 1], m5)) for w in range(1, 4))
    ok &= 0.9 <= tot5b <= 1.1
    # order-5 summary path agrees with itself on the content-column convention
    s5, all_nll, cont_nll = score_sentences_full(m5, [[1, 2, 1]], marker_ids={2})
    ok &= s5["n"] == 1 and s5["gmean_ppl_content"] != s5["gmean_ppl"]
    ok &= len(all_nll) == 1 and len(cont_nll) == 1 and all_nll[0] > 0.0
    # order-2 summary must be unchanged by the refactor (same numbers as above)
    s2b, _, _ = score_sentences_full(m, [[1, 2, 1, 2, 3]], marker_ids=set())
    ok &= s2b["gmean_ppl"] == s2["gmean_ppl"]
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def save_per_sentence(out_dir: Path, cond: str, seed: int, order: int,
                      fingerprint: str | None, ids: list[str],
                      nll_all: list[float], nll_content: list[float],
                      model_name: str, eval_n: int) -> None:
    """Write per-sentence ln-ppl arrays (.npy) + sidecar JSON (ids, fingerprint).

    Consumed by ags_mssc.py for the MSSC sentence-level pairing; the arrays are
    aligned with ``ids`` (the eval draw order after the len>=2 filter, identical
    to the trainer's scoring filter).
    """
    fp = fingerprint or "nofp"
    stem = f"ngram_persent_{cond}_seed{seed}_order{order}_fp{fp}"
    np.save(out_dir / f"{stem}_all.npy", np.asarray(nll_all, dtype=np.float64))
    np.save(out_dir / f"{stem}_content.npy", np.asarray(nll_content, dtype=np.float64))
    meta = {"condition": cond, "seed": seed, "order": order, "model": model_name,
            "eval_fingerprint": fingerprint, "eval_n_requested": eval_n,
            "n_scored": len(nll_all), "n_content": len(nll_content),
            "ids": ids, "note": "ln-ppl per sentence; ids aligned after len>=2 filter"}
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--order", type=int, choices=(2, 5), default=2,
                    help="2 = registered bigram floor (unchanged); 5 = G3_ags floor")
    ap.add_argument("--save-per-sentence", action="store_true",
                    help="write per-sentence ln-ppl .npy + ids sidecar (MSSC pairing)")
    ap.add_argument("--persentence-seeds", nargs="*", type=int, default=None,
                    help="extra eval-draw seeds for per-sentence arrays (trainer re-draws per seed)")
    ap.add_argument("--max-train-tokens", type=int, default=6_000_000)
    ap.add_argument("--min-free-mb", type=int, default=1200)
    ap.add_argument("--eval-n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0, help="evaluation draw seed (trainer default 0)")
    ap.add_argument("--out", default=str(REPO / "experiments_v2" / "analysis" / "outputs" / "ngram"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    import train_exp1 as G          # noqa: E402  (single source of the eval draw)

    conds = args.conditions
    if not conds:
        root = Path(os.environ.get("KALLINI_DATA_PATH", "/root/kallini_data")) / "babylm_data_perturbed"
        conds = sorted(p.name.replace("babylm_", "") for p in root.glob("babylm_*")
                       if (p / "babylm_test_affected").is_dir())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cond in conds:
        free = available_mb()
        if free < args.min_free_mb:
            print(f"[skip] {cond}: only {free:.0f} MB free (< {args.min_free_mb}); "
                  f"re-run when the box is idle")
            continue
        raw = read_train_tokens(cond, args.max_train_tokens)
        if args.order == 2:
            counts = fit_bigram(raw)
            m = finalize(counts)
        else:
            m = fit_ngram5(raw)
        model_name = "bigram_absdisc" if args.order == 2 else "5gram_absdisc"
        sample = G.load_eval_sentences(cond, args.seed)
        sents = list(sample)[: args.eval_n]
        res = score_sentences(m, sents, set(G.MARKER_IDS))
        rec = {"condition": cond, "draw_seed": args.seed, "eval_n": res["n"],
               "eval_fingerprint": getattr(sample, "fingerprint", None),
               "train_tokens_used": m["n_tokens"],
               "bigram_types" if args.order == 2 else "ngram_types": m["n_types"],
               "unigram_types": m["V"],
               "discount_D": round(m["D"], 5),
               "gmean_ppl": res["gmean_ppl"], "gmean_ppl_content": res["gmean_ppl_content"],
               "median_ppl": res["perplexity_median"], "model": model_name}
        if args.order == 5:
            rec["order"] = 5          # schema extension only on the new-model rows
        rows.append(rec)
        (out_dir / f"ngram_{cond}_seed{args.seed}.json").write_text(json.dumps(rec, indent=2))
        print(f"{cond:22s} train={m['n_tokens']/1e6:.1f}M tok  "
              f"ppl={res['gmean_ppl']:.2f}  content={res['gmean_ppl_content']:.2f}  "
              f"(n={res['n']})", flush=True)
        if args.save_per_sentence:
            # id alignment mirrors the trainer's scoring filter (len>=2, order kept)
            ids_kept = [sid for sid, toks in zip(sample.ids, sents) if len(toks) >= 2]
            _, all_nll, cont_nll = score_sentences_full(m, sents, set(G.MARKER_IDS))
            save_per_sentence(out_dir, cond, args.seed, args.order,
                              rec["eval_fingerprint"], ids_kept, all_nll, cont_nll,
                              model_name, args.eval_n)
            for extra_seed in (args.persentence_seeds or []):
                if extra_seed == args.seed:
                    continue
                smp = G.load_eval_sentences(cond, extra_seed)
                st = list(smp)[: args.eval_n]
                ids_kept = [sid for sid, toks in zip(smp.ids, st) if len(toks) >= 2]
                _, a_nll, c_nll = score_sentences_full(m, st, set(G.MARKER_IDS))
                save_per_sentence(out_dir, cond, extra_seed, args.order,
                                  getattr(smp, "fingerprint", None), ids_kept,
                                  a_nll, c_nll, model_name, args.eval_n)
                print(f"  [persentence] {cond} draw_seed={extra_seed} n={len(a_nll)}", flush=True)
        if args.order == 2:
            del m, counts
        else:
            del m
    if rows:
        with open(out_dir / "ngram_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"-> {out_dir}/ngram_summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
