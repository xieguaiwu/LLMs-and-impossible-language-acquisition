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

USAGE
    python3 ngram_baseline.py --conditions shuffle_control parity_word --out DIR
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


def score_sentences(m: dict, sents, marker_ids: set[int]) -> dict:
    """Per-sentence ppl, GPT-2 convention (position 0 unpredicted), + content-only.

    ``nll`` accumulates NEGATIVE log-probabilities (a sum of log p is negative;
    perplexity is exp(mean NLL)).
    """
    nll_all, nll_content = [], []
    for toks in sents:
        if len(toks) < 2:
            continue
        s = 0.0
        sc = 0.0
        nc = 0
        prev = toks[0]
        for w in toks[1:]:
            lp = logp(w, prev, m)
            s -= lp
            if w not in marker_ids:
                sc -= lp
                nc += 1
            prev = w
        n = len(toks) - 1
        nll_all.append(s / n)
        if nc:
            nll_content.append(sc / nc)
    def gmean(vals):
        return math.exp(float(np.mean(vals))) if vals else float("nan")
    return {"n": len(nll_all), "gmean_ppl": round(gmean(nll_all), 4),
            "gmean_ppl_content": round(gmean(nll_content), 4),
            "perplexity_median": round(math.exp(float(np.median(nll_all))), 4)}


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
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conditions", nargs="*", default=None)
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
        counts = fit_bigram(read_train_tokens(cond, args.max_train_tokens))
        m = finalize(counts)
        sample = G.load_eval_sentences(cond, args.seed)
        sents = list(sample)[: args.eval_n]
        res = score_sentences(m, sents, set(G.MARKER_IDS))
        rec = {"condition": cond, "draw_seed": args.seed, "eval_n": res["n"],
               "eval_fingerprint": getattr(sample, "fingerprint", None),
               "train_tokens_used": m["n_tokens"], "bigram_types": m["n_types"],
               "unigram_types": m["V"], "discount_D": round(m["D"], 5),
               "gmean_ppl": res["gmean_ppl"], "gmean_ppl_content": res["gmean_ppl_content"],
               "median_ppl": res["perplexity_median"], "model": "bigram_absdisc"}
        rows.append(rec)
        (out_dir / f"ngram_{cond}_seed{args.seed}.json").write_text(json.dumps(rec, indent=2))
        print(f"{cond:22s} train={m['n_tokens']/1e6:.1f}M tok  "
              f"ppl={res['gmean_ppl']:.2f}  content={res['gmean_ppl_content']:.2f}  "
              f"(n={res['n']})", flush=True)
        del m, counts
    if rows:
        with open(out_dir / "ngram_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"-> {out_dir}/ngram_summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
