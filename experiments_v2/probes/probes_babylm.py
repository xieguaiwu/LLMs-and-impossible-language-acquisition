#!/usr/bin/env python3
"""probes_babylm.py — BabyLM behavioral probes for the parity-negation rule (v3).

Replaces the voided SVO probe suite (``probes.py``; see FALSIFICATION F4) with the
construct-validity fixes REDTEAM #8 requires:

  P1  minimal pairs, **branch-matched**: each base sentence yields two variants
      (marker obeying vs violating the parity rule). The two position branches —
      obeing-first vs violating-last, and obeying-last vs violating-first — are
      reported separately, so a delta cannot be produced by the positional prior
      alone (REDTEAM #8(a)).
      Score = mean per-sentence NLL difference (violating − obeying) over the
      whole sentence, plus the marker-position NLL when the marker is not the
      sentence-initial token (in the Kallini evaluation convention position 0 has
      no prediction). Both are in nats.
  P2  length extrapolation with an **asymmetric cap**: pairs drawn from test
      sentences longer than the training cap (train <= 60 words, eval 60..200),
      so the probe is not empty by construction (REDTEAM #8(a)).
  P3  hidden-state diagnostic: logistic regression on the last hidden state
      *before* the marker on unmarked content, parity class as the target, 5-fold
      CV, with a ``fixed_start`` checkpoint as the negative control (REDTEAM #8(b)).
  P4  domain dissociation (the decisive probe): on the word-vs-BPE parity
      **disagreement subset**, compare the marker surprisal cost of obeying word
      parity vs obeying BPE parity, in the ``parity_word`` model and in the
      ``parity_tok`` model. A model that tracks word parity must penalise
      BPE-consistent placements more, and vice versa.

Reporting rules (STATS_PLAN_V3 §4): descriptive per-seed summaries only, no alpha
spent, sign consistency across seeds reported; ``fixed_start`` is the negative
control for every probe.

USAGE
    python3 probes_babylm.py --model-dir results/babylm_parity_word_100M/seed0/final \
        [--condition parity_word] [--pairs 500] [--out probe_report.json] [--smoke]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments_v2" / "kallini_repro"))
sys.path.insert(0, str(REPO / "experiments_v2" / "design_v3"))

import train_exp1 as G                          # noqa: E402  (protocol source)
from v3_conditions import _base_ids, _word_count, _not_start, _not_end  # noqa: E402

MARKER_START = _not_start()                     # ["Not"] ids
MARKER_END = _not_end()                         # [" Not"] ids
MAX_TRAIN_WORDS = 60                            # asymmetric cap, REDTEAM #8(a)
PARITY_CONDS = ("parity_word", "parity_tok")


# ------------------------------------------------------------------ pair gen ---

def load_base_pool(limit: int | None = None) -> list[dict]:
    """Held-out base sentences as {text, base_ids, n_words, n_tokens}.

    Source = the same tagged shim JSON the perturbed datasets are generated from,
    so the probe sentences are exactly the ones the models were evaluated on.
    """
    import glob
    base = G.BABYLM_DATA_PATH / "babylm_data" / "babylm_test"
    files = sorted(glob.glob(str(base / "*_parsed.json")))
    assert files, f"no tagged test JSON under {base}"
    pool: list[dict] = []
    for f in files:
        data = json.load(open(f))
        for line in data:
            for sent in line.get("sent_annotations", []):
                ids = _base_ids(sent)
                if not (1 < len(ids) <= 350):
                    continue
                pool.append({"text": sent["sent_text"].strip(),
                             "base_ids": ids,
                             "n_words": _word_count(sent),
                             "n_tokens": len(ids)})
                if limit and len(pool) >= limit:
                    return pool
    return pool


def make_pairs(pool: list[dict], n_pairs: int, domain: str = "word",
               seed: int = 42, min_words: int = 3, max_words: int | None = None,
               min_words_strict: int | None = None) -> list[dict]:
    """Rule-obeying vs rule-violating minimal pairs, tagged with their branch."""
    rng = random.Random(seed)
    items = list(pool)
    rng.shuffle(items)
    pairs: list[dict] = []
    for it in items:
        if it["n_words"] < min_words:
            continue
        if max_words is not None and it["n_words"] > max_words:
            continue
        if min_words_strict is not None and it["n_words"] <= min_words_strict:
            continue
        units = it["n_words"] if domain == "word" else it["n_tokens"]
        obey_first = (units % 2 == 0)
        base = it["base_ids"]
        start_ids = MARKER_START + base
        end_ids = base + MARKER_END
        obeying, violating = (start_ids, end_ids) if obey_first else (end_ids, start_ids)
        pairs.append({
            "n_words": it["n_words"], "n_tokens": it["n_tokens"], "n_units": units,
            "branch": "obey_first" if obey_first else "obey_last",
            "obeying": obeying, "violating": violating,
        })
        if len(pairs) >= n_pairs:
            break
    return pairs


def disagreement_pairs(pool: list[dict], n_pairs: int, seed: int = 42) -> list[dict]:
    """Word-parity vs BPE-parity disagreement subset (P4)."""
    rng = random.Random(seed)
    items = [it for it in pool if (it["n_words"] % 2) != (it["n_tokens"] % 2)]
    rng.shuffle(items)
    out = []
    for it in items[:n_pairs]:
        base = it["base_ids"]
        word_obey = (MARKER_START + base) if it["n_words"] % 2 == 0 else (base + MARKER_END)
        tok_obey = (MARKER_START + base) if it["n_tokens"] % 2 == 0 else (base + MARKER_END)
        out.append({"word_obeying": word_obey, "token_obeying": tok_obey,
                    "n_words": it["n_words"], "n_tokens": it["n_tokens"]})
    return out


# ---------------------------------------------------------------- surprisal ----

@torch.no_grad()
def sentence_nll(model, ids: list[int], device: str = "cpu") -> tuple[float, float | None]:
    """(mean NLL per predicted token, marker-position NLL or None).

    Kallini's convention: position 0 is never predicted (labels are shifted), so a
    sentence-initial marker has no marker NLL — the caller records None and relies
    on the whole-sentence delta for that branch.
    """
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model(input_ids=x, labels=x)
    logits = out.logits[0, :-1, :]
    labels = x[0, 1:]
    logp = torch.log_softmax(logits.float(), dim=-1)
    nll = -logp[torch.arange(labels.numel()), labels]
    mean_nll = float(nll.mean())
    marker_ids = set(MARKER_START + MARKER_END)
    positions = [i - 1 for i, t in enumerate(ids) if t in marker_ids and i >= 1]
    marker_nll = float(nll[positions[0]]) if positions else None
    return mean_nll, marker_nll


def _wilcoxon_p(diffs: list[float]) -> float | None:
    try:
        from scipy import stats
        return float(stats.wilcoxon(diffs, alternative="greater").pvalue)
    except Exception:
        n = len(diffs)
        if n == 0:
            return None
        pos = sum(1 for d in diffs if d > 0)
        # exact sign test on the count of positive deltas
        p = sum(math.comb(n, k) for k in range(pos, n + 1)) / 2 ** n
        return float(p)


def run_p1(model, pairs: list[dict], device: str) -> dict:
    by_branch: dict[str, list[float]] = {}
    marker_deltas: list[float] = []
    for p in pairs:
        o_mean, o_mark = sentence_nll(model, p["obeying"], device)
        v_mean, v_mark = sentence_nll(model, p["violating"], device)
        by_branch.setdefault(p["branch"], []).append(v_mean - o_mean)
        if o_mark is not None and v_mark is not None:
            marker_deltas.append(v_mark - o_mark)
    res = {}
    for branch, diffs in by_branch.items():
        res[branch] = {
            "n": len(diffs),
            "mean_delta_nats": round(float(np.mean(diffs)), 5),
            "pct_positive": round(100 * float(np.mean([d > 0 for d in diffs])), 1),
            "wilcoxon_p": _wilcoxon_p(diffs),
        }
    if marker_deltas:
        res["marker_position"] = {
            "n": len(marker_deltas),
            "mean_delta_nats": round(float(np.mean(marker_deltas)), 5),
            "pct_positive": round(100 * float(np.mean([d > 0 for d in marker_deltas])), 1),
            "wilcoxon_p": _wilcoxon_p(marker_deltas),
        }
    return res


@torch.no_grad()
def run_p4(model, pairs: list[dict], device: str) -> dict:
    """Domain dissociation: cost of word-consistent vs token-consistent placement."""
    word_d, tok_d = [], []
    for p in pairs:
        w_mean, _ = sentence_nll(model, p["word_obeying"], device)
        t_mean, _ = sentence_nll(model, p["token_obeying"], device)
        word_d.append(w_mean)
        tok_d.append(t_mean)
    if not word_d:
        return {}
    return {
        "n": len(word_d),
        "mean_nll_word_obeying": round(float(np.mean(word_d)), 5),
        "mean_nll_token_obeying": round(float(np.mean(tok_d)), 5),
        "delta_token_minus_word": round(float(np.mean(tok_d) - np.mean(word_d)), 5),
        "pct_token_cheaper": round(100 * float(np.mean([t < w for w, t in zip(word_d, tok_d)])), 1),
    }


@torch.no_grad()
def run_p3(model, pool: list[dict], device: str, n: int = 2000, layer: int = -1) -> dict:
    """Hidden-state diagnostic before the marker, on unmarked content."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
    except Exception:
        return {"error": "sklearn unavailable"}
    X, y = [], []
    for it in pool[:n]:
        ids = it["base_ids"]
        if len(ids) < 4:
            continue
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = model(input_ids=x, output_hidden_states=True)
        h = out.hidden_states[layer][0]                 # (L, d)
        X.append(h[len(ids) // 2].float().cpu().numpy())  # content position, no marker yet
        y.append(it["n_words"] % 2)
    if len(set(y)) < 2 or len(y) < 50:
        return {"error": "insufficient data", "n": len(y)}
    Xa, ya = np.asarray(X), np.asarray(y)
    clf = LogisticRegression(max_iter=2000)
    acc = cross_val_score(clf, Xa, ya, cv=5, scoring="accuracy")
    return {"n": len(ya), "cv_accuracy_mean": round(float(acc.mean()), 4),
            "cv_accuracy_sd": round(float(acc.std()), 4), "chance": 0.5}


# ---------------------------------------------------------------------- main ---

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, help="HF dir (results/<cell>/final)")
    ap.add_argument("--condition", default=None,
                    help="condition that produced the checkpoint (defaults to the dir name)")
    ap.add_argument("--domain", default="word", choices=["word", "tok"])
    ap.add_argument("--pairs", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true", help="10 pairs, skip P3/P2 (code-path check)")
    args = ap.parse_args()

    cond = args.condition
    if cond is None:
        cond = Path(args.model_dir).parent.parent.name.replace("babylm_", "").replace("_100M", "")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    model = GPT2LMHeadModel.from_pretrained(args.model_dir).eval().to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model_dir)
    report = {"model_dir": args.model_dir, "condition": cond, "domain": args.domain,
              "device": device, "marker_start_ids": MARKER_START, "marker_end_ids": MARKER_END}

    n_pairs = 10 if args.smoke else args.pairs
    pool = load_base_pool(limit=200 if args.smoke else None)
    pairs = make_pairs(pool, n_pairs, domain=args.domain, seed=args.seed)
    report["P1_minimal_pairs"] = run_p1(model, pairs, device)
    report["P1_meta"] = {"n_pairs": len(pairs), "domain": args.domain,
                         "branch_counts": {b: sum(1 for p in pairs if p["branch"] == b)
                                           for b in {p["branch"] for p in pairs}}}
    if not args.smoke:
        long_pairs = make_pairs(pool, 200, domain=args.domain, seed=args.seed,
                                min_words_strict=MAX_TRAIN_WORDS, max_words=200)
        report["P2_length_extrapolation"] = run_p1(model, long_pairs, device)
        report["P2_meta"] = {"n_pairs": len(long_pairs),
                             "cap": f"train<={MAX_TRAIN_WORDS} words, eval {MAX_TRAIN_WORDS}-200"}
        report["P3_hidden_state_diagnostic"] = run_p3(model, pool, device)
        if cond in PARITY_CONDS or cond == "fixed_start":
            dis = disagreement_pairs(pool, 300, seed=args.seed)
            report["P4_domain_dissociation"] = run_p4(model, dis, device)
            report["P4_meta"] = {"n_pairs": len(dis),
                                 "subset": "word parity != BPE parity"}
    out = Path(args.out) if args.out else Path(args.model_dir).parent / "probe_report_babylm.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
