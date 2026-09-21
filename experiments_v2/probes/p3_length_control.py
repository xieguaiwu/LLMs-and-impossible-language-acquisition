#!/usr/bin/env python3
"""p3_length_control.py — length-controlled re-run of probe P3 (hidden-state diagnostic).

Why this exists
---------------
``probes_babylm.py::run_p3`` asks whether the hidden state of an *unmarked* content
position linearly decodes the word-parity class of the sentence, and reports
5-fold CV accuracy against chance = 0.5.  The first three checkpoints showed
0.560 (fixed_start, no parity rule ever seen), 0.567 (parity_tok) and 0.579
(parity_word) — i.e. the negative control is **as decodable as the experimental
models**, so the raw number cannot be read as "the model represents the counting
rule" (this is registered in FALSIFICATION_SUMMARY.md, section 存疑未决).

A likely reason is that the label is a *deterministic function of sentence
length* (parity of the word count), and sentence length is exactly the kind of
information any representation carries.  This script therefore conditions the
measurement on length:

  * subset: the word-parity vs BPE-parity **disagreement** sentences, where word
    parity is the complement of token parity;
  * strata: **token-count buckets** (default width 4 tokens) — inside a bucket
    the token count is (almost) constant, so the trivially length-decoded route
    to word parity is closed, while word parity still varies;
  * reported: pooled CV accuracy inside strata, the bucket-level values, a
    within-stratum permutation null, and two baselines:
      - raw (unstratified) accuracy, i.e. what P3 reports;
      - a length-only classifier using ``n_words`` as the sole feature, which is
        ~1.0 by construction and demonstrates how trivial the label is.

Interpretation rules (STATS_PLAN_V3 §4: descriptive, no alpha spent):
  * stratified accuracy ≈ 0.5 and ≈ identical to ``fixed_start`` => the P3
    signal is length-borne, not rule-borne;
  * stratified accuracy clearly above the control => candidate rule encoding,
    still pending the position-matched control (``not_random``).

USAGE
    python3 p3_length_control.py --model-dir <results/<cell>/seed<N>/final> --label parity_word
    python3 p3_length_control.py --defaults --out /root/p3_length_control.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments_v2" / "kallini_repro"))
sys.path.insert(0, str(REPO / "experiments_v2" / "design_v3"))

import probes_babylm as P  # noqa: E402

DEFAULT_MODELS = [
    ("parity_word", "babylm_parity_word_100M"),
    ("parity_tok", "babylm_parity_tok_100M"),
    ("fixed_start", "babylm_fixed_start_100M"),
]


def disagreement_pool(cap: int | None = None, sample_seed: int = 0) -> list[dict]:
    """Sentences whose word parity differs from their BPE-token parity.

    The full disagreement pool has >5e5 sentences.  ``probes_babylm.py`` draws its
    300 pairs with a seeded ``random.Random(seed).shuffle``; the original P3 used
    ``pool[:2000]``, i.e. the first entries of a **genre-ordered** list (the loader
    globs the per-genre JSONs in sorted order), which is a genre-biased slice.
    Here the cap is applied after a seeded shuffle of the whole pool instead.
    """
    pool = [it for it in P.load_base_pool() if (it["n_words"] % 2) != (it["n_tokens"] % 2)]
    rng = random.Random(sample_seed)
    rng.shuffle(pool)
    return pool[:cap] if cap else pool


@torch.no_grad()
def hidden_features(model, items: list[dict], layer: int = -1, batch: int = 16) -> tuple[np.ndarray, list[int]]:
    """Last-layer hidden state at the mid-content position (same position as P3).

    Batched with right-padding + attention mask; the mid-content index is computed
    on the unpadded sequence, so padding cannot shift the probe position.
    """
    todo = [(i, it) for i, it in enumerate(items) if len(it["base_ids"]) >= 4]
    keep: list[int] = []
    feats: list[np.ndarray] = []
    for start in range(0, len(todo), batch):
        chunk = todo[start:start + batch]
        lens = [len(it["base_ids"]) for _, it in chunk]
        width = max(lens)
        x = torch.zeros((len(chunk), width), dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for r, ((_, it), L) in enumerate(zip(chunk, lens)):
            x[r, :L] = torch.tensor(it["base_ids"], dtype=torch.long)
            mask[r, :L] = 1
        out = model(input_ids=x, attention_mask=mask, output_hidden_states=True)
        h = out.hidden_states[layer]
        for r, (idx, it) in enumerate(chunk):
            pos = len(it["base_ids"]) // 2
            feats.append(h[r, pos].float().numpy())
            keep.append(idx)
    return np.asarray(feats), keep


def cv_accuracy(X: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 0) -> float | None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    if len(set(y.tolist())) < 2 or len(y) < folds * 4:
        return None
    clf = LogisticRegression(max_iter=2000)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return round(float(cross_val_score(clf, X, y, cv=skf, scoring="accuracy").mean()), 4)


def analyse(model, items: list[dict], bucket: int, min_bucket: int, folds: int,
            batch: int = 16) -> dict:
    X, keep = hidden_features(model, items, batch=batch)
    sub = [items[i] for i in keep]
    y = np.asarray([it["n_words"] % 2 for it in sub])
    n_tok = np.asarray([it["n_tokens"] for it in sub])
    n_words = np.asarray([it["n_words"] for it in sub])

    raw = cv_accuracy(X, y, folds)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    clf = LogisticRegression(max_iter=2000)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    length_only = round(float(cross_val_score(
        clf, n_words.reshape(-1, 1), y, cv=skf, scoring="accuracy").mean()), 4)

    keys = n_tok // bucket
    per_bucket, sizes = {}, []
    for k in sorted(set(keys.tolist())):
        m = keys == k
        if m.sum() < min_bucket or len(set(y[m].tolist())) < 2:
            continue
        acc = cv_accuracy(X[m], y[m], folds)
        if acc is not None:
            per_bucket[int(k * bucket)] = acc
            sizes.append(int(m.sum()))
    pooled = (round(float(np.average(list(per_bucket.values()), weights=sizes)), 4)
              if per_bucket else None)

    null = []
    rng = np.random.default_rng(0)
    for rep in range(3):
        y_perm = y.copy()
        for k in sorted(set(keys.tolist())):
            m = keys == k
            y_perm[m] = rng.permutation(y_perm[m])
        accs, ws = [], []
        for k in sorted(set(keys.tolist())):
            m = keys == k
            if m.sum() < min_bucket or len(set(y_perm[m].tolist())) < 2:
                continue
            acc = cv_accuracy(X[m], y_perm[m], folds, seed=10 + rep)
            if acc is not None:
                accs.append(acc)
                ws.append(int(m.sum()))
        if accs:
            null.append(round(float(np.average(accs, weights=ws)), 4))
    return {
        "n_items": len(sub),
        "raw_cv_accuracy": raw,
        "length_only_baseline": length_only,
        "stratified_cv_accuracy": pooled,
        "n_strata_used": len(per_bucket),
        "n_items_in_strata": int(sum(sizes)),
        "strata": per_bucket,
        "within_stratum_permutation_null": null,
        "prediction": "stratified accuracy ~0.5 (and ~equal to the fixed_start control) => no parity-specific encoding",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir")
    ap.add_argument("--label", default=None)
    ap.add_argument("--defaults", action="store_true",
                    help="analyse the three seed0 checkpoints (parity_word, parity_tok, fixed_start)")
    ap.add_argument("--cap", type=int, default=6000,
                    help="max disagreement sentences after a seeded shuffle (0 = all 5.6e5)")
    ap.add_argument("--sample-seed", type=int, default=0, help="seed for the pool shuffle")
    ap.add_argument("--bucket", type=int, default=4, help="token-count bucket width")
    ap.add_argument("--min-bucket", type=int, default=25)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16, help="forward-pass batch size")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not args.defaults and not args.model_dir:
        ap.error("either --model-dir or --defaults is required")

    from transformers import GPT2LMHeadModel

    items = disagreement_pool(args.cap or None, args.sample_seed)
    print(f"disagreement sentences: {len(items)}", flush=True)

    if args.defaults:
        base = REPO / "experiments_v2" / "kallini_repro" / "results"
        targets = [(label, base / cell / "seed0" / "final") for label, cell in DEFAULT_MODELS]
    else:
        targets = [(args.label or Path(args.model_dir).parent.parent.name, Path(args.model_dir))]

    report: dict[str, dict] = {}
    for label, model_dir in targets:
        if not Path(model_dir).is_dir():
            print(f"[skip] {label}: {model_dir} not found", flush=True)
            continue
        model = GPT2LMHeadModel.from_pretrained(str(model_dir)).eval().to("cpu")
        rec = analyse(model, items, args.bucket, args.min_bucket, args.folds, args.batch)
        rec["model_dir"] = str(model_dir)
        report[label] = rec
        print(label, json.dumps(rec, ensure_ascii=False), flush=True)
        del model

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"-> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())