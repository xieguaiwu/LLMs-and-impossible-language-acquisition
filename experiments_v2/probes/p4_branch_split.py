#!/usr/bin/env python3
"""p4_branch_split.py — branch-split decomposition of probe P4 (domain dissociation).

Why this exists
---------------
``probes_babylm.py::run_p4`` compares, on the word-parity vs BPE-parity
*disagreement* subset, the mean NLL of the word-consistent placement against the
token-consistent placement and reports a single delta.  That delta is **not
interpretable on its own**: the two variants differ in marker *position* as well
as in rule-consistency, so a model's positional prior enters the number directly
(REDTEAM #8(a) made exactly this point for P1; P1 was fixed by reporting the two
branches separately, P4 was not).

This script performs the branch-matched decomposition:

    D = NLL(marker at the end) - NLL(marker at the beginning)      [nats]
        > 0  =>  the model finds "marker at the end" the more expensive option

    case A: word-consistent placement = sentence-initial  (word count even)
    case B: word-consistent placement = sentence-final    (word count odd)

A model that tracks **word** parity must find the rule-consistent placement
cheaper => D_A > D_B.  A model that tracks **token/BPE** parity sees the opposite
assignment on this subset (token parity is the complement of word parity here)
=> D_A < D_B.  A pure positional prior with no parity knowledge gives
D_A ≈ D_B up to a case-correlated artefact, which is what the ``fixed_start``
checkpoint measures.

Reporting rules: descriptive only, no alpha spent (STATS_PLAN_V3 §4).  The
``fixed_start`` row is the negative control; a **position-matched** control
(``not_random``) is still required before the dissociation can be claimed
(design audit B1 / REDTEAM #2c).

USAGE
    python3 p4_branch_split.py --model-dir <results/<cell>/seed<N>/final> \
        [--label parity_word] [--pairs 300] [--out p4_branch_split.json]
    python3 p4_branch_split.py --defaults --out /root/p4_branch_split.json
"""

from __future__ import annotations

import argparse
import json
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


def marker_at_start(ids: list[int]) -> bool:
    return ids[0] in set(P.MARKER_START)


@torch.no_grad()
def analyse_model(model, pairs: list[dict], device: str = "cpu") -> dict:
    """Return the branch-split statistics for one checkpoint."""
    d_a: list[float] = []          # word-consistent = sentence-initial
    d_b: list[float] = []          # word-consistent = sentence-final
    raw: list[float] = []          # the un-decomposed P4 delta, for comparison
    for pr in pairs:
        w_ids, t_ids = pr["word_obeying"], pr["token_obeying"]
        nll_w, _ = P.sentence_nll(model, w_ids, device)
        nll_t, _ = P.sentence_nll(model, t_ids, device)
        nll_start, nll_end = (nll_w, nll_t) if marker_at_start(w_ids) else (nll_t, nll_w)
        d = nll_end - nll_start
        (d_a if marker_at_start(w_ids) else d_b).append(d)
        raw.append(nll_t - nll_w)
    if not d_a or not d_b:
        return {"error": "one branch is empty", "n_A": len(d_a), "n_B": len(d_b)}
    return {
        "n_A": len(d_a),
        "n_B": len(d_b),
        "D_A_mean": round(float(np.mean(d_a)), 4),
        "D_B_mean": round(float(np.mean(d_b)), 4),
        "D_A_minus_D_B": round(float(np.mean(d_a) - np.mean(d_b)), 4),
        "pct_D_A_positive": round(100 * float(np.mean([x > 0 for x in d_a])), 1),
        "pct_D_B_positive": round(100 * float(np.mean([x > 0 for x in d_b])), 1),
        "raw_delta_token_minus_word": round(float(np.mean(raw)), 4),
        "prediction": "D_A > D_B for a word-parity tracker; D_A < D_B for a token-parity tracker",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", help="HF dir (results/<cell>/seed<N>/final)")
    ap.add_argument("--label", default=None, help="label for this checkpoint in the report")
    ap.add_argument("--pairs", type=int, default=300, help="disagreement-subset pairs")
    ap.add_argument("--seed", type=int, default=42, help="pair-draw seed (probe default)")
    ap.add_argument("--defaults", action="store_true",
                    help="analyse the three seed0 checkpoints (parity_word, parity_tok, fixed_start)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not args.defaults and not args.model_dir:
        ap.error("either --model-dir or --defaults is required")

    from transformers import GPT2LMHeadModel

    pool = P.load_base_pool()
    pairs = P.disagreement_pairs(pool, args.pairs, seed=args.seed)
    print(f"disagreement pairs: {len(pairs)}", flush=True)

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
        rec = analyse_model(model, pairs, "cpu")
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