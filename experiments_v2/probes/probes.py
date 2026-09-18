"""Behavioral probes for the parity-negation rule (v2).

Three probes answer the construct-validity question that final-loss
comparisons cannot: did the model learn the *parity rule*, or just the
surface distribution of the "Not" marker?

P1  minimal pairs        -- 500 pairs of held-out sentences that are lexically
                            identical and differ only in whether the negation
                            placement obeys the parity rule. Score: mean
                            surprisal of the negation marker in obeying vs
                            violating positions, and the delta.
P2  violation detection  -- S(marker | rule-consistent context) minus
                            S(marker | rule-violating context) per pair,
                            aggregated; positive delta = rule encoded.
P3  length extrapolation -- train-set length distribution restricted to
                            <= max_train_words; probe sentences with MORE
                            words than any training sentence. A model that
                            abstracted "parity of word count" should still
                            place the marker correctly on unseen lengths;
                            a memorizer of surface patterns will not.

Probes run on trained checkpoints via GPT-2 forward passes (no training).
For LSTM runs, P1/P2 work identically; P3 requires the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments_v2" / "data_v2"))
sys.path.insert(0, str(REPO / "experiments_v2" / "training"))

from conditions import NEG_TOKEN, _strip_sentence  # noqa: E402


# ------------------------------------------------------------- pair gen ----

def make_minimal_pairs(sentences: list[str], n_pairs: int = 500,
                       special_token: bool = False,
                       seed: int = 42) -> list[dict]:
    """Build rule-obeying vs rule-violating sentence pairs.

    For a sentence with an even word count the rule puts the marker first;
    the violation puts it last (and vice versa for odd counts). Lexical
    content identical; only marker position differs.
    """
    rng = random.Random(seed)
    marker = NEG_TOKEN if special_token else "Not"
    pairs = []
    pool = [s for s in sentences if len(_strip_sentence(s).split()) >= 3]
    rng.shuffle(pool)
    for s in pool[:n_pairs]:
        words = _strip_sentence(s).split()
        obey_first = (len(words) % 2 == 0)  # rule: even -> marker first
        obeying = (marker + " " + " ".join(words)) if obey_first else (" ".join(words) + " " + marker)
        violating = (" ".join(words) + " " + marker) if obey_first else (marker + " " + " ".join(words))
        pairs.append({
            "sentence": " ".join(words),
            "n_words": len(words),
            "obeying": obeying + ".",
            "violating": violating + ".",
            "marker_index_obeying": 0 if obey_first else len(words),
            "marker_index_violating": len(words) if obey_first else 0,
        })
    return pairs


def make_extrapolation_pairs(train_sentences: list[str], test_sentences: list[str],
                             n_pairs: int = 200, special_token: bool = False,
                             seed: int = 42) -> list[dict]:
    """P3: pairs whose word count exceeds every training sentence length."""
    marker = NEG_TOKEN if special_token else "Not"
    max_train_len = max(len(_strip_sentence(s).split()) for s in train_sentences)
    rng = random.Random(seed)
    pool = [s for s in test_sentences
            if len(_strip_sentence(s).split()) > max_train_len]
    rng.shuffle(pool)
    out = []
    for s in pool:
        words = _strip_sentence(s).split()
        if len(words) <= max_train_len or len(out) >= n_pairs:
            break
        obey_first = (len(words) % 2 == 0)
        obeying = (marker + " " + " ".join(words)) if obey_first else (" ".join(words) + " " + marker)
        violating = (" ".join(words) + " " + marker) if obey_first else (marker + " " + " ".join(words))
        out.append({
            "n_words": len(words),
            "max_train_len": max_train_len,
            "obeying": obeying + ".",
            "violating": violating + ".",
        })
    return out


# ------------------------------------------------------------ surprisal ----

def marker_surprisal(model, tokenizer, sentence: str, marker: str = "Not") -> dict:
    """Mean surprisal (nats) of the marker token(s) at the position where it occurs.

    Searches BOTH surface forms: sentence-initial markers are encoded without a
    leading space ("Not"), medial/final ones with it (" Not") — the two are
    different BPE ids, and searching only one of them made every minimal pair
    score as not-found (v2 incident, 2026-09-19 recheck).
    """
    import torch

    ids = tokenizer.encode(sentence)
    candidates = []
    for surface in (marker.strip(), " " + marker.strip()):
        mi = tokenizer.encode(surface, add_special_tokens=False)
        if mi and mi not in candidates:
            candidates.append(mi)
    pos = None
    used_m = None
    for marker_ids in candidates:
        m = len(marker_ids)
        for i in range(len(ids) - m + 1):
            if ids[i : i + m] == marker_ids:
                pos = i
                used_m = m
                break
        if pos is not None:
            break
    if pos is None:
        return {"found": False, "surprisal": None}

    device = next(model.parameters()).device
    input_ids = torch.tensor([ids], device=device)
    with torch.no_grad():
        logits = model(input_ids).logits
    logprobs = torch.log_softmax(logits[0], dim=-1)
    marker_ids = ids[pos : pos + used_m]
    total = 0.0
    for j, tok in enumerate(marker_ids):
        pred_pos = pos + j - 1  # logits[t] predicts token t+1
        total += -float(logprobs[pred_pos, tok])
    return {"found": True, "surprisal": total / used_m, "position": pos}


def evaluate_pairs(model, tokenizer, pairs: list[dict], marker: str = "Not") -> dict:
    rows = []
    for p in pairs:
        s_ob = marker_surprisal(model, tokenizer, p["obeying"], marker)
        s_vi = marker_surprisal(model, tokenizer, p["violating"], marker)
        if not (s_ob.get("found") and s_vi.get("found")):
            continue
        rows.append({
            "n_words": p["n_words"],
            "S_obeying": s_ob["surprisal"],
            "S_violating": s_vi["surprisal"],
            "delta": s_vi["surprisal"] - s_ob["surprisal"],
        })
    if not rows:
        return {"n": 0}
    n = len(rows)
    mean = lambda k: sum(r[k] for r in rows) / n
    delta = mean("delta")
    sd = (sum((r["delta"] - delta) ** 2 for r in rows) / max(n - 1, 1)) ** 0.5
    return {
        "n": n,
        "mean_S_obeying": round(mean("S_obeying"), 4),
        "mean_S_violating": round(mean("S_violating"), 4),
        "mean_delta": round(delta, 4),
        "sd_delta": round(sd, 4),
        "se_delta": round(sd / math.sqrt(n), 4),
        "pct_rule_encoded": round(100.0 * sum(1 for r in rows if r["delta"] > 0) / n, 2),
    }


# ------------------------------------------------- diagnostic probe ----

def diagnostic_probe(model, tokenizer, sentences: list[str], hidden_layer: int = -1,
                     test_frac: float = 0.2, seed: int = 42) -> dict:
    """Logistic regression on last-token hidden state -> parity class.

    Target: 1 if the parity rule puts the marker FIRST for this sentence
    (even word count), 0 if LAST (odd). Chance = 50%. Accuracy well above
    chance means the model's representations encode word-count parity.
    """
    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    device = next(model.parameters()).device
    X, y = [], []
    for s in sentences:
        words = _strip_sentence(s).split()
        if len(words) < 3:
            continue
        ids = torch.tensor([tokenizer.encode(s)], device=device)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        hs = out.hidden_states[hidden_layer][0, -1, :].float().cpu().numpy()
        X.append(hs)
        y.append(1 if len(words) % 2 == 0 else 0)
    X = np.array(X)
    y = np.array(y)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_frac, random_state=seed)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, ytr)
    return {
        "n": int(len(y)),
        "layer": hidden_layer,
        "train_acc": round(float(clf.score(Xtr, ytr)), 4),
        "test_acc": round(float(clf.score(Xte, yte)), 4),
        "chance": 0.5,
    }


# ------------------------------------------------------------------ main ----

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True,
                        help="directory of a trained run (training_metrics.json + model files), or HF dir")
    parser.add_argument("--model", default="gpt2", choices=["gpt2", "gpt2_tiny", "lstm", "lstm_matched"])
    parser.add_argument("--special-token", action="store_true",
                        help="probe a parity_negation_negtok run (<NEG> marker)")
    parser.add_argument("--n-pairs", type=int, default=500)
    parser.add_argument("--probe-diagnostic", action="store_true")
    parser.add_argument("--extrapolation", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    marker = NEG_TOKEN if args.special_token else "Not"
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    if args.special_token:
        tokenizer.add_special_tokens({"additional_special_tokens": [NEG_TOKEN]})
    model = GPT2LMHeadModel.from_pretrained(args.model_dir)
    if args.special_token:
        model.resize_token_embeddings(len(tokenizer))
    model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

    # held-out sentences: reuse the SVO test split (identical pool for all conditions)
    svo_path = REPO / "experiments_v2" / "data_v2" / "svo_sentences.txt"
    with open(svo_path, encoding="utf-8") as f:
        sentences = [ln.strip() for ln in f if ln.strip()]
    from conditions import split_indices

    _train_idx, test_idx = split_indices(len(sentences))
    test_sentences = [sentences[i] for i in test_idx]

    report = {"model_dir": args.model_dir, "model": args.model, "marker": marker}

    pairs = make_minimal_pairs(test_sentences, args.n_pairs,
                               special_token=args.special_token)
    report["P1_P2_minimal_pairs"] = evaluate_pairs(model, tokenizer, pairs, marker)

    if args.extrapolation:
        from conditions import apply_condition  # noqa: F401

        train_pool = sentences  # proxy for the training pool lengths
        ext = make_extrapolation_pairs(train_pool, test_sentences,
                                       special_token=args.special_token)
        report["P3_length_extrapolation"] = evaluate_pairs(model, tokenizer, ext, marker)

    if args.probe_diagnostic:
        report["P4_diagnostic_probe"] = diagnostic_probe(model, tokenizer, test_sentences)

    out = Path(args.out) if args.out else Path(args.model_dir) / "probe_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
