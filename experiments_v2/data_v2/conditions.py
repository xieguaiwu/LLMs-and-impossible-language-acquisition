"""Language conditions (v2): original transformations + new controls.

Conditions
----------
natural          : the original sentences (control group).
reversed         : whole-sentence reversal (original Exp condition).
parity_negation  : NOT inserted sentence-initially for even word counts and
                   sentence-finally for odd word counts (original Exp condition).
fixed_start_neg  : CONTROL -- "Not " prepended to every sentence. Same marker
                   token and frequency (100%) as parity_negation, but no parity
                   rule. Isolates the distributional-marker effect from the
                   counting rule.
fixed_end_neg    : CONTROL -- " Not." appended to every sentence.
parity_unit_token: parity computed over GPT-2 BPE tokens instead of whitespace
                   words. Tests whether the parity rule is learnable when the
                   parity domain matches what the model actually sees.
word_shuffle     : reference condition from Kallini et al. (2024) -- within-
                   sentence random shuffle, seeds fixed per sentence.

All transformations are deterministic given the sentence, except word_shuffle
which uses a per-sentence deterministic seed (hash of the sentence) so the
corpus is reproducible.

Parity definition: parity is computed over whitespace-delimited words of the
sentence with trailing punctuation stripped (matching the original
``ImpossibleGrammarFactory``), unless ``unit="token"``.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

NEG_TOKEN = "<NEG>"  # special-token variant marker, single reserved vocab entry


def _strip_sentence(sentence: str) -> str:
    return sentence.rstrip(".?!").strip()


def _words(sentence: str) -> list[str]:
    return _strip_sentence(sentence).split()


def rule_reverse(sentence: str, unit: str = "word", special_token: bool = False) -> str:
    words = _words(sentence)
    return " ".join(reversed(words)) + "."


def _parity_applies(n_units: int) -> bool:
    """Original convention: even count -> NOT at the beginning, odd -> at the end."""
    return n_units % 2 == 0


def rule_parity_negation(sentence: str, unit: str = "word", special_token: bool = False) -> str:
    words = _words(sentence)
    if unit == "token":
        n_units = _bpe_len(sentence)
    else:
        n_units = len(words)
    marker = NEG_TOKEN if special_token else "Not"
    if _parity_applies(n_units):
        return marker + " " + " ".join(words) + "."
    return " ".join(words) + " " + marker + "."


def rule_fixed_start_negation(sentence: str, unit: str = "word", special_token: bool = False) -> str:
    marker = NEG_TOKEN if special_token else "Not"
    return marker + " " + _strip_sentence(sentence) + "."


def rule_fixed_end_negation(sentence: str, unit: str = "word", special_token: bool = False) -> str:
    marker = NEG_TOKEN if special_token else "Not"
    return _strip_sentence(sentence) + " " + marker + "."


def rule_word_shuffle(sentence: str, unit: str = "word", special_token: bool = False) -> str:
    words = _words(sentence)
    seed = int(hashlib.sha256(" ".join(words).encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    shuffled = words[:]
    rng.shuffle(shuffled)
    # Avoid trivial identity shuffles for short sentences.
    if shuffled == words and len(words) > 1:
        shuffled = words[1:] + words[:1]
    return " ".join(shuffled) + "."


_BPE_CACHE: dict[str, int] = {}


def _bpe_len(sentence: str) -> int:
    """GPT-2 BPE length; lazily imported so data tools work without transformers."""
    text = _strip_sentence(sentence)
    if text in _BPE_CACHE:
        return _BPE_CACHE[text]
    try:
        from transformers import GPT2TokenizerFast

        tok = GPT2TokenizerFast.from_pretrained("gpt2")
    except Exception as exc:  # pragma: no cover - requires transformers
        raise RuntimeError("token-unit parity requires transformers + gpt2 tokenizer") from exc
    n = len(tok.encode(text))
    _BPE_CACHE[text] = n
    return n


@dataclass
class ConditionSpec:
    name: str
    fn: str = "identity"          # function name inside this module
    unit: str = "word"            # parity unit, if applicable
    special_token: bool = False   # use <NEG> reserved token instead of "Not"


CONDITIONS: dict[str, ConditionSpec] = {
    "natural": ConditionSpec("natural"),
    "reversed": ConditionSpec("reversed", fn="rule_reverse"),
    "parity_negation": ConditionSpec("parity_negation", fn="rule_parity_negation"),
    "fixed_start_neg": ConditionSpec("fixed_start_neg", fn="rule_fixed_start_negation"),
    "fixed_end_neg": ConditionSpec("fixed_end_neg", fn="rule_fixed_end_negation"),
    "parity_negation_tok": ConditionSpec("parity_negation_tok", fn="rule_parity_negation", unit="token"),
    "parity_negation_negtok": ConditionSpec("parity_negation_negtok", fn="rule_parity_negation", special_token=True),
    "word_shuffle": ConditionSpec("word_shuffle", fn="rule_word_shuffle"),
}

_FUNCS = {
    "identity": lambda s, **kw: _strip_sentence(s) + ".",
    "rule_reverse": rule_reverse,
    "rule_parity_negation": rule_parity_negation,
    "rule_fixed_start_negation": rule_fixed_start_negation,
    "rule_fixed_end_negation": rule_fixed_end_negation,
    "rule_word_shuffle": rule_word_shuffle,
}


def apply_condition(sentences: list[str], condition: str) -> list[str]:
    spec = CONDITIONS[condition]
    fn = _FUNCS[spec.fn]
    return [fn(s, unit=spec.unit, special_token=spec.special_token) for s in sentences]


def write_condition(sentences: list[str], condition: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{condition}.txt"
    with open(path, "w", encoding="utf-8") as f:
        for s in apply_condition(sentences, condition):
            f.write(s + "\n")
    return path


def split_indices(n: int, test_frac: float = 0.05, seed: int = 42) -> tuple[list[int], list[int]]:
    """Deterministic train/test split over sentence indices.

    The SAME split is used for every condition so that all models are evaluated
    on held-out sentences drawn from the identical underlying sentence pool,
    perturbed per-condition afterwards (matching-pair evaluation protocol).
    """
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    n_test = max(1, int(round(n * test_frac)))
    return sorted(idx[n_test:]), sorted(idx[:n_test])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, default="experiments_v2/data_v2/svo_sentences.txt")
    parser.add_argument("--out-dir", type=str, default="experiments_v2/data_v2/conditions")
    parser.add_argument("--conditions", type=str, nargs="*", default=list(CONDITIONS.keys()))
    parser.add_argument("--test-frac", type=float, default=0.05)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        sentences = [ln.strip() for ln in f if ln.strip()]

    train_idx, test_idx = split_indices(len(sentences), args.test_frac)
    train = [sentences[i] for i in train_idx]
    test = [sentences[i] for i in test_idx]

    out_dir = Path(args.out_dir)
    for cond in args.conditions:
        try:
            data_train = apply_condition(train, cond)
            data_test = apply_condition(test, cond)
        except RuntimeError as exc:
            print(f"SKIP {cond}: {exc}")
            continue
        (out_dir / "train").mkdir(parents=True, exist_ok=True)
        (out_dir / "test").mkdir(parents=True, exist_ok=True)
        p_train = out_dir / "train" / f"{cond}.txt"
        p_test = out_dir / "test" / f"{cond}.txt"
        for path, data in ((p_train, data_train), (p_test, data_test)):
            with open(path, "w", encoding="utf-8") as f:
                for s in data:
                    f.write(s + "\n")
        print(f"{cond:24s} train={len(data_train):6d} -> {p_train}")
        print(f"{'':24s} test ={len(data_test):6d} -> {p_test}")
