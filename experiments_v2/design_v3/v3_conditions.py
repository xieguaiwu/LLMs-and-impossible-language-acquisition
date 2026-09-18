#!/usr/bin/env python3
"""v3 class-P perturbations in Kallini data format (DESIGN_V3 §1).

Produces Kallini-format token-ID lines (space-joined ints) for the paper's
parity-negation family and bare-reverse reference, usable directly by the
kallini_repro trainer (packing/eval protocol shared with the Shuffle/Reverse
replication). Reuses their tokenizer registry (utils.py) for exact fidelity.

Conditions (DESIGN_V3 §1.1):
  parity_word    "Not" sentence-final if odd WORD count, sentence-initial if even
                 (the paper's original rule); control: fixed_start / fixed_end.
  parity_tok     same rule over GPT-2 BPE token count of the raw sentence
                 BEFORE marker insertion (meaningful on BabyLM only).
  negtok         word parity, marker = reserved <NEG> special token (vocab +1).
  fixed_start    "Not " always sentence-initial (primary marker control).
  fixed_end      " Not" always sentence-final (position control).
  bare_reverse   whole-sentence reversal, no marker (labeled Reverse-bare;
                 replication claims are made ONLY by kallini_repro's
                 reverse_full / reverse_control pair).
  word_shuffle   per-sentence random shuffle (nondeterministic), our own
                 Kallini-NondeterministicShuffle analog used in v2.

All functions take a Kallini-style sentence annotation dict ({"sent_text": ...})
and return a list of token ids. Filters mirror Kallini's filter_shuffle
(>1 token, <=350 tokens) for cross-condition sentence-set identity.
"""

from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

KALLINI_REPO = Path("/root/mission-impossible-language-models")
sys.path.insert(0, str(KALLINI_REPO))
from utils import gpt2_original_tokenizer  # noqa: E402


def _base_ids(sent: dict, tokenizer=gpt2_original_tokenizer) -> list[int]:
    return tokenizer.encode(sent["sent_text"].strip())


# ------------------------------------------------------------- conditions ---

def perturb_parity_word(sent: dict, **kw) -> list[int]:
    """Paper's rule: even word count -> marker first; odd -> marker last."""
    words = sent["sent_text"].strip().rstrip(".?!").split()
    toks = _base_ids({"sent_text": sent["sent_text"]})
    not_ids = gpt2_original_tokenizer.encode(" Not", add_special_tokens=False)
    if len(words) % 2 == 0:
        first_ids = gpt2_original_tokenizer.encode("Not", add_special_tokens=False)
        return first_ids + toks
    return toks + not_ids


def perturb_parity_tok(sent: dict, **kw) -> list[int]:
    toks = _base_ids({"sent_text": sent["sent_text"]})
    # parity domain = BPE tokens of the sentence BEFORE marker insertion
    # (punctuation included — the model's actual countable units)
    n_units = len(toks)
    not_ids = gpt2_original_tokenizer.encode(" Not", add_special_tokens=False)
    if n_units % 2 == 0:
        first = gpt2_original_tokenizer.encode("Not", add_special_tokens=False)
        return first + toks
    return toks + not_ids


def perturb_fixed_start(sent: dict, **kw) -> list[int]:
    return gpt2_original_tokenizer.encode("Not", add_special_tokens=False) + \
        _base_ids({"sent_text": sent["sent_text"]})


def perturb_fixed_end(sent: dict, **kw) -> list[int]:
    return _base_ids({"sent_text": sent["sent_text"]}) + \
        gpt2_original_tokenizer.encode(" Not", add_special_tokens=False)


_NEG_TOK = None


def _neg_token_id() -> int:
    global _NEG_TOK
    if _NEG_TOK is None:
        tok = _register_negtok()
        _NEG_TOK = tok.convert_tokens_to_ids("<NEG>")
    return _NEG_TOK


def _register_negtok():
    tok = gpt2_original_tokenizer.__class__.from_pretrained("gpt2")
    tok.add_special_tokens({"additional_special_tokens": ["<NEG>"]})
    return tok


def perturb_negtok(sent: dict, **kw) -> list[int]:
    words = sent["sent_text"].strip().rstrip(".?!").split()
    toks = _base_ids({"sent_text": sent["sent_text"]})
    nid = _neg_token_id()
    if len(words) % 2 == 0:
        return [nid] + toks if (nid := _neg_token_id()) is not None else toks
    return toks + [nid]


def perturb_bare_reverse(sent: dict, **kw) -> list[int]:
    return _base_ids({"sent_text": sent["sent_text"]})[::-1]


def perturb_word_shuffle(sent: dict, seed: int = 0, **kw) -> list[int]:
    toks = _base_ids({"sent_text": sent["sent_text"]})
    rng = hashlib.sha256(("shuffle:" + sent["sent_text"]).encode()).digest()
    order = list(range(len(toks)))
    # deterministic per-sentence shuffle (numpy-free): Fisher-Yates from hash
    h = int(rng.hexdigest(), 16)
    for i in range(len(order) - 1, 0, -1):
        h = (h * 6364136223846793005 + 1442695040888963407) & ((1 << 63) - 1)
        j = h % (i + 1)
        order[i], order[j] = order[j], order[i]
    return [toks[k] for k in order]


def filter_short_long(sent: dict, tokenizer=gpt2_original_tokenizer) -> bool:
    """Kallini filter_shuffle analog: >1 token, <=350 tokens (verbatim bound)."""
    n = len(tokenizer.encode(sent["sent_text"]))
    return 1 < n <= 350


CONDITIONS = {
    "parity_word": {"fn": perturb_parity_word, "vocab_extra": 0},
    "parity_tok": {"fn": perturb_parity_tok, "vocab_add": 0},
    "negtok": {"fn": perturb_negtok, "vocab_add": 1},
    "fixed_start": {"fn": perturb_fixed_start, "vocab_add": 0},
    "fixed_end": {"fn": perturb_fixed_end, "vocab_add": 0},
    "bare_reverse": {"fn": lambda s, **kw: s and perturb_bare_reverse(s), "vocab_add": 0},
    "word_shuffle": {"fn": perturb_word_shuffle, "vocab_add": 0},
}


def write_condition(lang: str, tagged_json: Path, out_dir: Path, split_tag: str) -> None:
    """Emit Kallini-format perturbed files for one v3 condition.

    Paths mirror kallini_repro trainer expectations:
      train: out_dir / f"babylm_{lang}" / "babylm_100M" / "all.train"
      test : out_dir / f"babylm_{lang}" / "babylm_test_affected" / "all_affected.test"
    """
    import json

    spec = CONDITIONS[lang]
    data = json.load(open(tagged_json))
    if split_tag == "100M":
        out_dir = out_dir / f"babylm_{lang}" / "babylm_100M"
        out_file = out_dir / "all.train"
    else:
        out_dir = out_dir / f"babylm_{lang}" / "babylm_test_affected"
        out_file = out_dir / "all_affected.test"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_file, "w") as f:
        for line in data:
            for sent in line.get("sent_annotations", []):
                toks = spec["fn"](sent)
                if len([t for t in toks if t != _neg_token_id()] ) <= 1:
                    continue
                if len(toks) > 350 or len(toks) <= 1:
                    continue
                f.write(" ".join(str(t) for t in toks) + "\n")
                n += 1
    print(f"{lang:16s} {split_tag}: {n} sentences -> {out_file}")
