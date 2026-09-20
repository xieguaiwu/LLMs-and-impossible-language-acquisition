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
  not_random     NEW (2026-09-20, audit B1): marker placed at start/end with the
                 SAME marginal position distribution as parity_word, but with the
                 position INDEPENDENT of the sentence's parity (the exact multiset
                 of parity_word position flags, deterministically permuted across
                 sentences). This is Kallini's NoReverse move for the \*Reverse
                 class (marker at a random position, no transformation) applied to
                 class P: it isolates "marker position depends on the word count"
                 from "the marker sits at a 50/50 start-or-end position".
  bare_reverse   whole-sentence reversal, no marker (labeled Reverse-bare;
                 replication claims are made ONLY by kallini_repro's
                 reverse_full / reverse_control pair).
  word_shuffle   per-sentence random shuffle (nondeterministic), our own
                 Kallini-NondeterministicShuffle analog used in v2.

Shared sentence filter (2026-09-20, audit B5): every condition keeps exactly the
sentences whose **base** tokenization has 1 < n <= 350 tokens (Kallini's
``filter_shuffle`` semantics: the filter sees the unperturbed sentence, so the
marker or the transformation can never move a sentence in or out of the pool).
Before this change the filter was applied to the *perturbed* token count, so
conditions whose transform adds a token (all markered ones) silently dropped a
different sentence set than ``negtok``/the S/R classes. Consequence: all class-P
conditions now share one sentence set by construction (verified by
``kallini_repro/data_integrity_check.py``), matching EXPDESIGN_V3 §1.3.2.

All functions take a Kallini-style sentence annotation dict ({"sent_text": ...})
and return a list of token ids.``base`` may be passed in to avoid re-tokenizing.
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


def _base_of(sent: dict, base: list[int] | None) -> list[int]:
    return _base_ids(sent) if base is None else base


def _word_count(sent: dict) -> int:
    """The counting domain of ``parity_word`` (same expression as before)."""
    return len(sent["sent_text"].strip().rstrip(".?!").split())


def _not_start() -> list[int]:
    return gpt2_original_tokenizer.encode("Not", add_special_tokens=False)


def _not_end() -> list[int]:
    return gpt2_original_tokenizer.encode(" Not", add_special_tokens=False)


# ------------------------------------------------------------- conditions ---

def perturb_parity_word(sent: dict, base: list[int] | None = None, **kw) -> list[int]:
    """Paper's rule: even word count -> marker first; odd -> marker last."""
    toks = _base_of(sent, base)
    if _word_count(sent) % 2 == 0:
        return _not_start() + toks
    return toks + _not_end()


def perturb_parity_tok(sent: dict, base: list[int] | None = None, **kw) -> list[int]:
    toks = _base_of(sent, base)
    # parity domain = BPE tokens of the sentence BEFORE marker insertion
    # (punctuation included — the model's actual countable units)
    if len(toks) % 2 == 0:
        return _not_start() + toks
    return toks + _not_end()


def perturb_fixed_start(sent: dict, base: list[int] | None = None, **kw) -> list[int]:
    return _not_start() + _base_of(sent, base)


def perturb_fixed_end(sent: dict, base: list[int] | None = None, **kw) -> list[int]:
    return _base_of(sent, base) + _not_end()


def perturb_not_random(sent: dict, base: list[int] | None = None,
                       position: int | None = None, **kw) -> list[int]:
    """Entropy-matched, rule-free marker control (audit B1).

    ``position`` is supplied by ``write_condition`` from the deterministic
    permutation of ``parity_word``'s position flags (1 = sentence-final).
    Standalone calls (no position) fall back to a per-sentence hash draw with
    the same marginal probability — used only by ad-hoc tooling.
    """
    toks = _base_of(sent, base)
    if position is None:
        position = int(_word_count(sent) % 2)   # degenerate fallback = parity
    return toks + _not_end() if position else _not_start() + toks


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


def perturb_negtok(sent: dict, base: list[int] | None = None, **kw) -> list[int]:
    toks = _base_of(sent, base)
    nid = _neg_token_id()
    if _word_count(sent) % 2 == 0:
        return [nid] + toks
    return toks + [nid]


def perturb_bare_reverse(sent: dict, base: list[int] | None = None, **kw) -> list[int]:
    return _base_of(sent, base)[::-1]


def perturb_word_shuffle(sent: dict, base: list[int] | None = None, **kw) -> list[int]:
    toks = _base_of(sent, base)
    # deterministic per-sentence shuffle (numpy-free): Fisher-Yates from hash
    # (2026-09-19: this used to call .hexdigest() on the bytes returned by
    # .digest(), so word_shuffle raised AttributeError on the first sentence and
    # the whole v3 data block aborted after ~2h of regenerating the other six
    # conditions)
    digest = hashlib.sha256(("shuffle:" + sent["sent_text"]).encode()).hexdigest()
    order = list(range(len(toks)))
    h = int(digest, 16)
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
    "parity_word": {"fn": perturb_parity_word, "vocab_add": 0},
    "parity_tok": {"fn": perturb_parity_tok, "vocab_add": 0},
    "negtok": {"fn": perturb_negtok, "vocab_add": 1},
    "fixed_start": {"fn": perturb_fixed_start, "vocab_add": 0},
    "fixed_end": {"fn": perturb_fixed_end, "vocab_add": 0},
    # position flags come from parity_word (exact same multiset, permuted)
    "not_random": {"fn": perturb_not_random, "vocab_add": 0,
                   "positions_from": "parity_word"},
    "bare_reverse": {"fn": perturb_bare_reverse, "vocab_add": 0},
    "word_shuffle": {"fn": perturb_word_shuffle, "vocab_add": 0},
}


def write_condition(lang: str, tagged_json: Path, out_dir: Path, split_tag: str,
                    verbose: bool = True) -> dict:
    """Emit Kallini-format perturbed files for one v3 condition.

    Paths mirror kallini_repro trainer expectations:
      train: out_dir / f"babylm_{lang}" / "babylm_100M" / "{genre}_parsed.train"
      test : out_dir / f"babylm_{lang}" / "babylm_test_affected" / "{genre}_parsed_affected.test"

    Sentence pool = the shared base-token filter (1 < base tokens <= 350) for
    every condition (see the module docstring), so the pools are identical by
    construction and the emitted count is a hard gate value.
    """
    import json

    spec = CONDITIONS[lang]
    data = json.load(open(tagged_json))
    if split_tag == "100M":
        out_dir = out_dir / f"babylm_{lang}" / "babylm_100M"
        out_file = out_dir / f"{tagged_json.stem}.train"
    else:
        out_dir = out_dir / f"babylm_{lang}" / "babylm_test_affected"
        out_file = out_dir / f"{tagged_json.stem}_affected.test"
    out_dir.mkdir(parents=True, exist_ok=True)

    # pass 1 — shared sentence pool (identical across conditions by construction)
    kept: list[tuple[dict, list[int]]] = []
    for line in data:
        for sent in line.get("sent_annotations", []):
            base = _base_ids(sent)
            if 1 < len(base) <= 350:
                kept.append((sent, base))

    # optional exact-marginal position assignment (not_random)
    positions: list[int] | None = None
    src = spec.get("positions_from")
    if src:
        positions = [1 if _word_count(sent) % 2 else 0 for sent, _ in kept]
        rng = random.Random(f"{lang}:{tagged_json.stem}")
        rng.shuffle(positions)          # exact same multiset, parity correlation destroyed

    n = 0
    n_pos = 0
    with open(out_file, "w") as f:
        for idx, (sent, base) in enumerate(kept):
            if positions is not None:
                toks = spec["fn"](sent, base=base, position=positions[idx])
                n_pos += positions[idx]
            else:
                toks = spec["fn"](sent, base=base)
            assert 1 < len(base) <= 350
            f.write(" ".join(str(t) for t in toks) + "\n")
            n += 1
    if verbose:
        extra = f" markers_at_end={n_pos} ({n_pos / n:.4f})" if positions is not None else ""
        print(f"{lang:16s} {split_tag}: {n} sentences -> {out_file}{extra}")
    return {"lang": lang, "split": split_tag, "file": str(out_file), "n": n,
            "markers_at_end": n_pos if positions is not None else None}
