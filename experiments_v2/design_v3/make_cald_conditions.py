#!/usr/bin/env python3
"""make_cald_conditions.py — CALD positive-evidence condition family (§10c-13 A2).

Generates three synthetic conditions in Kallini token-ID format:

    cald_local : [K][V] f1..fd          (V adjacent to K => bigram-solvable)
    cald_long  : [K] f1..fd [V]         (d ~ U{8,32}, content-addressed only)
    cald_shuf  : identical to cald_long but the V assignment is permuted
                 across sentences (V ⊥ K; surface statistics preserved)

Generator contract (FROZEN in prereg §10c-13 A2, 2026-09-24):
  * fillers sampled from the BabyLM natural BIGRAM transition distribution
    (local dependency is required for induction-head formation — Aoyama et al.
    2025 — identical sampling for all three conditions; cald_shuf permutes only
    the V assignment so the filler stream is byte-identical to cald_long);
  * K/V tokens are coined single-BPE-token words, disjoint from the filler
    vocabulary and absent from BabyLM;
  * K=200 pairs, d ~ U{8,32}, pre/post padding fillers bring sentence length
    into 24-48 tokens;
  * train ~100M tokens per condition (3000 steps ~= 3 epochs), test pool 60k
    sentences (the trainer eval draws 10k);
  * deterministic under --seed; writes a _cald_manifest.json audit file per
    condition (line/token counts, vocab disjointness, KV ids).

Output layout (train_exp1.py-compatible):
    $KALLINI_DATA_PATH/babylm_data_perturbed/babylm_cald_<cond>/babylm_100M/cald.train
    $KALLINI_DATA_PATH/babylm_data_perturbed/babylm_cald_<cond>/babylm_test_affected/cald_affected.test
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
from transformers import GPT2TokenizerFast

DATA_ROOT = Path(os.environ.get("KALLINI_DATA_PATH", "/root/kallini_data")) / "babylm_data_perturbed"
K_PAIRS = 200
D_RANGE = (8, 32)
PRE_RANGE = (0, 12)
POST_RANGE = (0, 12)
LEN_MIN, LEN_MAX = 24, 48
TRAIN_TOKENS = 100_000_000
TEST_SENTS = 60_000
FILLER_VOCAB = 5000          # token-level: keep this many most frequent ids
BIGRAM_SMOOTH = 0.02         # interpolated backoff to the unigram prior
COIN_BASE = ["quorl", "vexim", "zandul", "mirek", "tolvin", "grasho", "pelna",
             "wovik", "drasem", "kunefa", "yelbit", "sorfane", "huxomi",
             "blanet", "crodif", "tessup", "varmol", "jinqat", "fylor", "nexpus"]


def coin_words(n: int) -> list[str]:
    """n coined words (deterministic), each a SINGLE GPT-2 BPE token with a
    leading space, extending the base list with numbered variants if needed."""
    out, i = [], 0
    while len(out) < n:
        w = COIN_BASE[i % len(COIN_BASE)] + ("" if i < len(COIN_BASE) else str(i // len(COIN_BASE)))
        out.append(" " + w)
        i += 1
    return out


def load_babylm_stream(max_tokens: int = 6_000_000) -> list[int]:
    """Token-ID stream from the shared v2 class-P pool (any P condition dir)."""
    src = DATA_ROOT / "babylm_parity_word" / "babylm_100M"
    files = sorted(src.glob("*.train"))
    assert files, f"no train files under {src}"
    ids: list[int] = []
    for f in files:
        for line in f.read_text().splitlines():
            ids.extend(int(t) for t in line.split())
            if len(ids) >= max_tokens:
                return ids[:max_tokens]
    return ids[:max_tokens]


def build_models(stream: list[int]):
    """Filler vocab (top-FILLER_VOCAB frequent ids) + dense bigram probability
    matrix with unigram backoff; returns (vocab, CUM) where row CUM[i] is the
    cumulative next-token distribution, so sampling = one searchsorted."""
    counts = np.bincount(stream, minlength=50257).astype(np.float64)
    vocab = np.sort(np.argsort(counts)[::-1][:FILLER_VOCAB])
    idx_of = {int(t): i for i, t in enumerate(vocab.tolist())}
    V = len(vocab)
    a = np.asarray(stream, dtype=np.int64)
    s1, s2 = a[:-1], a[1:]
    keep = np.isin(s1, vocab) & np.isin(s2, vocab)
    i1 = np.array([idx_of[int(t)] for t in s1[keep].tolist()], dtype=np.int64)
    i2 = np.array([idx_of[int(t)] for t in s2[keep].tolist()], dtype=np.int64)
    B = np.zeros((V, V), dtype=np.float32)
    np.add.at(B, (i1, i2), 1.0)
    big = B / np.maximum(B.sum(axis=1, keepdims=True), 1e-9)
    uni = counts[vocab][None, :].astype(np.float32).repeat(V, axis=0)
    uni = uni / np.maximum(uni.sum(axis=1, keepdims=True), 1e-9)
    P = (1.0 - BIGRAM_SMOOTH) * big + BIGRAM_SMOOTH * uni
    CUM = np.cumsum(P, axis=1, dtype=np.float32)
    CUM[:, -1] = 1.0
    return vocab, CUM


def sample_fillers(CUM, n_sents: int, n_tok: int, rng, starts) -> np.ndarray:
    """Vectorized bigram-chain walk: (n_sents, n_tok) filler-vocab indices."""
    V = CUM.shape[0]
    out = np.empty((n_sents, n_tok), dtype=np.int64)
    state = starts
    for t in range(n_tok):
        u = rng.random(n_sents, dtype=np.float32)
        nxt = (CUM[state] <= u[:, None]).sum(axis=1)
        out[:, t] = np.clip(nxt, 0, V - 1)
        state = out[:, t]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-tokens", type=int, default=TRAIN_TOKENS)
    ap.add_argument("--test-sents", type=int, default=TEST_SENTS)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    pyrng = random.Random(args.seed)

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    kv_words = coin_words(2 * K_PAIRS)
    kv_ids = []
    for w in kv_words:
        e = tok.encode(w, add_special_tokens=False)
        assert len(e) == 1, f"coined word {w!r} is not a single BPE token: {e}"
        kv_ids.append(e[0])
    print(f"[cald] {len(kv_ids)} K/V tokens reserved (single-BPE verified)")

    stream = load_babylm_stream()
    vocab, CUM = build_models(stream)
    assert not (set(kv_ids) & set(vocab.tolist())), "K/V ids leaked into filler vocab"
    print(f"[cald] filler vocab V={len(vocab)}, transition matrix {CUM.shape}")

    k_ids = kv_ids[:K_PAIRS]
    v_ids = kv_ids[K_PAIRS:]

    # ---------------- sentence plans (shared across conditions) ----------------
    # one structured record per sentence: (pre_ids, k, filler_ids, v, post_ids)
    sents: list[tuple[list[int], int, list[int], int, list[int]]] = []
    total = 0
    batch = 20000
    while total < args.train_tokens:
        pre = np.fromiter((pyrng.randint(*PRE_RANGE) for _ in range(batch)), dtype=np.int64, count=batch)
        d = np.fromiter((pyrng.randint(*D_RANGE) for _ in range(batch)), dtype=np.int64, count=batch)
        post = np.fromiter((pyrng.randint(*POST_RANGE) for _ in range(batch)), dtype=np.int64, count=batch)
        need = pre + d + post
        maxlen = int(need.max())
        starts = rng.choice(len(vocab), size=batch).astype(np.int64)
        fill = sample_fillers(CUM, batch, maxlen, rng, starts)
        ki = rng.integers(0, K_PAIRS, size=batch)
        vi = rng.integers(0, K_PAIRS, size=batch)
        for b in range(batch):
            p, dd, q = int(pre[b]), int(d[b]), int(post[b])
            fillers = vocab[fill[b, :dd].tolist()].tolist()
            pre_f = vocab[fill[b, dd:dd + p].tolist()].tolist() if p else []
            post_f = vocab[fill[b, dd + p:dd + p + q].tolist()].tolist() if q else []
            k, v = int(k_ids[ki[b]]), int(v_ids[vi[b]])
            length = p + 1 + dd + 1 + q
            assert LEN_MIN <= length <= LEN_MAX, f"len {length} outside [{LEN_MIN},{LEN_MAX}]"
            sents.append((pre_f, k, fillers, v, post_f))
            total += length + 1  # +EOS, matches the trainer's packing
    n_train = len(sents)
    print(f"[cald] train plan: {n_train} sentences, {total} tokens (target {args.train_tokens})")

    test_plans = [(pyrng.randint(*PRE_RANGE), pyrng.randint(*D_RANGE), pyrng.randint(*POST_RANGE))
                  for _ in range(args.test_sents)]
    # V-permutation source for cald_shuf (V ⊥ K; fillers byte-identical to long)
    perm = rng.permutation(n_train)

    def emit(cond: str, variant: str) -> None:
        out_dir = DATA_ROOT / f"babylm_{cond}"
        (out_dir / "babylm_100M").mkdir(parents=True, exist_ok=True)
        (out_dir / "babylm_test_affected").mkdir(parents=True, exist_ok=True)
        n_tok = 0
        lines: list[str] = []
        for i, (pre_f, k, f, v, post_f) in enumerate(sents):
            vv = int(v_ids[perm[i] % K_PAIRS]) if variant == "shuf" else v
            kk = k  # K assignment identical in all variants; shuf breaks K→V only
            seq = (pre_f + [kk, vv] + f + post_f) if variant == "local" \
                else (pre_f + [kk] + f + [vv] + post_f)
            lines.append(" ".join(map(str, seq)))
            n_tok += len(seq) + 1
        (out_dir / "babylm_100M" / "cald.train").write_text("\n".join(lines) + "\n")

        tlines: list[str] = []
        for (p, dd, q) in test_plans:
            st = rng.choice(len(vocab), size=1).astype(np.int64)
            ff = vocab[sample_fillers(CUM, 1, dd, rng, st)[0].tolist()].tolist()
            pre_f = vocab[rng.choice(len(vocab), size=p).tolist()].tolist() if p else []
            post_f = vocab[rng.choice(len(vocab), size=q).tolist()].tolist() if q else []
            k = int(k_ids[rng.integers(0, K_PAIRS)])
            v = int(v_ids[rng.integers(0, K_PAIRS)])
            seq = (pre_f + [k, v] + ff + post_f) if variant == "local" \
                else (pre_f + [k] + ff + [v] + post_f)
            tlines.append(" ".join(map(str, seq)))
        (out_dir / "babylm_test_affected" / "cald_affected.test").write_text("\n".join(tlines) + "\n")

        man = {"condition": cond, "variant": variant, "seed": args.seed,
               "K": K_PAIRS, "d_range": D_RANGE, "len_range": [LEN_MIN, LEN_MAX],
               "train_sentences": n_train, "train_tokens": n_tok,
               "test_sentences": len(tlines), "filler_vocab": int(len(vocab)),
               "kv_ids": kv_ids, "bigram_smooth": BIGRAM_SMOOTH,
               "surface_gate": "content |Δln ppl|<=0.15 all pairs; all column long<->shuf only",
               "note": "fillers = BabyLM natural bigram transitions (Aoyama 2025 IH requirement)"}
        (out_dir / "_cald_manifest.json").write_text(json.dumps(man, indent=2))
        print(f"[cald] wrote {cond:12s}: {n_train} train sents / {n_tok} tokens, {len(tlines)} test")

    emit("cald_long", "long")
    emit("cald_local", "local")
    emit("cald_shuf", "shuf")
    print("[cald] done — next: bigram/5-gram surface-match verification (prereg A2 gate)")


if __name__ == "__main__":
    main()
