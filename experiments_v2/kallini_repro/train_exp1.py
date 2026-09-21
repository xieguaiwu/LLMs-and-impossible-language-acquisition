#!/usr/bin/env python3
"""Faithful reimplementation of Experiment 1 of Kallini et al. (2024),
"Mission: Impossible Language Models" (ACL 2024), scoped to the *Shuffle and
*Reverse language classes, for a single RTX-3080-class GPU.

Verbatim elements (from their MIT-licensed repo jkallini/mission-impossible-
language-models):
  * perturbation datasets produced by THEIR data/perturb.py + utils.py;
  * their packing: shuffle sentence-token-lines with numpy rng(seed), join
    with EOS, chunk into 1024-token blocks (their training/babylm_dataset.py);
  * their per-sentence perplexity function (their perplexities/perplexities.py,
    copied with attribution);
  * eval: sample 10,000 perturbed test sentences; per-sentence perplexity;
    we report the geometric mean (their Figure 2 quantity) per checkpoint;
  * GPT-2 small hyperparameters: seq 1024, warmup 300 steps to LR 6e-4
    (their Appendix B), stability flags reorder_and_upcast_attn +
    scale_attn_by_inverse_layer_idx, checkpoint ladder every 100 steps;
  * their random seeds [0, 14, 41, 53, 96] (we run the first --seeds of them).

Documented deviations (single-GPU compute; see kallini_repro/README.md):
  * effective batch 512 -> 128 (micro 8 x accum 16): tokens/run 1.57B -> 0.39B
    (micro batch is env-overridable: REPRO_MICRO_BATCH, see below)
    (~3 epochs of the 100M-word corpus);
  * LR schedule: warmup 300 to 6e-4, then linear decay to 0 by step 3000
    (paper specifies only the warmup; the 4000-warmup note implies decay);
  * eval checkpoint ladder [100,300,500,1000,2000,3000] (sparser than their
    100-step ladder), evaluated in-process at the exact checkpoint steps;
  * *Hop languages excluded (require Stanza POS/lemma tags; their Experiment 1
    perplexity differences within the Hop class are minimal anyway);
  * BabyLM corpus from the HF mirror (single genre file); sentence
    segmentation via regex shim instead of Stanza (Shuffle/Reverse
    perturbations do not depend on POS annotations).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path

# 10 GB card: the fp32-upcast GPT-2 loss + eval path is close to the limit, so
# reduce allocator fragmentation before torch initializes CUDA. setdefault, so an
# explicit PYTORCH_CUDA_ALLOC_CONF from the queue script still wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

# Stack portability (§10c-11, 2026-09-21): the second (Blackwell) host runs
# torch >= 2.7 where ``torch.cuda.amp.*`` is deprecated.  The shim is numerically
# transparent on the 3080 stack; ``pin_numerics()`` only re-asserts the values
# that were already in effect there.  See experiments_v2/kallini_repro/stack_compat.py
from stack_compat import amp_autocast, grad_scaler, pin_numerics, stack_metadata

# ---------------------------------------------------------------- config ----

KALLINI_REPO = Path(os.environ.get("KALLINI_REPO", "/root/mission-impossible-language-models"))
BABYLM_DATA_PATH = Path(os.environ.get("KALLINI_DATA_PATH", "/root/kallini_data"))
RESULTS = Path(os.environ.get("REPRO_RESULTS", Path(__file__).resolve().parent / "results"))
TRAIN_SET = os.environ.get("REPRO_TRAIN_SET", "100M")
MAX_STEPS = int(os.environ.get("REPRO_MAX_STEPS", 3000))
WARMUP_STEPS = int(os.environ.get("REPRO_WARMUP_STEPS", 300))
PEAK_LR = 6e-4
EVAL_CHECKPOINTS = [100, 300, 500, 1000, 2000, 3000]
EVAL_SAMPLE = 10000
EFF_BATCH = 128
MICRO_BATCH = int(os.environ.get("REPRO_MICRO_BATCH", 4))
# Why the default is 4 and not 8: on the target 10 GB RTX 3080 the step-1 forward
# of GPT-2-small at micro 8 x seq 1024 OOMs (GPT-2 upcasts the LM-head loss to
# fp32 and reorder_and_upcast_attn keeps a fp32 attention copy; the process
# reaches ~8.4 GiB allocated and the caching allocator then needs another 1.5
# GiB). Verified 2026-09-19: micro 8 dies before step 100, micro 4 trains.
# The Kallini effective batch is unaffected: the accumulation factor is derived
# from EFF_BATCH=128, so micro 4 only trades step count for memory. Set
# REPRO_MICRO_BATCH=8 on a GPU with enough memory to restore the original split.
SEQ_LEN = 1024
DEFAULT_SEEDS = [0, 14, 41, 53, 96]

# ---- 2026-09-20 evening expansion (prereg §10c) — all env-gated, default off,
# so the running grid's behavior is unchanged until a pass restart pulls this. ----
# NoPE arm (§10c-4): zero + freeze the positional embeddings. Equivalent to
# Kallini's own gpt2_no_positional_encoding_model.py (their NoPE ablation),
# which drops wpe entirely; a zeroed frozen wpe contributes the same zero
# vector without needing a custom model class.
NOPE = os.environ.get("GPT2_NOPE", "0") == "1"
# In-process ladder probe (§10c-3): at every eval-checkpoint step, run the P1
# branch-matched minimal pairs on the CURRENT weights, so rule-acquisition
# dynamics come out of the same cells instead of a final-checkpoint-only probe.
LADDER_PROBE = os.environ.get("LADDER_PROBE", "0") == "1"
LADDER_PROBE_CONDS = {"parity_word", "parity_tok", "negtok", "fixed_start",
                      "fixed_end", "not_random"}
# Model-scale arm (§10c): GPT-2 medium = 355M (n_embd 1024 / 24 layers / 16 heads).
MODEL_SIZE = os.environ.get("REPRO_MODEL_SIZE", "gpt2")
MODEL_SIZES = {"gpt2": dict(n_embd=768, n_layer=12, n_head=12),
               "gpt2_medium": dict(n_embd=1024, n_layer=24, n_head=16)}
# Dataset override for the data-scale / LOGO arms (§10c): a full directory name
# under babylm_data_perturbed (e.g. babylm_parity_word_sub1M). Empty = default.
DATA_SUBDIR = os.environ.get("REPRO_DATA_SUBDIR", "")
# Output-dir suffix (datascale arms share one tree; the scale must be in the
# cell name: seed0_sub1M). Grid-status manifests expect the untagged names for
# every existing arm, so the default stays "".
DIR_TAG = os.environ.get("REPRO_DIR_TAG", "")
assert MODEL_SIZE in MODEL_SIZES, f"unknown REPRO_MODEL_SIZE={MODEL_SIZE}"

LANGUAGES = [
    "shuffle_control",            # NoShuffle (English control)
    "shuffle_nondeterministic",
    "shuffle_deterministic21",
    "shuffle_local3",
    "shuffle_local10",
    "shuffle_even_odd",
    "reverse_control",            # NoReverse (English + R-marker control)
    "reverse_partial",
    "reverse_full",
    # --- v3 class-P conditions (DESIGN_V3 §1.1; datasets via design_v3/v3_conditions.py)
    "parity_word",                # paper's counting rule (word domain)
    "parity_tok",                 # counting rule over BPE tokens (BabyLM-only meaningful)
    "negtok",                     # word parity, reserved <NEG> marker (vocab +1)
    "fixed_start",                # primary marker control of class P
    "fixed_end",                  # position control
    "bare_reverse",               # Reverse-bare (no marker; NOT a Kallini replication cell)
    "word_shuffle",               # our v2 Kallini-analog reference
    "not_random",                 # entropy-matched marker control (audit 2026-09-20, B1)
]
VOCAB_EXTRA = {"negtok": 1, "reverse_control": 1, "reverse_partial": 1, "reverse_full": 1}
# v3 class-P conditions (DESIGN_V3 §1.1). They live outside Kallini's
# PERTURBATIONS registry and use the marker-free GPT-2 tokenizer, so the
# trainer must not look them up in PERTURBATIONS.
V3_CONDITIONS = [
    "parity_word", "parity_tok", "negtok", "fixed_start", "fixed_end",
    "bare_reverse", "word_shuffle", "not_random",
]
# Marker token ids masked by the content-only (marker-masked) robustness metric:
#   1892 = " Not" (sentence-final form), 3673 = "Not" (sentence-initial form),
#   50257 = the reserved <NEG> / R marker slot (vocab +1 conditions).
# Documented caveat: a natural "Not" in the base sentence is masked too; the
# metric is a robustness check, not a re-definition of the primary endpoint.
MARKER_IDS = (1892, 3673, 50257, 50258)

sys.path.insert(0, str(KALLINI_REPO))
from utils import PERTURBATIONS, gpt2_original_tokenizer  # noqa: E402

EOS_TOKEN_ID = 50256


def dataset_key_of(dataset: str) -> str:
    """svo / svo_polluted -> svo-keyed budgets; babylm stays itself."""
    return dataset


def data_subdir_of(perturbation: str) -> str:
    """Directory name of a condition's data (REPRO_DATA_SUBDIR override wins).

    The data-scale / LOGO arms point this at a generated variant directory
    (``babylm_<cond>_sub1M`` etc.) while keeping ``perturbation`` itself a
    registered condition (tokenizer resolution stays valid).
    """
    return DATA_SUBDIR or f"babylm_{perturbation}"


# ----------------------------------------------------------- data packing ---

def load_packed_dataset(perturbation: str, seed: int) -> list[list[int]]:
    """Their babylm_dataset.py packing: shuffle token-ID sentences with
    numpy rng(seed), join with EOS, chunk into SEQ_LEN blocks."""
    data_dir = BABYLM_DATA_PATH / "babylm_data_perturbed" / data_subdir_of(perturbation) / f"babylm_{TRAIN_SET}"
    files = sorted(data_dir.glob("*.train"))
    assert files, f"no perturbed train files for {perturbation} under {data_dir}"
    all_sentences: list[str] = []
    for f in files:
        all_sentences.extend(f.read_text().splitlines())
    rng = np.random.default_rng(seed)
    rng.shuffle(all_sentences)
    all_tokens: list[int] = []
    for line in all_sentences:
        toks = [int(t) for t in line.split()]
        if not toks:
            continue
        all_tokens.extend(toks)
        all_tokens.append(EOS_TOKEN_ID)
    blocks = [all_tokens[i : i + SEQ_LEN] for i in range(0, len(all_tokens), SEQ_LEN)]
    # Their __chunk drops a trailing partial block (babylm_dataset.py: "Drop
    # last line if not a multiple of max_seq_len"). Without this, the short tail
    # eventually lands in a batch and kills every run at a random step:
    #   ValueError: expected sequence of length 1024 at dim 1 (got 167)
    # (a 100-step smoke cannot see it: the tail is drawn uniformly over the
    # first ~1038 steps, so it fired around step 500-1000 in production).
    if blocks and len(blocks[-1]) < SEQ_LEN:
        blocks.pop()
    assert blocks, f"no full {SEQ_LEN}-token block in {data_dir}"
    assert all(len(b) == SEQ_LEN for b in blocks), (
        "packed blocks must all be SEQ_LEN; a partial block would break batching")
    return blocks


def _sentence_id(line: str) -> str:
    """Stable 16-hex id of one evaluation sentence (its token-id line)."""
    return hashlib.blake2b(line.encode(), digest_size=8).hexdigest()


def _near_dup_rate(lines: list[str], sample: int = 20000, seed: int = 0) -> float:
    """Sampled near-duplicate rate (normalised text, cheap proxy for MinHash).

    The design asks for the near-duplication of the evaluation pool to be
    REPORTED (not filtered): BabyLM is transcript-heavy, so absolute ppl levels
    and any "held-out" claim depend on it. Normalisation = lowercase, strip
    punctuation, collapse whitespace; a sentence is a near-duplicate when its
    normalised form occurs more than once inside the sample.
    """
    if not lines:
        return 0.0
    rng = np.random.default_rng(seed)
    take = min(sample, len(lines))
    idx = rng.choice(len(lines), take, replace=False)
    seen: dict[str, int] = {}
    for i in idx:
        key = re.sub(r"[^a-z0-9 ]", "", lines[i].lower())
        key = re.sub(r"\s+", " ", key).strip()
        seen[key] = seen.get(key, 0) + 1
    return round(1.0 - len(seen) / take, 6)


class EvalSample(list):
    """The 10k evaluation draw plus its hygiene metadata (audit B5).

    A list subclass, so existing callers (``eval_sents[i:j]``, ``len()``) keep
    working unchanged — the metadata rides along without touching the protocol.
    """

    ids: list[str]
    fingerprint: str
    pool_n: int
    pool_exact_dups: int
    exact_dups_removed: int
    near_dup_rate_sampled: float | None


def load_eval_sentences(perturbation: str, seed: int, n: int = EVAL_SAMPLE,
                        dedup: bool = False) -> "EvalSample":
    """Their protocol: perturbed test sentences; sample n via numpy rng(seed).

    2026-09-20 (audit B5) adds the hygiene the frozen design requires but the
    code never did:
      * measure the pool's exact-duplication and sampled near-duplication rates
        (measured: **20.1 % exact duplicates** in the 987,793-sentence test pool
        — REDTEAM #7's warning, now quantified);
      * emit a stable per-sentence id list + an order fingerprint, so two cells
        can be aligned sentence-for-sentence and the duplicate-free subset can
        be selected **in the analysis stage** (``dedup_positions``).

    Why the draw itself stays unfiltered by default: the architecture axis
    compares GPT-2 against the CPU LSTM arm, and the CPU arm does not save
    weights (``LSTM_SAVE_CKPT=0``), so its sentence-level ppl is baked in. A
    filtered draw would change the sampled sentences and silently break
    like-for-like comparability with those cells. Keeping the draw identical and
    filtering the metric at analysis time gives every cell (past and future) the
    same treatment. ``dedup=True`` is available for new arms that want the
    filtered draw directly.

    The draw is numpy rng(seed).choice over the pool in file order, so the
    LSTM arm's "first 2000 of the same 10k sample" nesting still holds (it
    imports this function).
    """
    data_dir = BABYLM_DATA_PATH / "babylm_data_perturbed" / data_subdir_of(perturbation) / "babylm_test_affected"
    files = sorted(data_dir.glob("*_affected.test"))
    lines: list[str] = []
    for f in files:
        lines.extend(l for l in f.read_text().splitlines() if l.strip())
    pool_n = len(lines)

    # pool-level duplication statistics (always measured, never silent)
    seen_pool: set[str] = set()
    pool_dups = 0
    for line in lines:
        if line in seen_pool:
            pool_dups += 1
        else:
            seen_pool.add(line)
    del seen_pool

    exact_dups_removed = 0
    if dedup:
        uq: list[str] = []
        seen: set[str] = set()
        for line in lines:
            if line in seen:
                exact_dups_removed += 1
                continue
            seen.add(line)
            uq.append(line)
        lines = uq

    near_rate = _near_dup_rate(lines, sample=20000, seed=seed)

    if len(lines) > n:
        idx = np.random.default_rng(seed).choice(len(lines), n, replace=False)
        chosen = [lines[i] for i in idx]
    else:
        chosen = list(lines)

    sample = EvalSample([int(t) for t in line.split()] for line in chosen)
    sample.ids = [_sentence_id(line) for line in chosen]
    sample.fingerprint = hashlib.sha256("\n".join(sample.ids).encode()).hexdigest()[:16]
    sample.pool_n = pool_n
    sample.pool_exact_dups = pool_dups
    sample.exact_dups_removed = exact_dups_removed
    sample.near_dup_rate_sampled = near_rate
    return sample


def dedup_positions(ids: list[str]) -> list[int]:
    """Positions of the first occurrence of each sentence in an eval draw.

    Analysis-stage equivalent of "filter the pool before drawing": every cell
    (including the CPU-arm cells whose weights were discarded) can drop its
    duplicate sentences from the saved per-sentence ppl arrays with this index
    vector, without re-running anything.
    """
    seen: set[str] = set()
    keep: list[int] = []
    for i, k in enumerate(ids):
        if k in seen:
            continue
        seen.add(k)
        keep.append(i)
    return keep


# ------------------------------------------------------------------ eval ---

def create_attention_mask(token_lists):
    seq_length = max(len(i) for i in token_lists)
    mask = torch.zeros((len(token_lists), seq_length), dtype=torch.long)
    for i, tokens in enumerate(token_lists):
        mask[i, : len(tokens)] = 1
    return mask


def create_input_ids(token_lists, pad_token_id):
    """Their create_input_ids: the outer zip(*) undoes zip_longest's transpose.

    Without it the batch comes back as (L, B) while create_attention_mask
    returns (B, L), and get_perplexities dies at `loss * shift_attention_mask`
    with "The size of tensor a (31) must match the size of tensor b (23)" —
    i.e. the eval path crashed for every run at the first checkpoint
    (2026-09-19). The assertion keeps that transposition from coming back.
    """
    padded = zip(*zip_longest(*token_lists, fillvalue=pad_token_id))
    ids = torch.tensor(list(padded), dtype=torch.long)
    assert ids.shape[0] == len(token_lists), (
        f"create_input_ids returned {tuple(ids.shape)}; expected "
        f"({len(token_lists)}, L) rows-per-example — the outer zip(*) is missing")
    return ids


def get_perplexities(model, token_lists, pad_token_id, device="cuda", marker_ids=None):
    """Verbatim from Kallini et al. 2024 perplexities/perplexities.py (MIT).

    Returns a list of per-sentence ppl. When ``marker_ids`` is given, returns a
    TUPLE ``(ppls_all, ppls_content)`` where the second list excludes the marker
    positions from each sentence's mean loss — the frozen design's
    content-token-only robustness metric (DESIGN_V3 §A.2 / REDTEAM #2c).
    """
    input_ids = create_input_ids(token_lists, pad_token_id).to(device)
    labels = input_ids.clone()
    attention_mask = create_attention_mask(token_lists).to(device)
    outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_attention_mask = attention_mask[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
    loss = loss.view(shift_labels.size())
    loss = loss * shift_attention_mask
    per_example_loss = loss.sum(dim=1) / shift_attention_mask.sum(dim=1)
    ppls = torch.exp(per_example_loss).tolist()
    if marker_ids is None:
        return ppls
    ids = torch.tensor(sorted(marker_ids), device=shift_labels.device)
    is_marker = torch.isin(shift_labels, ids) & shift_attention_mask.bool()
    content_mask = (~is_marker).to(loss.dtype)
    content_loss = (loss * content_mask).sum(dim=1) / content_mask.sum(dim=1).clamp(min=1)
    return ppls, torch.exp(content_loss).tolist()


def evaluate_checkpoint(model, eval_sents, device="cuda", batch=8, marker_ids=None) -> dict:
    """Per-checkpoint perplexity. batch=8 mirrors their BATCH_SIZE in
    perplexities/perplexities.py (the port used 32, which OOMs a 10 GB card at
    the first checkpoint: 32 x ~350 tokens of fp32-upcast loss). Padding is
    masked per example, so the batch size does not change the numbers.

    ``marker_ids`` additionally yields the content-token-only (marker-masked)
    gmean, i.e. the same forward pass carries both metrics.
    """
    ppls: list[float] = []
    ppls_content: list[float] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(eval_sents), batch):
            chunk = [s[:SEQ_LEN] for s in eval_sents[i : i + batch] if len(s) >= 2]
            if not chunk:
                continue
            out = get_perplexities(model, chunk, EOS_TOKEN_ID, device, marker_ids)
            if marker_ids is None:
                ppls.extend(out)
            else:
                ppls.extend(out[0])
                ppls_content.extend(out[1])
    log_ppls = [math.log(p) for p in ppls]
    res = {
        "n": len(ppls),
        "gmean_ppl": round(float(math.exp(sum(log_ppls) / len(log_ppls))), 4),
        "mean_ppl": round(float(sum(ppls) / len(ppls)), 4),
        "ppls": [round(p, 4) for p in ppls],
    }
    if ppls_content:
        log_c = [math.log(p) for p in ppls_content]
        res["gmean_ppl_content"] = round(float(math.exp(sum(log_c) / len(log_c))), 4)
        res["ppls_content"] = [round(p, 4) for p in ppls_content]
        res["n_marker_masked"] = int(len(ppls_content))
    return res


# --------------------------------------------------------------- training ---

def lr_lambda(step: int) -> float:
    if step < WARMUP_STEPS:
        return step / max(1, WARMUP_STEPS)
    return max(0.0, (MAX_STEPS - step) / max(1, MAX_STEPS - WARMUP_STEPS))


def eval_checkpoints_for(steps: int) -> list[int]:
    """DESIGN_V3 ladder: 6 log-spaced points, scaled for extended budgets."""
    base = [100, 300, 500, 1000, 2000, 3000]
    if steps <= 3000:
        return [s for s in base if s <= steps] or [steps]
    pts = sorted({100, 300, 500, 1000, 2000, 3000, int(steps * 0.67), steps})
    return [s for s in pts if s <= steps]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenizer_for(perturbation: str):
    """Their tokenizer registry, extended with the v3 class-P conditions.

    v3 conditions are absent from PERTURBATIONS, so they must not be looked up
    there (a bare PERTURBATIONS[name] raises KeyError before any v3 handling).
    """
    spec = PERTURBATIONS.get(perturbation)
    if spec is not None:
        return spec["gpt2_tokenizer"]
    assert perturbation in V3_CONDITIONS, f"unknown perturbation: {perturbation}"
    tokenizer = gpt2_original_tokenizer
    if perturbation == "negtok":
        # marker-free GPT-2 plus the reserved <NEG> token (vocab +1, DESIGN_V3
        # §1.3.4), mirroring VOCAB_EXTRA and design_v3/v3_conditions.py
        tokenizer = type(tokenizer).from_pretrained("gpt2")
        tokenizer.add_special_tokens({"additional_special_tokens": ["<NEG>"]})
    return tokenizer


def train_one(perturbation: str, seed: int, out_dir: Path, device: str = "cuda",
              max_steps: int = None, warmup: int = None) -> dict:
    global MAX_STEPS, WARMUP_STEPS
    if max_steps:
        MAX_STEPS = max_steps
        WARMUP_STEPS = warmup
    from transformers import GPT2Config, GPT2LMHeadModel

    tokenizer = tokenizer_for(perturbation)
    reverse_mode = perturbation.startswith("reverse")
    vocab_extra = VOCAB_EXTRA.get(perturbation, 0)

    t_pack = time.time()
    blocks = load_packed_dataset(perturbation, seed)
    eval_sents = load_eval_sentences(perturbation, seed)
    print(f"[data] blocks={len(blocks)} eval_sents={len(eval_sents)} "
          f"({time.time()-t_pack:.0f}s)", flush=True)

    set_seed(seed)
    dims = MODEL_SIZES[MODEL_SIZE]
    config = GPT2Config(
        vocab_size=50257 + vocab_extra,
        n_positions=SEQ_LEN, n_embd=dims["n_embd"], n_layer=dims["n_layer"],
        n_head=dims["n_head"],
        resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1,
        reorder_and_upcast_attn=True, scale_attn_by_inverse_layer_idx=True,
    )
    model = GPT2LMHeadModel(config).to(device)
    if NOPE:
        # NoPE arm (§10c-4): zero + freeze the positional embeddings — same
        # semantics as Kallini's gpt2_no_positional_encoding_model.py (wpe
        # removed); the causal mask is the only remaining order signal.
        with torch.no_grad():
            model.transformer.wpe.weight.zero_()
        model.transformer.wpe.weight.requires_grad_(False)

    accum = EFF_BATCH // MICRO_BATCH
    assert accum * MICRO_BATCH == EFF_BATCH, (
        f"REPRO_MICRO_BATCH={MICRO_BATCH} does not divide the effective batch "
        f"{EFF_BATCH}; the gradient accumulation must reproduce the Kallini "
        f"effective batch exactly")
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    pin_numerics()
    scaler = grad_scaler()

    rng = np.random.default_rng(seed + 1)
    n_blocks = len(blocks)
    order = rng.permutation(n_blocks)
    ptr = 0

    def next_batch() -> list[list[int]]:
        """One MICRO batch: the accumulation loop assembles the effective batch.

        This used to fill EFF_BATCH (128) rows per call while the training loop
        called it `accum` times per step, so the step-1 forward was 128 x 1024
        tokens and the real effective batch was 128 x accum (4096) instead of
        the Kallini-faithful 128. On a 10 GB card that OOMs immediately
        (2026-09-19), and it silently broke the documented protocol.
        """
        nonlocal ptr, order
        batch: list[list[int]] = []
        while len(batch) < MICRO_BATCH:
            if ptr >= n_blocks:
                order = rng.permutation(n_blocks)
                ptr = 0
            batch.append(blocks[order[ptr]])
            ptr += 1
        return batch

    # Ladder probe wiring (§10c): one frozen pair set per condition, loaded
    # lazily so cells that do not need it pay nothing.
    probe_pairs = None
    if LADDER_PROBE and perturbation in LADDER_PROBE_CONDS:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from probes.probes_babylm import load_base_pool, make_pairs, run_p1
            probe_pool = load_base_pool(limit=20000)
            probe_pairs = make_pairs(probe_pool, 200,
                                     domain="tok" if perturbation == "parity_tok" else "word",
                                     seed=42)
            print(f"[probe] ladder probe armed: {len(probe_pairs)} branch-matched pairs", flush=True)
        except Exception as e:                       # analysis-side, never blocks the grid
            print(f"[probe] init failed ({e}); ladder probe disabled", flush=True)
    probe_trace: dict[str, dict] = {}

    eval_trace: dict[str, float] = {}
    eval_content_trace: dict[str, float] = {}
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)   # before the first checkpoint save
    model.train()
    checkpoints = eval_checkpoints_for(MAX_STEPS)
    for step in range(1, MAX_STEPS + 1):
        opt.zero_grad(set_to_none=True)
        loss_avg = 0.0
        for _ in range(accum):
            batch = next_batch()
            input_ids = torch.tensor(batch, dtype=torch.long, device=device)
            with amp_autocast():
                out = model(input_ids=input_ids, labels=input_ids.clone())
            scaler.scale(out.loss / accum).backward()
            loss_avg += float(out.loss) / accum
        # 2026-09-20 fix (§10c, P2): the clip used to run on the loss-scaled
        # gradients without unscale_, which normalized every step to unit true
        # norm (a different optimizer regime than the registered clip@1.0).
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        if step in checkpoints:
            trace = evaluate_checkpoint(model, eval_sents, device,
                                        batch=int(os.environ.get("REPRO_EVAL_BATCH", 8)),
                                        marker_ids=MARKER_IDS)
            eval_trace[str(step)] = trace["gmean_ppl"]
            if "gmean_ppl_content" in trace:
                eval_content_trace[str(step)] = trace["gmean_ppl_content"]
            torch.save(trace["ppls"], out_dir / f"ppls_step{step}.pt")
            if "ppls_content" in trace:
                torch.save(trace["ppls_content"], out_dir / f"ppls_content_step{step}.pt")
            if probe_pairs is not None:
                try:
                    probe_trace[str(step)] = run_p1(model, probe_pairs, device)
                except Exception as e:               # analysis-side, never blocks the grid
                    print(f"[probe] step {step} failed: {e}", flush=True)
            model.train()                            # 2026-09-20 fix (§10c, P1):
            # evaluate_checkpoint() left the model in eval() mode and nothing
            # restored it, so every cell silently trained WITHOUT dropout from
            # its first eval checkpoint on (uniform across all cells, but a
            # deviation from the registered protocol and from Kallini).
            print(f"[eval] {perturbation} seed{seed} step {step}: "
                  f"gmean_ppl={trace['gmean_ppl']} "
                  f"content={trace.get('gmean_ppl_content')} (n={trace['n']})", flush=True)
        if step % 100 == 0:
            print(f"[train] {perturbation} seed{seed} step {step}/{MAX_STEPS} "
                  f"loss={loss_avg:.4f} elapsed={(time.time()-t0)/60:.1f}m", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir / "final")
    tokenizer.save_pretrained(out_dir / "final")
    return {
        "language": perturbation,
        "train_set": TRAIN_SET,
        "seed": seed,
        "max_steps": MAX_STEPS,
        "effective_batch": EFF_BATCH,
        "micro_batch": MICRO_BATCH,
        "seq_len": SEQ_LEN,
        "peak_lr": PEAK_LR,
        "warmup": WARMUP_STEPS,
        "model_size": MODEL_SIZE,
        "nope": NOPE,
        "data_subdir": DATA_SUBDIR or f"babylm_{perturbation}",
        "dir_tag": DIR_TAG,
        "dropout_active_all_steps": True,
        "grad_clip_true_norm": True,
        "ladder_probe": probe_trace if probe_trace else None,
        "n_blocks": len(blocks),
        "eval_gmean": eval_trace,
        "eval_gmean_content": {k: v for k, v in eval_content_trace.items()},
        "eval_n": len(eval_sents),
        "eval_pool_n": getattr(eval_sents, "pool_n", None),
        "eval_pool_exact_dups": getattr(eval_sents, "pool_exact_dups", None),
        "eval_fingerprint": getattr(eval_sents, "fingerprint", None),
        "eval_exact_dups_removed": getattr(eval_sents, "exact_dups_removed", None),
        "eval_near_dup_rate_sampled": getattr(eval_sents, "near_dup_rate_sampled", None),
        "marker_ids_masked": list(MARKER_IDS),
        "wall_time_s": round(time.time() - t0, 1),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "stack": stack_metadata(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("perturbation", choices=LANGUAGES)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--skip-if-done", action="store_true")
    parser.add_argument("--steps", type=int, default=None,
                        help="override MAX_STEPS (H7 budget ladder); warmup stays at 10%%")
    args = parser.parse_args()

    if args.steps:
        global MAX_STEPS, WARMUP_STEPS, EVAL_CHECKPOINTS
        MAX_STEPS = args.steps
        WARMUP_STEPS = max(300, int(0.10 * args.steps))

    out_dir = RESULTS / f"babylm_{args.perturbation}_{TRAIN_SET}" / \
        (f"steps{args.steps}_seed{args.seed}{DIR_TAG}" if args.steps
         else f"seed{args.seed}{DIR_TAG}")
    done_marker = out_dir / "exp1_result.json"
    if args.skip_if_done and done_marker.exists():
        print(f"SKIP {done_marker} (already complete)")
        return

    result = train_one(args.perturbation, args.seed, out_dir, max_steps=MAX_STEPS, warmup=WARMUP_STEPS)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(done_marker, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["eval_gmean"], indent=2))


if __name__ == "__main__":
    main()
