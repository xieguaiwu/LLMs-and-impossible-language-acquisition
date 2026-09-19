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
import json
import math
import os
import random
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
]
VOCAB_EXTRA = {"negtok": 1, "reverse_control": 1, "reverse_partial": 1, "reverse_full": 1}
# v3 class-P conditions (DESIGN_V3 §1.1). They live outside Kallini's
# PERTURBATIONS registry and use the marker-free GPT-2 tokenizer, so the
# trainer must not look them up in PERTURBATIONS.
V3_CONDITIONS = [
    "parity_word", "parity_tok", "negtok", "fixed_start", "fixed_end",
    "bare_reverse", "word_shuffle",
]

sys.path.insert(0, str(KALLINI_REPO))
from utils import PERTURBATIONS, gpt2_original_tokenizer  # noqa: E402

EOS_TOKEN_ID = 50256


def dataset_key_of(dataset: str) -> str:
    """svo / svo_polluted -> svo-keyed budgets; babylm stays itself."""
    return dataset


# ----------------------------------------------------------- data packing ---

def load_packed_dataset(perturbation: str, seed: int) -> list[list[int]]:
    """Their babylm_dataset.py packing: shuffle token-ID sentences with
    numpy rng(seed), join with EOS, chunk into SEQ_LEN blocks."""
    data_dir = BABYLM_DATA_PATH / "babylm_data_perturbed" / f"babylm_{perturbation}" / f"babylm_{TRAIN_SET}"
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
    return [all_tokens[i : i + SEQ_LEN] for i in range(0, len(all_tokens), SEQ_LEN)]


def load_eval_sentences(perturbation: str, seed: int, n: int = EVAL_SAMPLE) -> list[list[int]]:
    """Their protocol: perturbed test sentences; sample n via numpy rng(seed)."""
    data_dir = BABYLM_DATA_PATH / "babylm_data_perturbed" / f"babylm_{perturbation}" / "babylm_test_affected"
    files = sorted(data_dir.glob("*_affected.test"))
    seqs: list[list[int]] = []
    for f in files:
        seqs.extend([int(t) for t in l.split()] for l in f.read_text().splitlines() if l.strip())
    if len(seqs) > n:
        idx = np.random.default_rng(seed).choice(len(seqs), n, replace=False)
        seqs = [seqs[i] for i in idx]
    return seqs


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


def get_perplexities(model, token_lists, pad_token_id, device="cuda"):
    """Verbatim from Kallini et al. 2024 perplexities/perplexities.py (MIT)."""
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
    return torch.exp(per_example_loss).tolist()


def evaluate_checkpoint(model, eval_sents, device="cuda", batch=8) -> dict:
    """Per-checkpoint perplexity. batch=8 mirrors their BATCH_SIZE in
    perplexities/perplexities.py (the port used 32, which OOMs a 10 GB card at
    the first checkpoint: 32 x ~350 tokens of fp32-upcast loss). Padding is
    masked per example, so the batch size does not change the numbers."""
    ppls: list[float] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(eval_sents), batch):
            chunk = [s[:SEQ_LEN] for s in eval_sents[i : i + batch] if len(s) >= 2]
            if not chunk:
                continue
            ppls.extend(get_perplexities(model, chunk, EOS_TOKEN_ID, device))
    log_ppls = [math.log(p) for p in ppls]
    return {
        "n": len(ppls),
        "gmean_ppl": round(float(math.exp(sum(log_ppls) / len(log_ppls))), 4),
        "mean_ppl": round(float(sum(ppls) / len(ppls)), 4),
        "ppls": [round(p, 4) for p in ppls],
    }


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
    config = GPT2Config(
        vocab_size=50257 + vocab_extra,
        n_positions=SEQ_LEN, n_embd=768, n_layer=12, n_head=12,
        resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1,
        reorder_and_upcast_attn=True, scale_attn_by_inverse_layer_idx=True,
    )
    model = GPT2LMHeadModel(config).to(device)

    accum = EFF_BATCH // MICRO_BATCH
    assert accum * MICRO_BATCH == EFF_BATCH, (
        f"REPRO_MICRO_BATCH={MICRO_BATCH} does not divide the effective batch "
        f"{EFF_BATCH}; the gradient accumulation must reproduce the Kallini "
        f"effective batch exactly")
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    scaler = torch.cuda.amp.GradScaler()

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

    eval_trace: dict[str, float] = {}
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
            with torch.cuda.amp.autocast():
                out = model(input_ids=input_ids, labels=input_ids.clone())
            scaler.scale(out.loss / accum).backward()
            loss_avg += float(out.loss) / accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        if step in checkpoints:
            trace = evaluate_checkpoint(model, eval_sents, device,
                                        batch=int(os.environ.get("REPRO_EVAL_BATCH", 8)))
            eval_trace[str(step)] = trace["gmean_ppl"]
            torch.save(trace["ppls"], out_dir / f"ppls_step{step}.pt")
            print(f"[eval] {perturbation} seed{seed} step {step}: "
                  f"gmean_ppl={trace['gmean_ppl']} (n={trace['n']})", flush=True)
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
        "n_blocks": len(blocks),
        "eval_gmean": eval_trace,
        "wall_time_s": round(time.time() - t0, 1),
        "completed_at": datetime.now(timezone.utc).isoformat(),
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
        (f"steps{args.steps}_seed{args.seed}" if args.steps else f"seed{args.seed}")
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
