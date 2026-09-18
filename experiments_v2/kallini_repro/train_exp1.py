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
from itertools import zip_longest
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------- config ----

KALLINI_REPO = Path(os.environ.get("KALLINI_REPO", "/root/mission-impossible-language-models"))
BABYLM_DATA_PATH = Path(os.environ.get("KALLINI_DATA_PATH", "/root/kallini_data"))
RESULTS = Path(os.environ.get("REPRO_RESULTS", Path(__file__).resolve().parent / "results"))
TRAIN_SET = os.environ.get("REPRO_TRAIN_SET", "100M")
MAX_STEPS = 3000
WARMUP_STEPS = 300
PEAK_LR = 6e-4
EVAL_CHECKPOINTS = [100, 300, 500, 1000, 2000, 3000]
EVAL_SAMPLE = 10000
EFF_BATCH = 128
MICRO_BATCH = 8
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
    padded = zip_longest(*token_lists, fillvalue=pad_token_id)
    return torch.tensor(list(padded), dtype=torch.long)


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


def evaluate_checkpoint(model, eval_sents, device="cuda", batch=32) -> dict:
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one(perturbation: str, seed: int, out_dir: Path, device: str = "cuda") -> dict:
    from transformers import GPT2Config, GPT2LMHeadModel

    spec = PERTURBATIONS[perturbation]
    tokenizer = spec["gpt2_tokenizer"]
    reverse_mode = perturbation.startswith("reverse")

    t_pack = time.time()
    blocks = load_packed_dataset(perturbation, seed)
    eval_sents = load_eval_sentences(perturbation, seed)
    print(f"[data] blocks={len(blocks)} eval_sents={len(eval_sents)} "
          f"({time.time()-t_pack:.0f}s)", flush=True)

    set_seed(seed)
    config = GPT2Config(
        vocab_size=50257 + (1 if reverse_mode else 0),
        n_positions=SEQ_LEN, n_embd=768, n_layer=12, n_head=12,
        resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1,
        reorder_and_upcast_attn=True, scale_attn_by_inverse_layer_idx=True,
    )
    model = GPT2LMHeadModel(config).to(device)

    accum = EFF_BATCH // MICRO_BATCH
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    scaler = torch.cuda.amp.GradScaler()

    rng = np.random.default_rng(seed + 1)
    n_blocks = len(blocks)
    order = rng.permutation(n_blocks)
    ptr = 0

    def next_batch() -> list[list[int]]:
        nonlocal ptr, order
        batch: list[list[int]] = []
        while len(batch) < EFF_BATCH:
            if ptr >= n_blocks:
                order = rng.permutation(n_blocks)
                ptr = 0
            batch.append(blocks[order[ptr]])
            ptr += 1
        return batch

    eval_trace: dict[str, float] = {}
    t0 = time.time()
    model.train()
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

        if step in EVAL_CHECKPOINTS:
            trace = evaluate_checkpoint(model, eval_sents, device)
            eval_trace[str(step)] = trace["gmean_ppl"]
            torch.save(trace["ppls"], out_dir / f"ppls_step{step}.pt")
            out_dir.mkdir(parents=True, exist_ok=True)
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
    args = parser.parse_args()

    out_dir = RESULTS / f"babylm_{args.perturbation}_{TRAIN_SET}" / f"seed{args.seed}"
    done_marker = out_dir / "exp1_result.json"
    if args.skip_if_done and done_marker.exists():
        print(f"SKIP {done_marker} (already complete)")
        return

    result = train_one(args.perturbation, args.seed, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(done_marker, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["eval_gmean"], indent=2))


if __name__ == "__main__":
    main()
