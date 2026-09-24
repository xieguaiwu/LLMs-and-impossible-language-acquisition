#!/usr/bin/env python3
"""LSTM arm of the v3 grid (DESIGN_V3 §2.2) — CPU box (cpu2).

Purpose
-------
Train the capacity-matched ``lstm_matched`` (~39M) on the *same* BabyLM
perturbed datasets and the *same* eval protocol as the GPT-2 arm
(``train_exp1.py``), so the architecture axis is byte-identical except for the
model itself.

Byte-identical guarantees (do not weaken)
-----------------------------------------
* Packing/eval primitives are **imported** from ``train_exp1`` — never copied —
  so the GPT-2 arm stays the single source of truth:
  ``load_packed_dataset`` (shuffle with rng(seed), EOS join, chunk),
  ``load_eval_sentences`` (same test sample), ``create_input_ids`` /
  ``create_attention_mask`` (Kallini's exact tensors), ``eval_checkpoints_for``
  (same ladder), ``set_seed``.
* Re-chunking to the LSTM's window: the token stream is rebuilt by
  concatenating the GPT-2 blocks and slicing into ``LSTM_SEQ_LEN`` windows.
  Blocks are consecutive slices of one stream, so the concatenation is the same
  stream — asserted by token-count equality below.
* Loss/perplexity math is Kallini's ``perplexities.py`` formula, mask-weighted,
  identical to ``train_exp1.get_perplexities`` (only the model call differs,
  because ``LSTMLM`` takes no ``attention_mask``).

Documented deviations (CPU reality; logged in every result JSON)
---------------------------------------------------------------
* ``LSTM_SEQ_LEN`` default 256 and ``LSTM_STEPS`` default 3000: the design's
  batch 32 × seq 512 × 50257-vocab logits alone are 3.3 GB forward + 3.3 GB
  backward, which OOM-kills a 7.7 GB CPU box (measured 2026-09-19, exit 137).
  Effective batch stays 32 (micro 16 × accum 2) as designed.
* No AMP / no GradScaler (CPU), AdamW + linear warmup/decay + clip 5.0 =
  the frozen v2 LSTM regime from ``training/train_lm.py``.
* Weights are not saved by default (``LSTM_SAVE_CKPT=1`` to keep them); the
  results branch excludes weight files anyway.

Usage
-----
    LSTM_SEQ_LEN=256 LSTM_STEPS=300 python train_exp1_lstm.py shuffle_control \
        --seed 0 --skip-if-done

Budget limitation (binding, owner ruling 2026-09-19)
---------------------------------------------------
This arm runs at ~1/160 of the GPT-2 arm's token budget (300 x 8192 = 2.46e6 vs
3000 x 128 x 1024 = 3.93e8). It must therefore never be reported as an
"architecture axis at equal budget": only the budget-dependent wording of the
F8/H7 family is licensed ("no detectable difference *at this budget*").
Epochs-matched on cpu2 would take ~160 h per cell (GPT-2 arm ~27k tok/s vs
lstm_matched ~680 tok/s measured here).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

# Stack portability (§10c-11): same shim as the GPT-2 trainer.
from stack_compat import amp_autocast, grad_scaler, pin_numerics, stack_metadata

# Thread discipline (2026-09-20): torch defaults to one intra-op thread per core,
# which oversubscribes against OMP_NUM_THREADS and made every step ~8x slower
# (measured on cpu2: 165 s/step vs 19.8 s/step with an explicit 2). The env var is
# the single knob; keep torch in lockstep with it.
_THREADS = int(os.environ.get("LSTM_THREADS", 2))
torch.set_num_threads(_THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))            # sibling module: train_exp1
sys.path.insert(0, str(HERE.parent))     # experiments_v2: training.models

import train_exp1 as G  # noqa: E402  (GPT-2 arm = protocol single source)
from training.models import LSTMLM, count_parameters  # noqa: E402

# ---------------------------------------------------------------- config ----

SEQ_LEN = int(os.environ.get("LSTM_SEQ_LEN", 256))
MICRO_BATCH = int(os.environ.get("LSTM_MICRO_BATCH", 4))
EFF_BATCH = int(os.environ.get("LSTM_EFF_BATCH", 32))
STEPS = int(os.environ.get("LSTM_STEPS", 3000))
# Device: "cpu" (the budget-limited cpu2 arm, default) or "cuda" (the
# equal-token-budget GPU arm registered 2026-09-20, audit B2). The default keeps
# the running cpu2 arm bit-for-bit on its old path.
DEVICE = os.environ.get("LSTM_DEVICE", "cpu")
# Optional fp16 autocast; off by default so the two LSTM arms share numerics.
AMP = os.environ.get("LSTM_AMP", "0") == "1"
WARMUP = int(os.environ.get("LSTM_WARMUP", 0)) or max(100, int(0.10 * STEPS))
PEAK_LR = float(os.environ.get("LSTM_LR", 1e-3))
WEIGHT_DECAY = float(os.environ.get("LSTM_WD", 1e-5))
CLIP = float(os.environ.get("LSTM_CLIP", 5.0))
EMB_DIM = int(os.environ.get("LSTM_EMB", 640))
HIDDEN_DIM = int(os.environ.get("LSTM_HIDDEN", 640))
N_LAYERS = int(os.environ.get("LSTM_LAYERS", 2))
DROPOUT = float(os.environ.get("LSTM_DROPOUT", 0.3))
SAVE_CKPT = os.environ.get("LSTM_SAVE_CKPT", "0") == "1"
RESULTS = Path(os.environ.get("LSTM_RESULTS", HERE / "results_lstm"))
EVAL_BATCH = int(os.environ.get("LSTM_EVAL_BATCH", 8))
# Eval cost on CPU is dominated by the 50257-vocab head (~3 s per 16x256 batch),
# so the LSTM arm evaluates the FIRST LSTM_EVAL_N of the *same* 10k test sample
# the GPT-2 arm draws (subset, not a different sample) — documented deviation.
EVAL_N = int(os.environ.get("LSTM_EVAL_N", 2000))
VOCAB_SIZE = 50257  # GPT-2 marker-free base; reverse_* conditions add a marker id
EOS = G.EOS_TOKEN_ID


BUDGET_NOTE = (
    "budget-limited arm: LSTM_STEPS x effective_batch x SEQ_LEN tokens vs the GPT-2 "
    "arm's 3000 x 128 x 1024 = 3.93e8 (ratio ~1/160 at the 300-step default). "
    "NO equal-budget architecture-axis claim is licensed by this arm; report it only "
    "as budget-dependent (F8/H7 family: 'no detectable difference at this budget'). "
    "Epochs-matched would need ~160 h per cell on cpu2 (measured: GPT-2 arm ~27k tok/s "
    "vs lstm_matched ~680 tok/s here)."
)
# Capacity-matched arm (§10c-2) overrides the note + arch tag via env so the
# result JSON records what the arm actually is (defaults keep the cpu2 arm's
# frozen wording byte-identical).
BUDGET_NOTE = os.environ.get("LSTM_BUDGET_NOTE", BUDGET_NOTE)
ARCH_TAG = os.environ.get("LSTM_ARCH_TAG", "lstm_matched")


def logits_gb(batch: int, seq: int) -> float:
    """Forward+backward footprint of the vocab projection (fp32)."""
    return 2 * batch * seq * VOCAB_SIZE * 4 / 1024 ** 3


# ------------------------------------------------------------------ data ----

# Cache tag: bump when the packing semantics change so stale .npy caches are
# never reused (2026-09-19 v2 = drop the trailing partial window, mirroring
# upstream babylm_dataset.py::__chunk).
PACK_VERSION = os.environ.get("LSTM_PACK_VERSION", "v2")


def _sentence_stream(perturbation: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-file token arrays with one EOS appended per sentence, plus lengths.

    Vectorised twin of ``train_exp1.load_packed_dataset``'s parsing stage: the
    token ids and their order inside a file are identical; only the container
    differs (numpy int32 instead of a Python list of ints). The GPT-2 arm's
    parser peaks at 6.65 GB RSS on this corpus (measured 2026-09-19) because
    every token id becomes a Python object — that is what swap-thrashed cpu2.
    """
    data_dir = G.BABYLM_DATA_PATH / "babylm_data_perturbed" / f"babylm_{perturbation}" / f"babylm_{G.TRAIN_SET}"
    files = sorted(data_dir.glob("*.train"))
    assert files, f"no perturbed train files for {perturbation} under {data_dir}"
    parts: list[np.ndarray] = []
    lengths: list[np.ndarray] = []
    for f in files:
        text = f.read_text()
        lines = text.splitlines()
        tok = np.array(text.split(), dtype=np.int32) if text.strip() else np.zeros(0, np.int32)
        lens = np.fromiter((len(l.split()) for l in lines), dtype=np.int64, count=len(lines))
        arr = np.empty(tok.size + lens.size, dtype=np.int32)
        if lens.size:
            eos_pos = np.cumsum(lens) + np.arange(lens.size)
            mask = np.ones(arr.size, dtype=bool)
            mask[eos_pos] = False
            arr[~mask] = G.EOS_TOKEN_ID
            arr[mask] = tok
        parts.append(arr)
        lengths.append(lens + 1)          # +1: the EOS that follows each sentence
    return np.concatenate(parts), np.concatenate(lengths)

def _data_fingerprint(perturbation: str) -> str:
    """Short hash of the condition's train files (names, sizes, mtimes)."""
    import hashlib
    data_dir = G.BABYLM_DATA_PATH / "babylm_data_perturbed" / f"babylm_{perturbation}" / f"babylm_{G.TRAIN_SET}"
    h = hashlib.blake2b(digest_size=4)
    for f in sorted(data_dir.glob("*.train")):
        st = f.stat()
        h.update(f"{f.name}:{st.st_size}:{st.st_mtime_ns}\n".encode())
    return h.hexdigest()

def packed_blocks(perturbation: str, seed: int) -> tuple[np.ndarray, int, int]:
    """Sentence-shuffled stream (same permutation as the GPT-2 arm) -> windows.

    Returns ``(windows, n_tokens, n_sentences)``. The shuffle is
    ``rng.shuffle`` over an index array of the same length as the sentence
    list, which is the same permutation the GPT-2 arm applies to its sentence
    list (numpy's shuffle depends only on length + RNG state).
    """
    stream, lens = _sentence_stream(perturbation)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    rng = np.random.default_rng(seed)
    perm = np.arange(lens.size)
    rng.shuffle(perm)
    sl = lens[perm]
    st = starts[perm]
    total = int(sl.sum())
    # ragged gather: destination offsets, then source indices
    offs = np.repeat(np.cumsum(sl) - sl, sl)
    src = np.repeat(st, sl)
    src += np.arange(total, dtype=np.int32) - offs
    shuffled = stream[src]
    del stream, src, st, sl, offs, perm
    # Drop the trailing partial window: upstream babylm_dataset.py::__chunk ends
    # with "# Drop last line if not a multiple of max_seq_len" + pop(), and the
    # GPU arm died randomly at steps 500-1000 (ValueError: expected sequence of
    # length 1024 at dim 1) whenever a partial block reached a batch.
    n_full = total // SEQ_LEN
    kept = n_full * SEQ_LEN
    windows = np.ascontiguousarray(shuffled[:kept].reshape(n_full, SEQ_LEN))
    del shuffled
    # ---- packing invariants (asserted, not smoke-tested) ----
    assert windows.shape[1] == SEQ_LEN, f"window width {windows.shape[1]} != {SEQ_LEN}"
    assert windows.shape[0] == n_full == (total - (total % SEQ_LEN)) // SEQ_LEN, (
        f"window count {windows.shape[0]} != {n_full} (token total {total})")
    assert windows.size == kept, f"kept {windows.size} != {kept} tokens"
    assert len(windows) == 0 or windows[-1].shape[0] == SEQ_LEN, "trailing partial window survived"
    if n_full == 0:
        raise RuntimeError(f"no full window for {perturbation} seed{seed}: total={total}")
    # Persist + mmap: the packed stream is 545 MB per (condition, seed). Two
    # workers holding it anonymously pushed cpu2 into swap thrash (measured
    # 2026-09-19: 3.7 GB swap full, both workers in D state at 10% CPU).
    # File-backed pages are shared and evictable, so training RSS stays low.
    cache_dir = RESULTS / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Data fingerprint in the cache name (2026-09-20): the pack version alone
    # cannot see a *dataset* change, so a regenerated condition under the same
    # pack tag would silently reuse the old packed stream (this is how the
    # pool-v1/pool-v2 sentence-set change would have slipped through on cpu2).
    cache_file = cache_dir / (f"{perturbation}_seed{seed}_seq{SEQ_LEN}_"
                              f"{PACK_VERSION}_{_data_fingerprint(perturbation)}.npy")
    if not cache_file.exists():
        np.save(cache_file, windows)
    del windows
    windows = np.load(cache_file, mmap_mode="r")
    return windows, kept, int(lens.size)


# ------------------------------------------------------------------ model ----

def model_logits(model: LSTMLM, input_ids: torch.Tensor) -> torch.Tensor:
    x = model.dropout(model.embed(input_ids))
    out, _ = model.lstm(x)
    if model.pre_head is not None:
        out = model.pre_head(out)
    return model.head(out)


def get_perplexities_lstm(model, token_lists, pad_token_id, device="cpu", marker_ids=None):
    """Kallini's perplexities.py math, LSTMLM call path (no attention_mask arg).

    Mirrors ``train_exp1.get_perplexities`` including the optional
    content-token-only (marker-masked) second metric, so the architecture axis
    can be reported with and without the marker positions.
    """
    input_ids = G.create_input_ids(token_lists, pad_token_id).to(device)
    labels = input_ids.clone()
    attention_mask = G.create_attention_mask(token_lists).to(device)
    logits = model_logits(model, input_ids)
    shift_logits = logits[..., :-1, :].contiguous()
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


def evaluate_checkpoint(model, eval_sents, device="cpu", batch: int = EVAL_BATCH,
                        marker_ids=None) -> dict:
    """Mirror of train_exp1.evaluate_checkpoint (same truncation, same gmean)."""
    ppls: list[float] = []
    ppls_content: list[float] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(eval_sents), batch):
            chunk = [s[:SEQ_LEN] for s in eval_sents[i : i + batch] if len(s) >= 2]
            if not chunk:
                continue
            out = get_perplexities_lstm(model, chunk, EOS, device, marker_ids)
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
        res["n_marker_masked"] = len(ppls_content)
    return res


# --------------------------------------------------------------- training ----

def lr_at(step: int, warmup: int, total: int) -> float:
    """v2 LSTM regime: linear warmup, then linear decay to 0."""
    if step < warmup:
        return step / max(1, warmup)
    return max(0.0, (total - step) / max(1, total - warmup))


def train_one(perturbation: str, seed: int, out_dir: Path, steps: int, warmup: int) -> dict:
    accum = EFF_BATCH // MICRO_BATCH
    assert accum * MICRO_BATCH == EFF_BATCH, (
        f"LSTM_MICRO_BATCH={MICRO_BATCH} does not divide LSTM_EFF_BATCH={EFF_BATCH}")

    t_pack = time.time()
    windows, n_tokens, n_sents = packed_blocks(perturbation, seed)
    eval_sents = G.load_eval_sentences(perturbation, seed)
    if 0 < EVAL_N < len(eval_sents):
        eval_sents = eval_sents[:EVAL_N]          # subset of the GPT-2 arm's sample
    print(f"[data] windows={len(windows)} tokens={n_tokens} sents={n_sents} "
          f"eval_sents={len(eval_sents)} ({time.time()-t_pack:.0f}s)", flush=True)

    footprint = logits_gb(MICRO_BATCH, SEQ_LEN)
    print(f"[mem] micro={MICRO_BATCH} seq={SEQ_LEN} logits≈{footprint:.2f} GB "
          f"(fwd+bwd)", flush=True)

    G.set_seed(seed)
    vocab_size = VOCAB_SIZE + G.VOCAB_EXTRA.get(perturbation, 0)   # reverse_*: +1 marker
    model = LSTMLM(
        vocab_size=vocab_size, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM,
        num_layers=N_LAYERS, dropout=DROPOUT, pad_token_id=EOS,
    ).to(DEVICE)
    n_params = count_parameters(model)
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY)
    pin_numerics()
    scaler = grad_scaler(enabled=AMP) if DEVICE.startswith("cuda") else None

    rng = np.random.default_rng(seed + 1)
    order = rng.permutation(len(windows))
    ptr = 0

    def next_batch() -> torch.Tensor:
        """One MICRO batch (mirrors the GPT-2 arm's fixed semantics)."""
        nonlocal ptr, order
        rows = np.empty((MICRO_BATCH, SEQ_LEN), dtype=np.int32)
        for i in range(MICRO_BATCH):
            if ptr >= len(order):
                order = rng.permutation(len(windows))
                ptr = 0
            rows[i] = windows[order[ptr]]
            ptr += 1
        return torch.from_numpy(rows.astype(np.int64)).to(DEVICE)
    checkpoints = G.eval_checkpoints_for(steps)
    eval_trace: dict[str, float] = {}
    eval_content_trace: dict[str, float] = {}
    losses: list[float] = []
    t0 = time.time()
    model.train()
    for step in range(1, steps + 1):
        for p in opt.param_groups:
            p["lr"] = PEAK_LR * lr_at(step, warmup, steps)
        opt.zero_grad(set_to_none=True)
        loss_avg = 0.0
        for _ in range(accum):
            input_ids = next_batch()
            if AMP:
                with amp_autocast():
                    out = model(input_ids, labels=input_ids.clone())
                    loss = out["loss"]
                scaler.scale(loss / accum).backward()
            else:
                out = model(input_ids, labels=input_ids.clone())
                loss = out["loss"]
                (loss / accum).backward()
            loss_avg += float(loss) / accum
        if AMP:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        if AMP:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        losses.append(loss_avg)

        if step in checkpoints:
            trace = evaluate_checkpoint(model, eval_sents, device=DEVICE,
                                        marker_ids=getattr(G, "MARKER_IDS", None))
            model.train()                            # 2026-09-24 fix (mirror of the GPT-2 F9 fix):
            # evaluate_checkpoint() left the model in eval() mode and nothing
            # restored it -> the next backward() aborts on the cudnn-LSTM path
            # ("cudnn RNN backward can only be called in training mode"); on CPU
            # it silently trained without dropout from the first eval on.
            eval_trace[str(step)] = trace["gmean_ppl"]
            if "gmean_ppl_content" in trace:
                eval_content_trace[str(step)] = trace["gmean_ppl_content"]
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / f"eval_step{step}.json", "w") as f:
                json.dump(trace, f)
            print(f"[eval] {perturbation} seed{seed} step {step}: "
                  f"gmean_ppl={trace['gmean_ppl']} content={trace.get('gmean_ppl_content')} "
                  f"(n={trace['n']})", flush=True)
        if step % 50 == 0:
            print(f"[train] {perturbation} seed{seed} step {step}/{steps} "
                  f"loss={loss_avg:.4f} elapsed={(time.time()-t0)/60:.1f}m", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_CKPT:
        torch.save(model.state_dict(), out_dir / "final.pt")
    return {
        "language": perturbation,
        "train_set": G.TRAIN_SET,
        "seed": seed,
        "arch": ARCH_TAG,
        "emb_dim": EMB_DIM,
        "hidden_dim": HIDDEN_DIM,
        "n_params": n_params,
        "device": DEVICE,
        "amp": AMP,
        "vocab_size": vocab_size,
        "max_steps": steps,
        "effective_batch": EFF_BATCH,
        "micro_batch": MICRO_BATCH,
        "seq_len": SEQ_LEN,
        "peak_lr": PEAK_LR,
        "weight_decay": WEIGHT_DECAY,
        "clip": CLIP,
        "warmup": warmup,
        "n_windows": len(windows),
        "n_tokens": n_tokens,
        "n_sentences": n_sents,
        "tokens_per_step": EFF_BATCH * SEQ_LEN,
        "token_budget": EFF_BATCH * SEQ_LEN * steps,
        "budget_note": BUDGET_NOTE,
        "eval_gmean": eval_trace,
        "eval_gmean_content": eval_content_trace,
        "eval_n": len(eval_sents),
        "eval_pool_n": getattr(eval_sents, "pool_n", None),
        "eval_pool_exact_dups": getattr(eval_sents, "pool_exact_dups", None),
        "eval_fingerprint": getattr(eval_sents, "fingerprint", None),
        "eval_near_dup_rate_sampled": getattr(eval_sents, "near_dup_rate_sampled", None),
        "final_loss": round(float(np.mean(losses[-50:])), 4) if losses else None,
        "wall_time_s": round(time.time() - t0, 1),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "stack": stack_metadata(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("perturbation", choices=G.LANGUAGES)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--skip-if-done", action="store_true")
    ap.add_argument("--prepack-only", action="store_true",
                    help="build the packed .npy cache and exit (serial prepack phase)")
    args = ap.parse_args()

    steps = args.steps or STEPS
    warmup = max(100, int(0.10 * steps))
    out_dir = RESULTS / f"babylm_{args.perturbation}_{G.TRAIN_SET}" / f"seed{args.seed}"
    done_marker = out_dir / "lstm_result.json"
    if args.skip_if_done and done_marker.exists():
        print(f"SKIP {done_marker} (already complete)")
        return

    if args.prepack_only:
        windows, total, n_sents = packed_blocks(args.perturbation, args.seed)
        print(f"PREPACK {args.perturbation} seed{args.seed}: windows={len(windows)} "
              f"tokens={total} sents={n_sents}")
        return

    result = train_one(args.perturbation, args.seed, out_dir, steps, warmup)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(done_marker, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["eval_gmean"], indent=2))


if __name__ == "__main__":
    main()
