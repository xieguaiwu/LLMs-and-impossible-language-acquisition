#!/usr/bin/env python3
"""Unified multi-seed training entry point (v2).

Replicates and extends the original single-seed experiments:

    python train_lm.py --model gpt2 --dataset svo --condition natural --seed 42
    python train_lm.py --model gpt2 --dataset babylm --condition reversed --seed 42
    python train_lm.py --model lstm_matched --dataset svo --condition parity_negation --seed 43
    python train_lm.py --model gpt2_tiny --dataset svo --condition fixed_start_neg --seed 42

Design constraints (see experiments_v2/preregistration.md):
- every run is one (model, dataset, condition, seed) cell; the seed governs
  weight init, data shuffling and dropout, everything else is frozen;
- fixed training-step budget per dataset (no early stopping rule ambiguity);
- evaluation is cross-entropy on a condition-perturbed held-out test set
  produced by data_v2/conditions.py (identical sentence pool for all
  conditions, so test PPL is comparable across conditions);
- output JSON is backward-compatible with the old statistics format and adds
  a ``summary`` block of per-run scalars (training/metrics.py).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- config ----

REPO = Path(__file__).resolve().parents[2]
DATA_V2 = REPO / "experiments_v2" / "data_v2"
RESULTS = REPO / "experiments_v2" / "results"

# Frozen budgets. SVO: 141 steps mirrors the original Experiment-1 protocol
# (effective batch 32). BabyLM: 401 steps mirrors the original Experiment-2.
BUDGETS = {"svo": {"steps": 141, "warmup": 20},
           "babylm": {"steps": 401, "warmup": 56}}

# Capacity-matched pair + original subjects.
LSTM_SPECS = {
    "lstm": dict(emb_dim=650, hidden_dim=650, num_layers=2, dropout=0.3, lr=1e-3),
    "lstm_matched": dict(emb_dim=640, hidden_dim=640, num_layers=2, dropout=0.3, lr=1e-3),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_texts(dataset: str, condition: str) -> tuple[str, str, str]:
    """Return (train_path, val_path, test_path) for a condition."""
    base = DATA_V2 / ("babylm_conditions" if dataset == "babylm" else "conditions")
    d = base / "train" / f"{condition}.txt"
    t = base / "test" / f"{condition}.txt"
    if not d.exists() or not t.exists():
        raise FileNotFoundError(
            f"missing condition files for {dataset}/{condition}; "
            f"run data_v2/generate_svo.py + data_v2/conditions.py first "
            f"(and babylm preparation for dataset=babylm)"
        )
    return str(d), str(d), str(t)  # val==train slice handled inside trainer


# ------------------------------------------------------------- gpt2 path ----

def train_gpt2(args, seed: int) -> dict:
    import torch
    from transformers import (DataCollatorForLanguageModeling, GPT2Config,
                              GPT2LMHeadModel, GPT2TokenizerFast,
                              TextDataset, Trainer, TrainingArguments)

    set_seed(seed)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_path, _val_path, test_path = load_texts(args.dataset, args.condition)
    budget = BUDGETS[args.dataset]

    if args.model == "gpt2":
        config = GPT2Config.from_pretrained("gpt2")
        model = GPT2LMHeadModel(config)  # from scratch
        model_name = "gpt2_small_124M"
        lr = 5e-5
        batch = 4
        accum = 8
    elif args.model == "gpt2_tiny":
        config = GPT2Config(
            vocab_size=50257, n_positions=128, n_embd=512, n_layer=6, n_head=8,
            resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1,
        )
        model = GPT2LMHeadModel(config)
        model_name = "gpt2_tiny_6L512d"
        lr = 5e-5
        batch = 4
        accum = 8
    else:
        raise ValueError(args.model)

    from training.models import count_parameters

    n_params = count_parameters(model)

    train_dataset = TextDataset(tokenizer=tokenizer, file_path=train_path,
                                block_size=128, overwrite_cache=True)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    run_dir = RESULTS / args.dataset / args.model / f"{args.condition}_seed{seed}"
    targs = TrainingArguments(
        output_dir=str(run_dir),
        overwrite_output_dir=True,
        max_steps=budget["steps"],
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum,
        learning_rate=lr,
        lr_scheduler_type="linear",
        warmup_steps=budget["warmup"],
        weight_decay=0.01,
        adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
        max_grad_norm=1.0,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=seed,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
    )
    trainer = Trainer(model=model, args=targs, data_collator=collator,
                      train_dataset=train_dataset)
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    losses = [float(x["loss"]) for x in trainer.state.log_history if "loss" in x]
    test_loss = eval_loss_gpt2(model, tokenizer, test_path)

    from training.metrics import write_run_json

    hp = dict(model=model_name, n_params=n_params, lr=lr, batch=batch, accum=accum,
              effective_batch=batch * accum, max_steps=budget["steps"],
              warmup=budget["warmup"], block_size=128, weight_decay=0.01,
              from_scratch=True, seed=seed)
    return write_run_json(
        run_dir / "training_metrics.json",
        run_id=f"{args.dataset}_{args.model}_{args.condition}_seed{seed}",
        experiment=f"v2_{args.dataset}",
        model=model_name, dataset=args.dataset, condition=args.condition,
        seed=seed, hyperparameters=hp, losses=losses, test_loss=test_loss,
        total_steps=budget["steps"], training_time_seconds=elapsed,
    )


def eval_loss_gpt2(model, tokenizer, test_path: str, block_size: int = 128,
                   batch_size: int = 8) -> float:
    import torch
    from transformers import DataCollatorForLanguageModeling, TextDataset

    dataset = TextDataset(tokenizer=tokenizer, file_path=test_path,
                          block_size=block_size, overwrite_cache=True)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    model.eval()
    total_loss, total_tokens = 0.0, 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = collator([dataset[j] for j in range(i, min(i + batch_size, len(dataset)))])
            input_ids = batch["input_ids"].to(device)
            out = model(input_ids=input_ids, labels=input_ids.clone())
            n = input_ids.numel()
            total_loss += float(out.loss) * n
            total_tokens += n
    model.train()
    return total_loss / max(total_tokens, 1)


# ------------------------------------------------------------- lstm path ----

def train_lstm(args, seed: int) -> dict:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    set_seed(seed)
    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    spec = LSTM_SPECS[args.model]
    budget = BUDGETS[args.dataset]

    train_path, _val_path, test_path = load_texts(args.dataset, args.condition)

    if args.condition == "parity_negation_negtok":
        tokenizer.add_special_tokens({"additional_special_tokens": ["<NEG>"]})

    class LineDataset(Dataset):
        def __init__(self, path: str, block: int = 64):
            with open(path, encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            self.examples = []
            for ln in lines:
                ids = tokenizer.encode(ln)[: block - 1]
                if len(ids) < 2:
                    continue
                ids = ids + [tokenizer.eos_token_id]
                self.examples.append(torch.tensor(ids, dtype=torch.long))

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, i):
            return self.examples[i]

    def collate(batch):
        maxlen = max(len(x) for x in batch)
        pad = tokenizer.eos_token_id
        input_ids = torch.full((len(batch), maxlen), pad, dtype=torch.long)
        for i, x in enumerate(batch):
            input_ids[i, : len(x)] = x
        return {"input_ids": input_ids}

    from training.models import LSTMLM, count_parameters

    model = LSTMLM(
        vocab_size=len(tokenizer), emb_dim=spec["emb_dim"],
        hidden_dim=spec["hidden_dim"], num_layers=spec["num_layers"],
        dropout=spec["dropout"], pad_token_id=tokenizer.eos_token_id,
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    n_params = count_parameters(model)

    ds = LineDataset(train_path)
    loader = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate,
                        drop_last=True, generator=torch.Generator().manual_seed(seed))

    opt = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=1e-5)
    total_steps = budget["steps"]
    warmup = budget["warmup"]

    def lr_at(step):
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    losses = []
    step = 0
    model.train()
    t0 = time.time()
    device = next(model.parameters()).device
    while step < total_steps:
        for batch in loader:
            if step >= total_steps:
                break
            input_ids = batch["input_ids"].to(device)
            labels = input_ids.clone()
            for p in opt.param_groups:
                p["lr"] = spec["lr"] * lr_at(step)
            out = model(input_ids, labels=labels)
            opt.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(out["loss"]))
            step += 1

    elapsed = time.time() - t0

    # held-out evaluation
    test_ds = LineDataset(test_path)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate)
    model.eval()
    total_loss, total_n = 0.0, 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            out = model(input_ids, labels=input_ids.clone())
            n = input_ids.numel()
            total_loss += float(out["loss"]) * n
            total_n += n
    test_loss = total_loss / max(total_n, 1)

    from training.metrics import write_run_json

    hp = dict(model=args.model, n_params=n_params, **{k: v for k, v in spec.items()},
              max_steps=total_steps, warmup=warmup, batch=32, from_scratch=True, seed=seed)
    return write_run_json(
        RESULTS / args.dataset / args.model / f"{args.condition}_seed{seed}" / "training_metrics.json",
        run_id=f"{args.dataset}_{args.model}_{args.condition}_seed{seed}",
        experiment=f"v2_{args.dataset}",
        model=args.model, dataset=args.dataset, condition=args.condition,
        seed=seed, hyperparameters=hp, losses=losses, test_loss=test_loss,
        total_steps=total_steps, training_time_seconds=elapsed,
    )


# ------------------------------------------------------------------ main ----

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        choices=["gpt2", "gpt2_tiny", "lstm", "lstm_matched"])
    parser.add_argument("--dataset", required=True, choices=["svo", "babylm"])
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--skip-if-done", action="store_true",
                        help="ralph-loop resumability: exit 0 immediately if this run's JSON already exists")
    args = parser.parse_args()

    os.chdir(REPO)  # relative imports + HF cache stability
    import sys

    sys.path.insert(0, str(REPO / "experiments_v2" / "training"))

    out_json = (RESULTS / args.dataset /
                ("lstm" if args.model.startswith("lstm") else args.model) /
                f"{args.condition}_seed{args.seed}" / "training_metrics.json")
    if args.skip_if_done and out_json.exists():
        try:
            prev = json.load(open(out_json))
            if prev.get("summary"):
                print(f"SKIP {out_json} (already complete)")
                return
        except Exception:
            pass  # corrupt file -> retrain

    if args.model.startswith("lstm"):
        record = train_lstm(args, args.seed)
    else:
        record = train_gpt2(args, args.seed)
    print(json.dumps(record["summary"], indent=2))


if __name__ == "__main__":
    main()
