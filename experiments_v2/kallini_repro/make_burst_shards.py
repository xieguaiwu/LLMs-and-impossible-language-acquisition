#!/usr/bin/env python3
"""make_burst_shards.py — turn the remaining registered grid into per-GPU shard files.

WHY
---
A second host can only help if every GPU gets an explicit, fixed cell list: the
monolithic ``kallini_queue.sh`` walks *all* arms in a fixed order against a
per-host result tree, so two hosts would duplicate cells.  This script emits one
TSV per GPU from the **same manifests the sentinel uses** (``grid_status.py``), so
shards and the pending count cannot drift apart.

The launch recipes (env + args) below are copied verbatim from
``kallini_queue.sh`` sections [4b]/[4c]/[4c2]/[4c3]/[4d]/[4d2]; the TSV carries them
so the runner stays generic (one place to audit fidelity).

Split policy: whole arms per shard (no arm crosses stacks), paper-critical P
family first, stretch tier last.

USAGE
    python3 make_burst_shards.py --out-dir /root/burst/shards --gpus 4
    python3 make_burst_shards.py --out-dir /root/burst/shards --bridge
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _repo() -> Path:
    """Repo root: BURST_REPO > script location (if it is inside a repo) > cwd."""
    env = os.environ.get("BURST_REPO")
    if env:
        return Path(env)
    if len(HERE.parents) > 1 and (HERE.parents[1] / "experiments_v2").is_dir():
        return HERE.parents[1]
    return Path.cwd()


REPO = _repo()

LADDER_P_BLOCK = {"parity_word", "fixed_start", "parity_tok", "negtok", "not_random"}

COLS = ["kind", "condition", "seed", "steps", "env", "result_path", "note"]


def _grid():
    for cand in (HERE / "grid_status.py",
                 REPO / "experiments_v2" / "kallini_repro" / "grid_status.py"):
        if cand.is_file():
            spec = importlib.util.spec_from_file_location("grid_status", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit("grid_status.py not found — run this from the repo, or set BURST_REPO")


def _cond_of(cell: Path) -> str:
    return cell.parent.parent.name.replace("babylm_", "").replace("_100M", "")


def _seed_of(cell: Path) -> tuple[str, str, str]:
    """(seed, steps, tag) from the cell's own directory name."""
    name = cell.parent.name
    m = re.match(r"steps(\d+)_seed(\d+)$", name)
    if m:
        return m.group(2), m.group(1), f"steps{m.group(1)}"
    m = re.match(r"seed(\d+)(?:_(.+))?$", name)
    if m:
        return m.group(1), "", (m.group(2) or "")
    return "", "", name


# --------------------------------------------------------------- recipes -------

def gpt2_row(cond: str, seed: str, steps: str, tag: str, result: Path,
             results_rel: str, ladder: bool) -> dict:
    env = [f"REPRO_RESULTS={results_rel}"]
    if ladder:
        env.append("LADDER_PROBE=1")
    return {"kind": "gpt2", "condition": cond, "seed": seed, "steps": steps,
            "env": " ".join(env), "result_path": str(result), "note": tag}


def lstm_rows(kind: str, cond: str, seed: str, result: Path, capmatch_lr: str) -> dict:
    base = ["LSTM_DEVICE=cuda", "LSTM_SEQ_LEN=1024", "LSTM_EFF_BATCH=128",
            "LSTM_MICRO_BATCH=8", "LSTM_STEPS=3000", "LSTM_EVAL_N=10000",
            "LSTM_SAVE_CKPT=0", "LSTM_PACK_VERSION=v2"]
    if kind == "lstm_gpu":
        env = base + ["LSTM_RESULTS=experiments_v2/kallini_repro/results_lstm_gpu",
                      "LSTM_LR=1e-3"]
    else:                                            # capmatch
        env = base + ["LSTM_RESULTS=experiments_v2/kallini_repro/results_lstm_gpu_capmatch",
                      f"LSTM_LR={capmatch_lr}", "LSTM_EMB=1620", "LSTM_HIDDEN=1620",
                      "LSTM_ARCH_TAG=lstm_capmatch124"]
    return {"kind": kind, "condition": cond, "seed": seed, "steps": "3000",
            "env": " ".join(env), "result_path": str(result), "note": ""}


def collect(results_root: Path) -> dict[str, list[dict]]:
    gs = _grid()
    r = results_root
    out: dict[str, list[dict]] = {k: [] for k in
                                  ("gpt2", "lstm_gpu", "capmatch", "nope",
                                   "datascale", "logo", "model_scale", "ladder_probe")}

    for cell in gs.gpt2_cells():
        if cell.exists():
            continue
        cond, (seed, steps, tag) = _cond_of(cell), _seed_of(cell)
        rel = str(Path("experiments_v2/kallini_repro/results"))
        ladder = cond in LADDER_P_BLOCK or cond in ("fixed_end", "not_random") or seed in ("53", "96")
        out["gpt2"].append(gpt2_row(cond, seed, steps, tag, cell, rel, ladder))

    cap_lr = "1e-3"
    frozen = r / "results_lstm_gpu_capmatch" / ".frozen_lr"
    if frozen.exists():
        cap_lr = frozen.read_text().strip() or cap_lr

    for cell in gs.lstm_gpu_cells():
        if not cell.exists():
            out["lstm_gpu"].append(lstm_rows("lstm_gpu", _cond_of(cell), _seed_of(cell)[0], cell, cap_lr))
    for cell in gs.lstm_capmatch_cells():
        if not cell.exists():
            out["capmatch"].append(lstm_rows("capmatch", _cond_of(cell), _seed_of(cell)[0], cell, cap_lr))
    for cell in gs.nope_cells():
        if not cell.exists():
            out["nope"].append({"kind": "nope", "condition": _cond_of(cell), "seed": _seed_of(cell)[0],
                                "steps": "", "env": "REPRO_RESULTS=experiments_v2/kallini_repro/results_nope GPT2_NOPE=1 LADDER_PROBE=1",
                                "result_path": str(cell), "note": "F7 position ablation"})
    for cell in gs.datascale_cells():
        seed, steps, tag = _seed_of(cell)
        scale = tag.lstrip("_")
        cond = _cond_of(cell)
        env = (f"REPRO_RESULTS=experiments_v2/kallini_repro/results_datascale "
               f"REPRO_DATA_SUBDIR=babylm_{cond}_{scale} REPRO_DIR_TAG=_{scale}")
        out["datascale"].append({"kind": "datascale", "condition": cond, "seed": seed,
                                 "steps": "", "env": env, "result_path": str(cell),
                                 "note": f"F8 datascale {scale}"})
    for cell in gs.logo_cells():
        cond = _cond_of(cell)
        env = (f"REPRO_RESULTS=experiments_v2/kallini_repro/results_logo "
               f"REPRO_DATA_SUBDIR=babylm_{cond}_logo7sw REPRO_DIR_TAG=_logo7sw")
        out["logo"].append({"kind": "logo", "condition": cond, "seed": "0", "steps": "",
                            "env": env, "result_path": str(cell), "note": "LOGO generalization"})
    for cell in gs.model_scale_cells():
        env = ("REPRO_RESULTS=experiments_v2/kallini_repro/results_model_scale "
               "REPRO_MODEL_SIZE=gpt2_medium REPRO_MICRO_BATCH=2")
        out["model_scale"].append({"kind": "model_scale", "condition": _cond_of(cell),
                                   "seed": _seed_of(cell)[0], "steps": "", "env": env,
                                   "result_path": str(cell), "note": "F9 355M"})
    for cell in gs.ladder_probe_cells():
        out["ladder_probe"].append({"kind": "ladder_probe", "condition": _cond_of(cell), "seed": "0",
                                    "steps": "", "env": "REPRO_RESULTS=experiments_v2/kallini_repro/results_ladder_probe LADDER_PROBE=1",
                                    "result_path": str(cell), "note": "probe replay"})
    return out


# ------------------------------------------------------------------ policy -----

POLICY = {
    # shard 0: paper-critical class-P + H7 budget ladder
    # shard 1: architecture families (capmatch needs its own LR freeze)
    # shard 2: S/R replication panel (the rest of the gpt2 arm)
    # shard 3: stretch tier (exploratory) + probe replay
    "arms": [
        ("p_block", ["gpt2_p"]),
        ("arch", ["nope", "lstm_gpu", "capmatch"]),
        ("sr_panel", ["gpt2_rest"]),
        ("stretch", ["ladder_probe", "datascale", "logo", "model_scale"]),
    ],
}


def write_tsv(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as fh:
        fh.write("\t".join(COLS) + "\n")
        for row in rows:
            fh.write("\t".join(str(row.get(c, "")) for c in COLS) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default=str(REPO / "experiments_v2" / "kallini_repro" / "results"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--bridge", action="store_true",
                    help="emit bridge cells (existing 3080 cells re-run on the new stack)")
    ap.add_argument("--bridge-conditions", default="parity_word,fixed_start")
    ap.add_argument("--bridge-seeds", default="0")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.bridge:
        rows = []
        for cond in args.bridge_conditions.split(","):
            for seed in args.bridge_seeds.split(","):
                rows.append({"kind": "gpt2", "condition": cond, "seed": seed, "steps": "",
                             "env": "REPRO_RESULTS=experiments_v2/kallini_repro/results_bridge LADDER_PROBE=1",
                             "result_path": str(Path("experiments_v2/kallini_repro/results_bridge")
                                                / f"babylm_{cond}_100M" / f"seed{seed}" / "exp1_result.json"),
                             "note": "bridge (new stack, compare against the 3080 cell)"})
        path = out_dir / "shard_bridge.tsv"
        write_tsv(path, rows)
        print(f"bridge cells: {len(rows)} -> {path}")
        return 0

    all_rows = collect(Path(args.results_root))
    for kind, rows in all_rows.items():
        print(f"pending {kind:14s} {len(rows):3d}")

    p_block = set(LADDER_P_BLOCK)
    gpt2_all = all_rows.pop("gpt2", [])
    sources: dict[str, list[dict]] = {
        "gpt2_p": [r for r in gpt2_all if r["condition"] in p_block],
        "gpt2_rest": [r for r in gpt2_all if r["condition"] not in p_block],
    }
    sources.update(all_rows)                     # one source per arm name

    total = 0
    for name, names in POLICY["arms"][: max(1, args.gpus)]:
        rows: list[dict] = []
        for src in names:
            rows += sources.pop(src, [])
        path = out_dir / f"shard_{name}.tsv"
        write_tsv(path, rows)
        total += len(rows)
        print(f"shard_{name:10s} {len(rows):3d} cells -> {path}")
    leftover = [r for rows in sources.values() for r in rows]
    if leftover:
        path = out_dir / "shard_leftover.tsv"
        write_tsv(path, leftover)
        print(f"WARNING: {len(leftover)} cells matched no shard -> {path} (append manually)")
    print(f"total scheduled: {total} (+{len(leftover)} leftover)")
    return 0


if __name__ == "__main__":
    sys.exit(main())