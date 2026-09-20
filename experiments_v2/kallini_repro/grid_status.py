#!/usr/bin/env python3
"""grid_status.py -- pending-work manifest for the v3 GPU box (single source of truth).

WHY
    The chain sentinel rebuilds the ``kallini-gpu`` unit whenever that unit is
    inactive. Once the registered grid is finished, that rule restarts a
    completed chain every 15 minutes forever (and, worse, it makes "the chain
    stopped" indistinguishable from "the chain finished"). This script answers
    the only question the sentinel needs: how many *registered* cells are still
    missing, per arm.

    Keep this list in sync when an arm is registered (design audit amendments),
    otherwise the sentinel will either restart a finished chain or leave a new
    arm unstarted.

USAGE
    python3 experiments_v2/kallini_repro/grid_status.py            # human report
    python3 experiments_v2/kallini_repro/grid_status.py --one-line # pending=N ... for the sentinel
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
R = REPO / "experiments_v2" / "kallini_repro" / "results"
L = REPO / "experiments_v2" / "kallini_repro" / "results_lstm_gpu"
CPU_LSTM = Path(os.environ.get("LSTM_RESULTS", "/root/llm-impossible-lstm/experiments_v2/kallini_repro/results_lstm"))

SR9 = ["shuffle_control", "shuffle_nondeterministic", "shuffle_deterministic21",
       "shuffle_local3", "shuffle_local10", "shuffle_even_odd",
       "reverse_control", "reverse_partial", "reverse_full"]
P4 = ["parity_word", "parity_tok", "negtok", "fixed_start"]
SEEDS3 = [0, 14, 41]
EXT_SEEDS = [53, 96]
# LSTM arms: conditions x seeds, per arm.
LSTM_GPU_CONDS = ["shuffle_control", "reverse_full", "parity_word", "not_random"]
LSTM_CPU_CONDS = ["shuffle_control", "reverse_full", "reverse_control",
                  "parity_word", "parity_tok", "negtok", "fixed_start"]
LSTM_CPU_SEEDS = [0, 14, 41, 53, 96]


def gpt2_cells() -> list[Path]:
    want: list[Path] = []
    for c in SR9:                                     # T0 replication panel
        for s in SEEDS3:
            want.append(R / f"babylm_{c}_100M" / f"seed{s}" / "exp1_result.json")
    for c in P4:                                      # class P grid
        for s in SEEDS3:
            want.append(R / f"babylm_{c}_100M" / f"seed{s}" / "exp1_result.json")
    for c in ["parity_word", "fixed_start", "shuffle_control"]:   # H7 2x at seed 0
        want.append(R / f"babylm_{c}_100M" / "steps6000_seed0" / "exp1_result.json")
    for c in ["shuffle_control", "reverse_full", "parity_word", "fixed_start",
              "parity_tok", "negtok"]:                # extension tier, seeds 53/96
        for s in EXT_SEEDS:
            want.append(R / f"babylm_{c}_100M" / f"seed{s}" / "exp1_result.json")
    for c in ["fixed_end", "not_random"]:             # F2 control + entropy-matched control
        for s in SEEDS3:
            want.append(R / f"babylm_{c}_100M" / f"seed{s}" / "exp1_result.json")
    for c in ["shuffle_control", "parity_word"]:      # H7 2x at seeds 14/41
        for s in [14, 41]:
            want.append(R / f"babylm_{c}_100M" / f"steps6000_seed{s}" / "exp1_result.json")
    return want


def lstm_gpu_cells() -> list[Path]:
    return [L / f"babylm_{c}_100M" / f"seed{s}" / "lstm_result.json"
            for c in LSTM_GPU_CONDS for s in SEEDS3]


def lstm_cpu_cells() -> list[Path]:
    return [CPU_LSTM / f"babylm_{c}_100M" / f"seed{s}" / "lstm_result.json"
            for c in LSTM_CPU_CONDS for s in LSTM_CPU_SEEDS]


def summarize() -> dict:
    arms = {"gpt2": gpt2_cells(), "lstm_gpu": lstm_gpu_cells(), "lstm_cpu": lstm_cpu_cells()}
    out = {}
    for name, want in arms.items():
        done = sum(1 for p in want if p.exists())
        out[name] = {"expected": len(want), "done": done, "pending": len(want) - done}
    out["pending_total"] = sum(v["pending"] for k, v in out.items() if isinstance(v, dict))
    out["complete_marker"] = (R / "ALL_KALLINI_DONE").exists()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--one-line", action="store_true",
                    help="emit 'pending=N pending_gpt2=N ...' for the sentinel")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    s = summarize()
    if args.json:
        print(json.dumps(s, indent=2))
        return 0
    if args.one_line:
        print(" ".join(f"pending_{k}={v['pending']}" if isinstance(v, dict)
                       else f"{k}={v}" for k, v in s.items()))
        return 0
    for name in ("gpt2", "lstm_gpu", "lstm_cpu"):
        v = s[name]
        print(f"{name:9s} done {v['done']:3d}/{v['expected']:3d}  pending {v['pending']:3d}")
    print(f"pending_total={s['pending_total']}  ALL_KALLINI_DONE={s['complete_marker']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
