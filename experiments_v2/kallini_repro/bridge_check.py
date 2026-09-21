#!/usr/bin/env python3
"""bridge_check.py — measure cross-stack drift between a bridge cell and the 3080 cell.

WHY
---
A second host necessarily runs a different CUDA/torch stack (RTX 5090 = sm_120,
torch >= 2.7 / cu128; the original grid ran torch 2.2.2 / cu121).  Kernel and
cuDNN differences can perturb training slightly, and the project has a precedent
for silent regime drift invalidating cells (FALSIFICATION #F9).  So the second
stack is never trusted on faith: **bridge cells** (the same condition + seed +
budget, re-run there) are compared against the 3080 result before any
cross-stack family is reported.

WHAT IT CHECKS (same condition/seed, two result JSONs)
  1. ``eval_fingerprint`` equality — proves the *evaluation draw* is identical
     (a mismatch means the data/eval path differs: stop, do not compare numbers);
  2. the per-checkpoint perplexity ladder (all + content-only) — relative deltas;
  3. the ladder-probe deltas, when both cells carry them;
  4. the stack metadata of both cells (torch/CUDA/cuDNN/device);
  5. a verdict with explicit thresholds:
       drift <= 1%   -> "stack-equivalent" (cross-stack contrasts may be pooled
                        per family, still reported with the venue recorded)
       1% < d <= 5%  -> "minor drift" (pool only with a sensitivity note)
       > 5%          -> "INHOMOGENEOUS" (do not pool; re-run the family on one stack)

USAGE
    python3 bridge_check.py --a <3080>/exp1_result.json --b <newhost>/exp1_result.json
    python3 bridge_check.py --dir-a results --dir-b results_bridge --conditions parity_word,fixed_start
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THRESH_EQUIV, THRESH_MINOR = 1.0, 5.0


def ladder_steps(result: dict) -> list[str]:
    steps = set()
    for key in ("eval_gmean", "eval_gmean_content"):
        for s in (result.get(key) or {}):
            steps.add(str(s))
    return sorted(steps, key=lambda s: int(s))


def rel_delta(a: float, b: float) -> float:
    if a in (0, None) or b in (None,):
        return float("nan")
    return 100.0 * (b - a) / a


def compare(a: dict, b: dict) -> dict:
    out: dict = {"same_fingerprint": None, "ladder": {}, "extra": {}}
    fa, fb = a.get("eval_fingerprint"), b.get("eval_fingerprint")
    out["same_fingerprint"] = (fa == fb) and fa is not None
    for step in ladder_steps(a):
        ea, eb = (a.get("eval_gmean") or {}).get(step), (b.get("eval_gmean") or {}).get(step)
        ca, cb = (a.get("eval_gmean_content") or {}).get(step), (b.get("eval_gmean_content") or {}).get(step)
        out["ladder"][step] = {
            "ppl_3080": ea, "ppl_new": eb, "delta_pct": None if None in (ea, eb) else round(rel_delta(ea, eb), 3),
            "content_3080": ca, "content_new": cb,
            "content_delta_pct": None if None in (ca, cb) else round(rel_delta(ca, cb), 3),
        }
    pa, pb = a.get("ladder_probe") or {}, b.get("ladder_probe") or {}
    for step in sorted(set(pa) & set(pb), key=lambda s: int(s)):
        for branch in ("obey_first", "obey_last"):
            va = (pa[step].get(branch) or {}).get("mean_delta_nats")
            vb = (pb[step].get(branch) or {}).get("mean_delta_nats")
            if va is not None and vb is not None:
                out["extra"].setdefault("ladder_probe", {})[f"{step}/{branch}"] = round(vb - va, 4)
    out["stack_a"] = a.get("stack") or {"note": "no stack metadata (pre-§10c-11 cell)"}
    out["stack_b"] = b.get("stack") or {"note": "no stack metadata"}
    finals = [v["delta_pct"] for v in out["ladder"].values() if v["delta_pct"] is not None]
    worst = max((abs(d) for d in finals), default=float("nan"))
    out["worst_abs_delta_pct"] = round(worst, 3) if worst == worst else None
    if not out["same_fingerprint"]:
        out["verdict"] = "STOP: eval fingerprint differs — data/eval path mismatch, numbers are not comparable"
    elif worst != worst:
        out["verdict"] = "no comparable checkpoints"
    elif worst <= THRESH_EQUIV:
        out["verdict"] = f"stack-equivalent (worst {worst:.2f}% <= {THRESH_EQUIV}%)"
    elif worst <= THRESH_MINOR:
        out["verdict"] = f"minor drift (worst {worst:.2f}%) — pool with a sensitivity note"
    else:
        out["verdict"] = f"INHOMOGENEOUS (worst {worst:.2f}%) — do not pool; re-run the family on one stack"
    return out


def auto_pairs(dir_a: Path, dir_b: Path, conditions: list[str]) -> list[tuple[Path, Path]]:
    pairs = []
    for cond in conditions:
        for seed_dir in sorted((dir_b / f"babylm_{cond}_100M").glob("seed*")):
            cand_a = dir_a / f"babylm_{cond}_100M" / seed_dir.name / "exp1_result.json"
            cand_b = seed_dir / "exp1_result.json"
            if cand_a.exists() and cand_b.exists():
                pairs.append((cand_a, cand_b))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a"), ap.add_argument("--b")
    ap.add_argument("--dir-a"), ap.add_argument("--dir-b")
    ap.add_argument("--conditions", default="parity_word,fixed_start")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.a and args.b:
        pairs = [(Path(args.a), Path(args.b))]
    elif args.dir_a and args.dir_b:
        pairs = auto_pairs(Path(args.dir_a), Path(args.dir_b), args.conditions.split(","))
        if not pairs:
            print("no matching bridge pairs found", file=sys.stderr)
            return 1
    else:
        ap.error("give --a/--b or --dir-a/--dir-b")

    report = {}
    for pa, pb in pairs:
        key = f"{pa.parent.parent.name}/{pa.parent.name}"
        res = compare(json.loads(pa.read_text()), json.loads(pb.read_text()))
        report[key] = res
        print(f"\n### {key}\n  verdict: {res['verdict']}")
        print(f"  fingerprint equal: {res['same_fingerprint']}")
        for step, row in res["ladder"].items():
            print(f"  step {step:>5s}: ppl {row['ppl_3080']} -> {row['ppl_new']} "
                  f"({row['delta_pct']}%)  content {row['content_3080']} -> {row['content_new']} "
                  f"({row['content_delta_pct']}%)")
        for name, vals in (res.get("extra") or {}).items():
            print(f"  {name}: {vals}")
        print(f"  stack a: {res['stack_a'].get('stack_id')} {res['stack_a'].get('device_name')}")
        print(f"  stack b: {res['stack_b'].get('stack_id')} {res['stack_b'].get('device_name')}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())