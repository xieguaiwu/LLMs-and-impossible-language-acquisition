#!/usr/bin/env python3
"""data_integrity_check.py -- cross-condition sentence-set integrity audit (report-only).

WHY THIS EXISTS (2026-09-20)
    gpu2 `babylm_parity_word/babylm_100M/simple_wikipedia_parsed.train` was found
    truncated to 632,436 of 1,023,786 lines (interrupted write at 16:43 in the
    16:25-16:43 regeneration pass). The queue's data gate only checks that every
    per-genre train/test file EXISTS, so a truncated file passes. The GPT-2 arm
    would have trained the treatment condition (`parity_word`) on 3.9% less data
    than its own control (`fixed_start`) -- a systematic bias in the direction of
    the paper's central hypothesis -- while the cpu2 LSTM arm held the complete
    file, breaking the architecture comparison as well.

WHAT IT CHECKS
    1. Per-genre line counts must be IDENTICAL across the conditions of one pool
       (train and test separately). Pools = condition families that share the
       same sentence set by design.
    2. Optional md5 fingerprints per genre (--md5 G) so the same command on two
       hosts proves the two arms train on byte-identical data.

WHAT IT DOES NOT DO
    No writes, no repair, no deletion. Exit code 1 on any mismatch.

USAGE
    python3 data_integrity_check.py --data-dir /root/kallini_data/babylm_data_perturbed
    python3 data_integrity_check.py --md5 aochildes          # + cross-host md5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

GENRES = ["aochildes", "bnc_spoken", "cbt", "children_stories", "gutenberg",
          "open_subtitles", "qed", "simple_wikipedia", "switchboard", "wikipedia"]

# Conditions that must share one sentence set (per design: identical sentence IDs).
POOLS = {
    "S_class": ["shuffle_control", "shuffle_nondeterministic", "shuffle_deterministic21",
                "shuffle_local3", "shuffle_local10", "shuffle_even_odd"],
    "R_class": ["reverse_control", "reverse_partial", "reverse_full"],
    "P_class_perturbed_filter": ["parity_word", "parity_tok", "fixed_start", "fixed_end"],
}
# Single-member pools: no internal identity to check, counted for the record only.
SINGLETONS = {
    "P_class_base_filter": ["negtok"],
    "reference": ["word_shuffle", "bare_reverse"],
}


def train_file(data_dir: Path, cond: str, genre: str) -> Path:
    # S/R class: {genre}.train ; class P (v3_conditions): {genre}_parsed.train
    for name in (f"{genre}_parsed.train", f"{genre}.train"):
        p = data_dir / f"babylm_{cond}" / "babylm_100M" / name
        if p.exists():
            return p
    return data_dir / f"babylm_{cond}" / "babylm_100M" / f"{genre}_parsed.train"


def test_file(data_dir: Path, cond: str, genre: str) -> Path:
    for name in (f"{genre}_parsed_affected.test", f"{genre}_affected.test"):
        p = data_dir / f"babylm_{cond}" / "babylm_test_affected" / name
        if p.exists():
            return p
    return data_dir / f"babylm_{cond}" / "babylm_test_affected" / f"{genre}_parsed_affected.test"


def count_lines(path: Path) -> int | None:
    if not path.exists():
        return None
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="/root/kallini_data/babylm_data_perturbed")
    ap.add_argument("--md5", default=None, metavar="GENRE",
                    help="also print md5 of that genre's train+test file per condition")
    ap.add_argument("--json", default=None, help="write the full result table to this JSON")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"FAIL data dir missing: {data_dir}")
        return 1

    problems: list[str] = []
    report: dict = {"data_dir": str(data_dir), "pools": {}, "md5": {}, "problems": []}

    all_pools = dict(POOLS)
    for pool, conds in SINGLETONS.items():
        all_pools[pool] = conds

    print(f"data dir: {data_dir}")
    for pool, conds in all_pools.items():
        present = [c for c in conds if (data_dir / f"babylm_{c}").is_dir()]
        if not present:
            continue
        report["pools"][pool] = {"conditions": present, "split": {}}
        for split, fn in (("train", train_file), ("test", test_file)):
            table: dict[str, dict[str, int | None]] = {}
            for genre in GENRES:
                table[genre] = {c: count_lines(fn(data_dir, c, genre)) for c in present}
            report["pools"][pool]["split"][split] = table
            if len(present) < 2:
                continue
            for genre, row in table.items():
                vals = {v for v in row.values() if v is not None}
                missing = [c for c, v in row.items() if v is None]
                if missing:
                    problems.append(f"[{pool}/{split}/{genre}] missing file(s): {missing}")
                if len(vals) > 1:
                    lo = min(vals)
                    detail = " ".join(f"{c}={v}" for c, v in row.items())
                    problems.append(f"[{pool}/{split}/{genre}] count mismatch: {detail}"
                                    f"  (min={lo}, ratio={max(vals)/lo:.4f})")

    # ---- verdict + compact printout -----------------------------------------
    for pool, data in report["pools"].items():
        conds = data["conditions"]
        print(f"\n== {pool}: {len(conds)} condition(s) -- {', '.join(conds)}")
        for split in ("train", "test"):
            table = data["split"][split]
            tot = {c: sum(v for v in (table[g][c] or 0 for g in GENRES)) for c in conds}
            print(f"   {split}: " + " ".join(f"{c}={tot[c]:,}" for c in conds))
            for genre in GENRES:
                row = table[genre]
                vals = {v for v in row.values() if v is not None}
                if len(vals) > 1:
                    print(f"     !! {genre}: " +
                          " ".join(f"{c}={v:,}" if v is not None else f"{c}=MISSING"
                                   for c, v in row.items()))

    if args.md5:
        g = args.md5
        print(f"\n== md5 fingerprints (genre={g}) -- compare across hosts with this command")
        for conds in all_pools.values():
            for c in conds:
                if not (data_dir / f"babylm_{c}").is_dir():
                    continue
                for split, fn in (("train", train_file), ("test", test_file)):
                    p = fn(data_dir, c, g)
                    digest = md5(p) if p.exists() else "MISSING"
                    report["md5"][f"{c}/{split}"] = digest
                    print(f"   {c:24s} {split:5s} {digest}")

    report["problems"] = problems
    if problems:
        print("\nFAIL -- pool identity violated:")
        for p in problems:
            print("  " + p)
    else:
        print("\nOK -- every pool has identical per-genre sentence counts (train + test)")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"report -> {args.json}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
