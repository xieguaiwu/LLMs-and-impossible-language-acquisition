#!/usr/bin/env python3
"""Byte-identity check: LSTM packer vs the GPT-2 arm's packer (train_exp1).

Why this exists
---------------
The architecture axis of the v3 paper is only valid if the LSTM consumes the
*identical* token stream as the GPT-2 arm. ``train_exp1_lstm._sentence_stream``
is a memory-frugal numpy twin of ``train_exp1.load_packed_dataset``; this script
proves the twin is exact on truncated copies of the real corpus (multi-file
concatenation order, sentence shuffle, EOS placement, token ids).

Run:
    KALLINI_DATA_PATH=/root/kallini_data python packing_equivalence_check.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

COND = "shuffle_control"
KEEP_LINES = 300          # per file; keeps the check cheap while still covering
N_FILES = 2               # multi-file concatenation order


def build_truncated_root(tmp: Path, src_root: Path) -> Path:
    src = src_root / "babylm_data_perturbed" / f"babylm_{COND}" / "babylm_100M"
    files = sorted(src.glob("*.train"))[:N_FILES]
    assert len(files) == N_FILES, f"need {N_FILES} train files under {src}, found {len(files)}"
    dst = tmp / "babylm_data_perturbed" / f"babylm_{COND}" / "babylm_100M"
    dst.mkdir(parents=True, exist_ok=True)
    for f in files:
        lines = f.read_text().splitlines()[:KEEP_LINES]
        (dst / f.name).write_text("\n".join(lines) + "\n")
    return tmp


def main() -> int:
    src_root = Path(os.environ.get("KALLINI_DATA_PATH", "/root/kallini_data"))
    real = src_root / "babylm_data_perturbed" / f"babylm_{COND}" / "babylm_100M"
    if not real.exists():
        print(f"SKIP: no local data at {real} (run sync_from_gpu.sh)")
        return 2

    tmp = Path(tempfile.mkdtemp(prefix="packcheck_"))
    try:
        build_truncated_root(tmp, src_root)
        os.environ["KALLINI_DATA_PATH"] = str(tmp)
        import train_exp1 as G          # noqa: E402  (must import after env set)
        import train_exp1_lstm as L     # noqa: E402

        for seed in (0, 14, 41):
            blocks = G.load_packed_dataset(COND, seed)              # their path
            assert all(len(b) == G.SEQ_LEN for b in blocks), (
                "upstream block width changed (tail-drop invariant broken)")
            theirs = np.array([t for b in blocks for t in b], dtype=np.int32)
            windows, kept, n_sents = L.packed_blocks(COND, seed)   # mine
            mine = windows.reshape(-1)
            _stream, lens = L._sentence_stream(COND)
            total = int(_stream.size)
            expected_kept = (total // L.SEQ_LEN) * L.SEQ_LEN
            same_len = len(theirs) == len(mine) == kept == expected_kept
            same_tok = bool(np.array_equal(theirs, mine))
            print(f"seed={seed:>3} sentences={n_sents:>7} tokens={total:>9} "
                  f"kept={kept:>9} (dropped tail={total - kept:>4}) "
                  f"len_match={same_len} identical={same_tok}")
            if not (same_len and same_tok):
                n = min(len(theirs), len(mine))
                bad = np.flatnonzero(theirs[:n] != mine[:n])[:5]
                print(f"  MISMATCH len theirs={len(theirs)} mine={len(mine)} "
                      f"kept={kept} at {bad.tolist()}")
                return 1
        print("PASS: LSTM packer reproduces the GPT-2 arm's token stream exactly "
              "(same tail-drop semantics, all windows full-width)")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
