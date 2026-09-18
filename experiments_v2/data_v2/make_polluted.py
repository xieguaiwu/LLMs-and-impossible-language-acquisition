"""Corpus-pollution diagnostic (v2): reproduce the ORIGINAL paper's corpus bug.

The original ``data/generate_input.py`` wrote TWO lines per sentence into
input.txt::

    Original: <sentence>
    <sentence>

and ``generate_impossible.py`` treated every line as a sentence. The original
SVO corpora therefore contained every sentence TWICE per condition (plus an
"Original:" prefix artifact). Adjacent duplicates allow within-context
copying, which plausibly explains the original paper's very low natural-loss
(0.825) relative to clean-corpus runs (~1.7-1.9 in v2).

``make_polluted.py`` writes train/test copies of any v2 condition in that
exact polluted format, so we can quantify how much of the paper's Exp-1
pattern (natural 0.83 << parity 1.63 < reversed 1.81) is a corpus artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def pollute_lines(lines: list[str]) -> list[str]:
    out = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        out.append(f"Original: {ln}")
        out.append(ln)
        out.append("")          # blank separator, as in the original writer
    return out


def main() -> None:
    import sys

    repo = Path(__file__).resolve().parents[2]
    src = repo / "experiments_v2" / "data_v2" / "conditions"
    dst = repo / "experiments_v2" / "data_v2" / "conditions_polluted"
    conditions = sys.argv[1:] or ["natural", "reversed", "parity_negation"]
    for split in ("train", "test"):
        (dst / split).mkdir(parents=True, exist_ok=True)
        for cond in conditions:
            src_file = src / split / f"{cond}.txt"
            if not src_file.exists():
                print(f"skip missing {src_file}")
                continue
            with open(src_file, encoding="utf-8") as f:
                lines = f.read().splitlines()
            out_file = dst / split / f"{cond}.txt"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write("\n".join(pollute_lines(lines)) + "\n")
            print(f"{split}/{cond}: {len(lines)} sentences -> polluted format -> {out_file}")


if __name__ == "__main__":
    main()
