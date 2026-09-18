"""Aggregate Experiment-1 reproduction results (Kallini et al. 2024).

Reads experiments_v2/kallini_repro/results/babylm_{lang}_{train_set}/seed*/exp1_result.json
and writes:
  - exp1_summary.csv: language, seed, step, gmean_ppl
  - exp1_summary.md: per-step table (languages x checkpoints), mirroring the
    ordering narrative of their Figure 2.
"""

from __future__ import annotations

import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def main() -> None:
    rows = []
    per_lang: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(str(RESULTS / "babylm_*" / "seed*" / "exp1_result.json"))):
        r = json.load(open(f))
        lang, seed = r["language"], r["seed"]
        for step, gmean in sorted(r["eval_gmean"].items(), key=lambda kv: int(kv[0])):
            rows.append({"language": lang, "seed": seed, "step": int(step),
                         "gmean_ppl": gmean})
            rows[-1]["gmean_ppl"] = gmean
    if not rows:
        print("no exp1 results yet")
        return
    import csv

    out_csv = RESULTS / "exp1_summary.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["language", "seed", "step", "gmean_ppl"])
        w.writeheader()
        w.writerows(rows)
        # rows may carry the gmean assignment bug-free? ensure above
    # markdown table: mean over seeds per (language, step)
    agg = defaultdict(list)
    for r in rows:
        agg[(r["language"], r["step"])].append(r["gmean_ppl"])
    steps = sorted({r["step"] for r in rows})
    langs = sorted({r["language"] for r in rows}, key=lambda L: (L != "shuffle_control", L != "reverse_control", L))
    lines = ["# Experiment 1 reproduction — geometric-mean test PPL",
             "", "language | " + " | ".join(f"step {s}" for s in steps),
             "--- | " + " | ".join("---" for _ in steps)]
    for L in langs:
        cells = []
        for s in steps:
            v = agg.get((L, s))
            cells.append(f"{st.mean(v):.1f} (n={len(v)})" if v else "—")
        lines.append(f"{L} | " + " | ".join(cells))
    out_md = RESULTS / "exp1_summary.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_csv} and {out_md}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
