#!/usr/bin/env python3
"""make_datascale_subsets.py — deterministic reduced-corpus variants of the v3 pools.

Registered 2026-09-20 evening (preregistration.md §10c-5 / §10c-8). Two variants:

  DATA-SCALE axis (§10c-5, the PoS analog)
      ``babylm_<cond>_sub1M``  /  ``babylm_<cond>_sub10M``
      Train = a deterministic stratified subsample of the condition's TRAIN pool
      (target ≈ 1e6 / 1e7 BPE tokens, proportional to each genre's token share),
      test = an exact copy of the condition's own test pool. Because the budget
      of the consuming cells stays fixed (3000 steps x 128 x 1024), the axis
      asks: at FIXED optimisation budget, how much does data scarcity hurt each
      condition? If the natural-language advantage grows as data shrinks, the
      transformer's inductive bias behaves like an architectural prior in the
      sense the poverty-of-the-stimulus argument attributes to humans.

  LOGO generalization (§10c-8)
      ``babylm_<cond>_logo7sw``
      Train = the same pool with the ``simple_wikipedia`` genre REMOVED (9/10 of
      the corpus), test = the full copy. The analysis compares the LOGO model's
      per-genre perplexity on the held-out genre against the full-data model's
      (same frozen evaluation draw), i.e. domain transfer rather than in-domain
      fit.

Design rules
------------
* The subsample is drawn from the condition's OWN transformed pool, so the
  transformation (and therefore the markered/parity structure) is intact; only
  the number of sentences changes.
* Deterministic: ``numpy.random.default_rng(seed)`` with seed fixed per
  (condition, scale, genre) — no dependence on file order or run time.
* Sampling is a without-replacement permutation of the genre's line indices;
  lines are taken until the genre's token target is reached, then written in
  the ORIGINAL file order (the trainer shuffles sentences itself).
* A manifest JSON (``_datascale_manifest.json``) records per-file line counts
  and token totals, so an audit can verify a subset without recomputing it.

USAGE
    python3 make_datascale_subsets.py                 # build the data-scale dirs
    python3 make_datascale_subsets.py --logo          # build the LOGO dirs
    python3 make_datascale_subsets.py --force         # rewrite existing dirs
    python3 make_datascale_subsets.py --selftest      # sampling logic only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

DATA_ROOT = Path(os.environ.get("KALLINI_DATA_PATH", "/root/kallini_data")) / "babylm_data_perturbed"
GENRES = ["aochildes", "bnc_spoken", "cbt", "children_stories", "gutenberg",
          "open_subtitles", "qed", "simple_wikipedia", "switchboard", "wikipedia"]
DATASCALE_CONDS = ["shuffle_control", "parity_word", "fixed_start"]
LOGO_CONDS = ["shuffle_control", "parity_word"]
LOGO_EXCLUDE = "simple_wikipedia"
SCALES = {"sub1M": 1_000_000, "sub10M": 10_000_000}
SEED_BASE = 20260920


def _file_of(cond: str, genre: str, kind: str) -> Path:
    # The P pool uses the ``_parsed`` naming; the S/R pool stores plain
    # ``{genre}.train`` / ``{genre}_affected.test``.  Accept both.
    root = DATA_ROOT / f"babylm_{cond}"
    if kind == "train":
        for name in (f"{genre}_parsed.train", f"{genre}.train"):
            cand = root / "babylm_100M" / name
            if cand.exists():
                return cand
        return root / "babylm_100M" / f"{genre}_parsed.train"
    for name in (f"{genre}_parsed_affected.test", f"{genre}_affected.test"):
        cand = root / "babylm_test_affected" / name
        if cand.exists():
            return cand
    return root / "babylm_test_affected" / f"{genre}_parsed_affected.test"


def _line_token_counts(text: str) -> np.ndarray:
    """Token count per line (space-separated BPE ids), vectorised enough for 1M lines."""
    lines = text.splitlines()
    if not lines:
        return np.zeros(0, dtype=np.int64)
    return np.fromiter((l.count(" ") + 1 if l.strip() else 0 for l in lines),
                       dtype=np.int64, count=len(lines))


def sample_indices(tokens: np.ndarray, target: int, seed: int) -> np.ndarray:
    """Without-replacement sample of line indices totalling ~= target tokens.

    Deterministic greedy walk over a seeded permutation: take a line when it
    still fits the remaining budget, else skip. Returns sorted indices.
    """
    n = tokens.size
    if n == 0 or target <= 0:
        return np.zeros(0, dtype=np.int64)
    perm = np.random.default_rng(seed).permutation(n)
    take: list[int] = []
    used = 0
    for i in perm:
        t = int(tokens[i])
        if t == 0:
            continue
        if used + t > target and take:
            continue
        take.append(int(i))
        used += t
        if used >= target:
            break
    return np.sort(np.array(take, dtype=np.int64))


def write_subset(cond: str, out_dir_name: str, budget: int | None,
                 exclude_genre: str | None, force: bool) -> dict:
    """Build one variant directory; returns its manifest entry."""
    out_root = DATA_ROOT / out_dir_name
    train_dir = out_root / "babylm_100M"
    test_dir = out_root / "babylm_test_affected"
    manifest_path = out_root / "_datascale_manifest.json"
    if manifest_path.exists() and not force:
        return json.loads(manifest_path.read_text())

    genres = [g for g in GENRES if g != exclude_genre]
    # token share per genre (from the parent condition's train pool)
    load: dict[str, np.ndarray] = {}
    for g in GENRES:
        f = _file_of(cond, g, "train")
        if not f.exists():
            raise SystemExit(f"missing parent train file: {f}")
        load[g] = _line_token_counts(f.read_text())
    total_tokens = int(sum(int(v.sum()) for v in load.values()))
    kept_tokens = int(sum(int(load[g].sum()) for g in genres))

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    entry: dict = {"condition": cond, "dir": out_dir_name, "budget_tokens": budget,
                   "exclude_genre": exclude_genre, "parent_total_tokens": total_tokens,
                   "genres": {}}
    for g in genres:
        toks = load[g]
        seed = SEED_BASE + 7 * GENRES.index(g) + (budget or 0) % 1_000_003
        if budget is None:                      # LOGO: keep the whole genre
            idx = np.arange(toks.size)
        else:
            share = int(round(budget * (int(toks.sum()) / max(1, kept_tokens))))
            idx = sample_indices(toks, share, seed)
        lines = _file_of(cond, g, "train").read_text().splitlines()
        kept = [lines[i] for i in idx]
        (train_dir / f"{g}_parsed.train").write_text("\n".join(kept) + "\n")
        src_test = _file_of(cond, g, "test")
        dst_test = test_dir / f"{g}_parsed_affected.test"
        shutil.copyfile(src_test, dst_test)
        entry["genres"][g] = {"lines": len(kept), "tokens": int(toks[idx].sum()),
                              "source_lines": int(toks.size)}
    entry["total_lines"] = int(sum(v["lines"] for v in entry["genres"].values()))
    entry["total_tokens"] = int(sum(v["tokens"] for v in entry["genres"].values()))
    manifest_path.write_text(json.dumps(entry, indent=2))
    return entry


def selftest() -> int:
    ok = True
    toks = np.array([5, 3, 7, 1, 4, 2], dtype=np.int64)
    idx = sample_indices(toks, 10, seed=1)
    ok &= int(toks[idx].sum()) >= 10 and int(toks[idx].sum()) <= 10 + 7   # reaches ~target
    ok &= sample_indices(toks, 10, seed=1).tolist() == idx.tolist()        # deterministic
    ok &= np.all(np.diff(idx) > 0)                                        # sorted, unique
    ok &= sample_indices(toks, 0, seed=1).size == 0
    # token counting
    ok &= _line_token_counts("1 2 3\n\n4 5\n").tolist() == [3, 0, 2]
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logo", action="store_true", help="build the LOGO (hold out one genre) dirs")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    built = []
    if args.logo:
        for cond in LOGO_CONDS:
            built.append(write_subset(cond, f"babylm_{cond}_logo7sw", None,
                                      LOGO_EXCLUDE, args.force))
    else:
        for cond in DATASCALE_CONDS:
            for scale, budget in SCALES.items():
                built.append(write_subset(cond, f"babylm_{cond}_{scale}", budget,
                                          None, args.force))
    for e in built:
        print(f"{e['dir']}: {e['total_lines']} lines / {e['total_tokens']} tokens "
              f"(parent {e['parent_total_tokens']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
