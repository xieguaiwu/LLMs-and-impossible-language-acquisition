#!/usr/bin/env python3
"""Regenerate the v3 class-P condition datasets (single source of truth).

Called by ``kallini_queue.sh`` §[3b] (and usable standalone on any host that has
the tagged shim JSONs). Byte-identical output on both hosts is the point: the
two arms (GPU GPT-2 / CPU LSTM) must train on the same files, and the pool
version marker makes a *semantics* change detectable, which a file-existence
gate cannot do (2026-09-20 audit).

POOL VERSION
    pool-v1  (deprecated): filter applied to the PERTURBED token count, so every
             markered condition silently kept a different sentence set than
             ``negtok`` / the S/R classes.
    pool-v2  (current):    filter applied to the BASE sentence tokenization
             (Kallini ``filter_shuffle`` semantics: 1 < n <= 350), then the
             transformation. All class-P conditions share one sentence set by
             construction (EXPDESIGN_V3 §1.3.2).

USAGE
    python3 regenerate_conditions.py                 # fill in what is missing
    python3 regenerate_conditions.py --force         # rebuild everything (pool change)
    python3 regenerate_conditions.py --conditions parity_word not_random
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_conditions import write_condition  # noqa: E402

POOL_VERSION = "pool-v2-base-filter"
DEFAULT_CONDITIONS = ["parity_word", "parity_tok", "negtok", "fixed_start",
                      "fixed_end", "not_random", "bare_reverse", "word_shuffle"]


def complete(base: Path, lang: str, tagged_train: list[str], tagged_test: list[str]) -> bool:
    d_train = base / "babylm_data_perturbed" / f"babylm_{lang}" / "babylm_100M"
    d_test = base / "babylm_data_perturbed" / f"babylm_{lang}" / "babylm_test_affected"
    return (all((d_train / f"{Path(tf).stem}.train").exists() for tf in tagged_train)
            and all((d_test / f"{Path(tf).stem}_affected.test").exists() for tf in tagged_test))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=os.environ.get("KALLINI_DATA_PATH", "/root/kallini_data"))
    ap.add_argument("--conditions", nargs="*", default=DEFAULT_CONDITIONS)
    ap.add_argument("--force", action="store_true",
                    help="regenerate every condition even if the files exist (pool change)")
    ap.add_argument("--no-version-marker", action="store_true")
    args = ap.parse_args()

    base = Path(args.data_root)
    tagged_train = sorted(glob.glob(str(base / "babylm_data" / "babylm_100M" / "*_parsed.json")))
    tagged_test = sorted(glob.glob(str(base / "babylm_data" / "babylm_test" / "*_parsed.json")))
    assert tagged_train and tagged_test, (
        f"shim tag the corpus first: expected *_parsed.json under "
        f"{base}/babylm_data/{{babylm_100M,babylm_test}}")

    out = base / "babylm_data_perturbed"
    totals: dict[str, int] = {}
    for lang in args.conditions:
        if complete(base, lang, tagged_train, tagged_test) and not args.force:
            print(f"{lang} skip (all genres present)")
            continue
        n_train = n_test = 0
        for tf in tagged_train:
            n_train += write_condition(lang, Path(tf), out, "100M")["n"]
        for tf in tagged_test:
            n_test += write_condition(lang, Path(tf), out, "test")["n"]
        totals[lang] = n_train
        print(f"[done] {lang}: train={n_train} test={n_test}", flush=True)

    if totals and not args.no_version_marker:
        (out / ".pool_version").write_text(POOL_VERSION + "\n")
        print(f"pool version marker -> {out / '.pool_version'} = {POOL_VERSION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
