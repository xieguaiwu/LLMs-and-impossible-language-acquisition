#!/usr/bin/env bash
# Best-effort official BabyLM 100M acquisition: cambridge-climb/BabyLM
# per-genre files (exactly Kallini's GENRES layout) via HF mirror.
# Safe to re-run (skips when raw data already present).
# The earlier Sree1994/babylm_100M mirror turned out to be a 10M-word corpus
# under a 100M name — caught by the word-count guard on 2026-09-19.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # absolute: systemd-safe
cd "$SCRIPT_DIR/.."   # repo root (script lives in experiments_v2/)

RAW_DIR=experiments_v2/data_v2/babylm/raw
mkdir -p "$RAW_DIR"
BASE="https://hf-mirror.com/datasets/cambridge-climb/BabyLM/resolve/main/clean"
GENRES="aochildes bnc_spoken cbt children_stories gutenberg open_subtitles qed simple_wikipedia switchboard wikipedia"

if ls "$RAW_DIR"/100M_*.txt >/dev/null 2>&1; then
  echo "raw BabyLM already present"
else
  for g in $GENRES; do
    f="$RAW_DIR/100M_${g}.txt"
    [ -f "$f" ] || curl -sL --retry 3 -o "$f" "$BASE/100M/$g.txt" \
      || { echo "HF FETCH FAIL $g"; exit 9; }
  done
fi

WORDS=$(cat "$RAW_DIR"/100M_*.txt 2>/dev/null | wc -w || echo 0)
echo "train corpus words: $WORDS"
if [ "$WORDS" -lt 50000000 ]; then
  echo "BabyLM corpus suspiciously small ($WORDS words) -- aborting phase"
  exit 9
fi
touch experiments_v2/results/BABYLM_RAW_READY
echo "BabyLM raw data ready"
