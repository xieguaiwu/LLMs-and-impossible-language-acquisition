#!/usr/bin/env bash
# Best-effort BabyLM 100M acquisition: HuggingFace mirror -> raw text files.
# Safe to re-run (skips when raw data already present). Non-fatal failures
# are tolerated: ralph_loop only gates the BabyLM phase on the marker file.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # absolute: systemd-safe
cd "$SCRIPT_DIR/.."   # repo root (script lives in experiments_v2/)

RAW_DIR=experiments_v2/data_v2/babylm/raw
mkdir -p "$RAW_DIR"
BASE="https://hf-mirror.com/datasets/Sree1994/babylm_100M/resolve/main/data"

if ls "$RAW_DIR"/train*.txt >/dev/null 2>&1; then
  echo "raw BabyLM already present"
else
  for spec in "train-00000-of-00001-483d4930a18b1694.parquet" \
              "valid-00000-of-00001-dbeb923c899f3527.parquet"; do
    f="$RAW_DIR/$spec"
    [ -f "$f" ] || curl -sL --retry 3 -o "$f" "$BASE/$spec" || return 1 2>/dev/null || exit 9
  done
  python3 - <<'EOF'
import pandas as pd, pathlib
raw = pathlib.Path("experiments_v2/data_v2/babylm/raw")
for pq in sorted(raw.glob("*.parquet")):
    df = pd.read_parquet(pq)
    text_col = next(c for c in df.columns if df[c].map(lambda v: isinstance(v, str)).all())
    out = raw / (pq.stem + ".txt")
    with open(out, "w", encoding="utf-8") as f:
        for row in df[text_col]:
            f.write(str(row).replace("\n", " ") + "\n")
    print(f"{pq.name} -> {out} ({sum(len(str(r)) for r in df[text_col])/1e6:.1f}M chars)")
EOF
fi

WORDS=$(cat "$RAW_DIR"/train*.txt 2>/dev/null | wc -w || echo 0)
echo "train corpus words: $WORDS"
if [ "$WORDS" -lt 50000000 ]; then
  echo "BabyLM corpus suspiciously small ($WORDS words) -- aborting phase"
  exit 9
fi
touch experiments_v2/results/BABYLM_RAW_READY
echo "BabyLM raw data ready"
