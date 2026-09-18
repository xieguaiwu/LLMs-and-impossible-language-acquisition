#!/usr/bin/env bash
# kallini_queue.sh — idempotent reproduction queue for Experiment 1 of
# Kallini et al. (2024), Shuffle + Reverse classes (9 languages x N seeds).
# Idempotent + failure-tolerant, same pattern as ralph_queue.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."   # repo root

PYTHON=${PYTHON:-/root/anaconda3/bin/python3}
[ -x "$PYTHON" ] || PYTHON=$(command -v python3)
SEEDS=${SEEDS:-"0 14 41"}
NICE_LEVEL=${NICE_LEVEL:-off}
NICE="nice -n $NICE_LEVEL"
[ "$NICE_LEVEL" = "off" ] && NICE=""

export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export KALLINI_REPO=${KALLINI_REPO:-/root/mission-impossible-language-models}
export KALLINI_DATA_PATH=${KALLINI_DATA_PATH:-/root/kallini_data}
export REPRO_RESULTS=${REPRO_RESULTS:-experiments_v2/kallini_repro/results}

LANGS="shuffle_control shuffle_nondeterministic shuffle_deterministic21 shuffle_local3 shuffle_local10 shuffle_even_odd reverse_control reverse_partial reverse_full"

fail=0
note() { echo "[$(date -Is)] $*"; }
mkdir -p experiments_v2/kallini_repro/results

# ---------- [1] their repo ------------------------------------------------------
if [ ! -d "$KALLINI_REPO" ]; then
  git clone -q https://github.com/jkallini/mission-impossible-language-models.git "$KALLINI_REPO" \
    || { note "KALLINI CLONE FAIL"; exit 9; }
fi
grep -q "^BABYLM_DATA_PATH = \"${KALLINI_DATA_PATH}\"" "$KALLINI_REPO/utils.py" || \
  sed -i "s|^BABYLM_DATA_PATH = .*|BABYLM_DATA_PATH = \"${KALLINI_DATA_PATH}\"|" "$KALLINI_REPO/utils.py"

# ---------- [2] BabyLM raw text from HF mirror (same source as fetch_babylm) ----
if [ ! -f "$KALLINI_DATA_PATH/babylm_data/babylm_100M/all_parsed.json" ]; then
  note "fetching BabyLM parquet via HF mirror"
  mkdir -p "$KALLINI_DATA_PATH/parquet"
  for spec in train-00000-of-00001-483d4930a18b1694.parquet \
              valid-00000-of-00001-dbeb923c899f3527.parquet \
              test-00000-of-00001-dbeb923c899f3527.parquet; do
    [ -f "$KALLINI_DATA_PATH/parquet/$spec" ] || \
      curl -sL --retry 3 -o "$KALLINI_DATA_PATH/parquet/$spec" \
        "https://hf-mirror.com/datasets/Sree1994/babylm_100M/resolve/main/data/$spec" \
        || { note "HF FETCH FAIL"; exit 9; }
  done
  note "writing raw text files"
  $PYTHON - <<'PYEOF' >> experiments_v2/kallini_repro/data_prep.log 2>&1 \
    || { note "TEXT EXTRACT FAIL"; exit 9; }
import pandas as pd, glob, os
base = "/root/kallini_data"
os.makedirs(f"{base}/babylm_data/babylm_100M", exist_ok=True)
os.makedirs(f"{base}/babylm_data/babylm_dev", exist_ok=True)
os.makedirs(f"{base}/babylm_data/babylm_test", exist_ok=True)
def dump(pq_glob, out_path):
    pq = sorted(glob.glob(pq_glob))[0]
    df = pd.read_parquet(pq)
    col = next(c for c in df.columns if df[c].map(lambda v: isinstance(v, str)).all())
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in df[col]:
            f.write(str(row).replace("\n", " ").strip() + "\n")
            n += 1
    print(f"{out_path}: {n} lines")
dump(f"{base}/parquet/train*.parquet", f"{base}/babylm_data/babylm_100M/all.train")
dump(f"{base}/parquet/valid*.parquet", f"{base}/babylm_data/babylm_dev/all.dev")
dump(f"{base}/parquet/test*.parquet",  f"{base}/babylm_data/babylm_test/all.test")
PYEOF
  note "shim-tagging train + test"
  $PYTHON experiments_v2/kallini_repro/shim_tag.py \
    "$KALLINI_DATA_PATH/babylm_data/babylm_100M/all.train" \
    "$KALLINI_DATA_PATH/babylm_data/babylm_test/all.test" \
    >> experiments_v2/kallini_repro/data_prep.log 2>&1 \
    || { note "SHIM TAG FAIL"; exit 9; }
fi

# ---------- [3] perturbed datasets via THEIR perturb.py -------------------------
need_perturb() {
  [ ! -d "$KALLINI_DATA_PATH/babylm_data_perturbed/babylm_$1/babylm_100M" ]
}
if need_perturb shuffle_control || need_perturb reverse_partial; then
  note "perturbing train (100M) + test splits with their perturb.py"
  printf '%s\n' $LANGS | xargs -P 3 -I{} bash -c '
    NICE_LEVEL="'$NICE_LEVEL'"; NICE="nice -n $NICE_LEVEL"; [ "$NICE_LEVEL" = off ] && NICE=""
    '"$PYTHON"' '"$KALLINI_REPO"'/data/perturb.py {} 100M >> '"$PWD"'/experiments_v2/kallini_repro/data_prep.log 2>&1 || true
    '"$PYTHON"' '"$KALLINI_REPO"'/data/perturb.py {} test  >> '"$PWD"'/experiments_v2/kallini_repro/data_prep.log 2>&1 || true
  ' || note "WARN some perturb jobs failed (see data_prep.log)"
fi

# ---------- [4] training queue --------------------------------------------------
run() {
  local lang=$1 seed=$2
  if $NICE $PYTHON experiments_v2/kallini_repro/train_exp1.py "$lang" --seed "$seed" --skip-if-done \
      >> experiments_v2/kallini_repro/queue.log 2>&1; then
    note "OK   $lang/seed$seed"
  else
    note "FAIL $lang/seed$seed"
    fail=$((fail+1))
  fi
}

for seed in $SEEDS; do
  for lang in $LANGS; do
    run "$lang" "$seed"
  done
done

# ---------- [5] aggregate --------------------------------------------------------
$PYTHON experiments_v2/kallini_repro/aggregate_exp1.py >> experiments_v2/kallini_repro/queue.log 2>&1 \
  || note "WARN aggregation failed"

n_done=$(find experiments_v2/kallini_repro/results -name exp1_result.json 2>/dev/null | wc -l)
note "kallini pass done: $n_done runs complete, $fail failures"
if [ "$fail" -gt 0 ]; then exit 1; fi
touch experiments_v2/kallini_repro/results/ALL_KALLINI_DONE
exit 0
