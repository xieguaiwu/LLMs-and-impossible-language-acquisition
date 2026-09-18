#!/usr/bin/env bash
# BabyLM queue (v2 replication of the paper's Experiment 2): the cell the
# original study left at n=1, now n=5 seeds. Idempotent + failure-tolerant
# like ralph_queue.sh. Requires babylm_conditions prepared (prepare_babylm.py).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # absolute: systemd-safe
cd "$SCRIPT_DIR/../.."   # repo root
SEEDS=(42 43 44 45 46)
PYTHON=${PYTHON:-/root/anaconda3/bin/python3}   # systemd-safe
[ -x "$PYTHON" ] || PYTHON=$(command -v python3)
SKIP="--skip-if-done"

export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-3}
NICE="nice -n 10"

fail=0
note() { echo "[$(date -Is)] $*"; }
mkdir -p experiments_v2/results

# ---------- prepare babylm conditions ----------
if [ ! -f experiments_v2/data_v2/babylm_conditions/train/natural.txt ]; then
  note "preparing babylm conditions (sentence split + transforms)"
  $PYTHON -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null \
    || pip install -q spacy >> experiments_v2/results/deps.log 2>&1 || true
  $PYTHON -m spacy validate 2>/dev/null | grep -q en_core_web_sm \
    || $PYTHON -m spacy download en_core_web_sm >> experiments_v2/results/deps.log 2>&1 \
    || note "spacy model unavailable -> regex splitter fallback"
  $NICE $PYTHON experiments_v2/data_v2/prepare_babylm.py \
    --raw experiments_v2/data_v2/babylm/raw \
    >> experiments_v2/results/babylm_prepare.log 2>&1 \
    || { note "BABYLM PREP FAIL"; exit 9; }
fi

run() {
  local model=$1 cond=$2 seed=$3
  if $NICE $PYTHON experiments_v2/training/train_lm.py \
      --model "$model" --dataset babylm --condition "$cond" --seed "$seed" $SKIP \
      >> experiments_v2/results/queue_babylm.log 2>&1; then
    note "OK   babylm $model/$cond/seed$seed"
  else
    note "FAIL babylm $model/$cond/seed$seed"
    fail=$((fail+1))
  fi
}

for cond in natural reversed parity_negation; do
  for seed in "${SEEDS[@]}"; do
    run gpt2 "$cond" "$seed"
  done
done

# ---------- probes ----------
for seed in "${SEEDS[@]}"; do
  dir=experiments_v2/results/babylm/gpt2/parity_negation_seed$seed
  if [ -f "$dir/probe_report.json" ] || [ ! -f "$dir/training_metrics.json" ]; then continue; fi
  $NICE $PYTHON experiments_v2/probes/probes.py --model-dir "$dir" --model gpt2 \
    --n-pairs 500 --extrapolation --probe-diagnostic \
    >> experiments_v2/results/probes_babylm.log 2>&1 \
    && note "OK   babylm probe seed$seed" || fail=$((fail+1))
done

# ---------- analysis ----------
$PYTHON experiments_v2/analysis/aggregate_seeds.py >> experiments_v2/results/analysis.log 2>&1 || fail=$((fail+1))
$PYTHON experiments_v2/analysis/stats_tests.py >> experiments_v2/results/analysis.log 2>&1 || fail=$((fail+1))
$PYTHON experiments_v2/analysis/plots.py >> experiments_v2/results/analysis.log 2>&1 || true

n_done=$(find experiments_v2/results/babylm -name training_metrics.json 2>/dev/null | wc -l)
note "babylm pass done: $n_done runs complete, $fail failures"
if [ "$fail" -gt 0 ]; then exit 1; fi
touch experiments_v2/results/ALL_BABYLM_DONE
exit 0
