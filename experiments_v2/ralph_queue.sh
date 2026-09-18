#!/usr/bin/env bash
# Ralph-loop experiment queue for the paper "LLMs and Impossible Language
# Acquisition" (v2 suite). Idempotent: every run is skipped when its
# training_metrics.json already exists, so this script can be retried
# forever by ralph_loop.sh without redoing finished work.
#
# Fails (exit != 0) if ANY queued cell is still missing after the pass,
# so the caller knows a retry is needed. Individual failures do NOT stop
# the queue -- all remaining cells still run in the same pass.

cd "$(dirname "$0")/../.."          # repo root
SEEDS=(42 43 44 45 46)
PYTHON=${PYTHON:-/root/anaconda3/bin/python3}   # systemd-safe
[ -x "$PYTHON" ] || PYTHON=$(command -v python3)
SKIP="--skip-if-done"

export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
if [ -z "${OMP_NUM_THREADS:-}" ]; then
  export OMP_NUM_THREADS=$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))
fi
NICE=""
command -v nice >/dev/null && NICE="nice -n 10"

fail=0
note() { echo "[$(date -Is)] $*"; }

# ---------- data ----------
if [ ! -f experiments_v2/data_v2/conditions/train/natural.txt ]; then
  note "generating SVO corpus + conditions"
  $PYTHON experiments_v2/data_v2/generate_svo.py --count 10000 --seed 42 || { note "DATA FAIL"; exit 9; }
  $PYTHON experiments_v2/data_v2/conditions.py || { note "DATA FAIL"; exit 9; }
fi

run() { # model dataset condition seed
  local model=$1 cond=$2 seed=$3
  if $NICE $PYTHON experiments_v2/training/train_lm.py \
      --model "$model" --dataset svo --condition "$cond" --seed "$seed" $SKIP \
      >> experiments_v2/results/queue_1.log 2>&1; then
    note "OK   $model/$cond/seed$seed"
  else
    note "FAIL $model/$cond/seed$seed"
    fail=$((fail+1))
  fi
}

mkdir -p experiments_v2/results

# ---------- main conditions x 5 seeds x 3 models (decision-first order) ------
# lstm_matched + gpt2_tiny first: capacity-matched claim (H5), cheap on CPU.
for cond in natural reversed parity_negation; do
  for seed in "${SEEDS[@]}"; do
    run gpt2_tiny "$cond" "$seed"
    run lstm_matched "$cond" "$seed"
  done
done
# gpt2 124M (heaviest) last among mains
for cond in natural reversed parity_negation; do
  for seed in "${SEEDS[@]}"; do
    run gpt2 "$cond" "$seed"
  done
done

# ---------- control conditions on gpt2 (H2/H3/H4) -----------------------------
for cond in fixed_start_neg fixed_end_neg parity_negation_negtok word_shuffle parity_negation_tok; do
  for seed in "${SEEDS[@]}"; do
    run gpt2 "$cond" "$seed"
  done
done

# ---------- probes on parity checkpoints (inference only) ----------------------
if [ "$fail" -eq 0 ]; then
  for seed in "${SEEDS[@]}"; do
    dir=experiments_v2/results/svo/gpt2/parity_negation_seed$seed
    if [ -f "$dir/probe_report.json" ]; then continue; fi
    $NICE $PYTHON experiments_v2/probes/probes.py --model-dir "$dir" --model gpt2 \
      --n-pairs 500 --extrapolation --probe-diagnostic \
      >> experiments_v2/results/probes.log 2>&1 \
      && note "OK   probe seed$seed" || { note "FAIL probe seed$seed"; fail=$((fail+1)); }
  done
fi

# ---------- analysis -----------------------------------------------------------
$PYTHON experiments_v2/analysis/aggregate_seeds.py >> experiments_v2/results/analysis.log 2>&1 \
  || fail=$((fail+1))
$PYTHON experiments_v2/analysis/stats_tests.py >> experiments_v2/results/analysis.log 2>&1 \
  || fail=$((fail+1))
$PYTHON experiments_v2/analysis/plots.py >> experiments_v2/results/analysis.log 2>&1 || true

# ---------- verdict ------------------------------------------------------------
n_done=$(find experiments_v2/results/svo -name training_metrics.json 2>/dev/null | wc -l)
note "queue pass done: $n_done runs complete, $fail failures this pass"
if [ "$fail" -gt 0 ]; then
  exit 1
fi
note "ALL DONE"
touch experiments_v2/results/ALL_SVO_DONE
exit 0
