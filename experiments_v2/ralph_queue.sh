#!/usr/bin/env bash
# Ralph-loop experiment queue for the paper "LLMs and Impossible Language
# Acquisition" (v2 suite). Idempotent: every run is skipped when its
# training_metrics.json already exists, so this script can be retried
# forever by ralph_loop.sh without redoing finished work.
#
# Fails (exit != 0) if ANY queued cell is still missing after the pass,
# so the caller knows a retry is needed. Individual failures do NOT stop
# the queue -- all remaining cells still run in the same pass.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # absolute: systemd-safe
cd "$SCRIPT_DIR/.."          # repo root (script lives in experiments_v2/)
SEEDS=(42 43 44 45 46)
PYTHON=${PYTHON:-/root/anaconda3/bin/python3}   # systemd-safe
[ -x "$PYTHON" ] || PYTHON=$(command -v python3)
SKIP="--skip-if-done"

# multi-host scoping (env):
#   MODELS         which SVO main models this host runs (default all three)
#   RUN_CONTROLS   1/0 gpt2 control conditions
#   RUN_PROBES     1/0 probe suite (needs gpt2 parity checkpoints on this host)
MODELS=${MODELS:-"gpt2_tiny lstm_matched gpt2"}
RUN_CONTROLS=${RUN_CONTROLS:-1}
RUN_PROBES=${RUN_PROBES:-1}
NICE_LEVEL=${NICE_LEVEL:-10}   # set NICE_LEVEL=off for dedicated boxes

export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
if [ -z "${OMP_NUM_THREADS:-}" ]; then
  export OMP_NUM_THREADS=$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))
fi
NICE="nice -n $NICE_LEVEL"
[ "$NICE_LEVEL" = "off" ] && NICE=""

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

run_if() { # model cond seed -> only when model in $MODELS
  local model=$1
  case " $MODELS " in *" $model "*) run "$model" "$2" "$3";; esac
}

mkdir -p experiments_v2/results

# ---------- main conditions x 5 seeds (decision-first order) -----------------
for cond in natural reversed parity_negation; do
  for seed in "${SEEDS[@]}"; do
    run_if gpt2_tiny    "$cond" "$seed"
    run_if lstm_matched "$cond" "$seed"
    run_if gpt2         "$cond" "$seed"
  done
done

# ---------- control conditions on gpt2 (H2/H3/H4) -----------------------------
if [ "$RUN_CONTROLS" = "1" ]; then
  for cond in fixed_start_neg fixed_end_neg parity_negation_negtok word_shuffle parity_negation_tok; do
    for seed in "${SEEDS[@]}"; do
      run_if gpt2 "$cond" "$seed"
    done
  done
fi

# ---------- extended-budget arms (H7: does the natural advantage EMERGE with
# more compute?) and polluted-corpus diagnostic (H8: are the original paper's
# Exp-1 numbers a corpus-duplication artifact?). GPU-only by default.
if [ "$RUN_EXTRA" = "1" ] && [ "$RUN_CONTROLS" = "1" ]; then
  mkdir -p experiments_v2/data_v2/conditions_polluted
  $PYTHON experiments_v2/data_v2/make_polluted.py natural reversed parity_negation \
    >> experiments_v2/results/polluted_prep.log 2>&1 || fail=$((fail+1))
  for cond in natural reversed parity_negation fixed_start_neg; do
    for seed in "${SEEDS[@]}"; do
      if $NICE $PYTHON experiments_v2/training/train_lm.py \
          --model gpt2 --dataset svo --condition "$cond" --seed "$seed" \
          --budget extended $SKIP >> experiments_v2/results/queue_1.log 2>&1; then
        note "OK   ext $cond/seed$seed"
      else
        note "FAIL ext $cond/seed$seed"; fail=$((fail+1))
      fi
    done
  done
  for cond in natural reversed parity_negation; do
    for seed in "${SEEDS[@]}"; do
      if $NICE $PYTHON experiments_v2/training/train_lm.py \
          --model gpt2 --dataset svo_polluted --condition "$cond" --seed "$seed" $SKIP \
          >> experiments_v2/results/queue_1.log 2>&1; then
        note "OK   polluted $cond/seed$seed"
      else
        note "FAIL polluted $cond/seed$seed"; fail=$((fail+1))
      fi
    done
  done
fi

# ---------- probes on parity checkpoints (inference only) ----------------------
if [ "$RUN_PROBES" = "1" ] && [ "$fail" -eq 0 ]; then
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
note "ALL DONE (models: $MODELS)"
touch experiments_v2/results/ALL_SVO_DONE
exit 0
