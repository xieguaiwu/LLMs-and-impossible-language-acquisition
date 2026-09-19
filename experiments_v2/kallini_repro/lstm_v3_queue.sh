#!/usr/bin/env bash
# lstm_v3_queue.sh — idempotent LSTM arm queue (DESIGN_V3 §2.2) for the 4-core CPU box.
#
# 7 conditions x 5 seeds = 35 runs, skip-if-done, N parallel workers.
# Own completion marker: results_lstm/ALL_LSTM_V3_DONE (never touches ALL_SVO_DONE).
#
# env knobs:
#   LSTM_WORKERS=3        parallel runs (each pinned to 1 compute thread)
#   LSTM_SEQ_LEN=256      window size (see train_exp1_lstm.py deviations)
#   LSTM_STEPS=3000       steps per run
#   CONDITIONS="..."      override the condition list
#   SEEDS="..."           override the seed list
#   PUBLISH=1             push results to RESULTS_BRANCH with weights excluded
#   RESULTS_BRANCH=v2-results-lstm
#
# Exit: 0 = every requested cell has a result; 1 = something still missing.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.." || exit 3   # repo root
REPO_ROOT="$PWD"

PYTHON=${PYTHON:-/root/anaconda3/bin/python3}
[ -x "$PYTHON" ] || PYTHON=$(command -v python3)
export PYTHONPATH="$REPO_ROOT/experiments_v2/kallini_repro:$REPO_ROOT/experiments_v2:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export KALLINI_DATA_PATH=${KALLINI_DATA_PATH:-/root/kallini_data}
export LSTM_RESULTS=${LSTM_RESULTS:-$SCRIPT_DIR/results_lstm}
export LSTM_SEQ_LEN=${LSTM_SEQ_LEN:-256}
export LSTM_STEPS=${LSTM_STEPS:-3000}
LSTM_WORKERS=${LSTM_WORKERS:-3}
RESULTS_BRANCH=${RESULTS_BRANCH:-v2-results-lstm}
PUBLISH=${PUBLISH:-0}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

CONDITIONS=${CONDITIONS:-"shuffle_control reverse_full reverse_control parity_word parity_tok negtok fixed_start"}
SEEDS=${SEEDS:-"0 14 41 53 96"}

mkdir -p "$LSTM_RESULTS"
LOG="$LSTM_RESULTS/queue.log"
note() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

# ---------- data gate: only queue conditions whose perturbed data is local ----
DATA_ROOT="$KALLINI_DATA_PATH/babylm_data_perturbed"
ready_conds=""
missing_conds=""
for c in $CONDITIONS; do
  d="$DATA_ROOT/babylm_$c"
  n_train=$(ls "$d"/babylm_100M/*.train 2>/dev/null | wc -l)
  n_test=$(ls "$d"/babylm_test_affected/*_affected.test 2>/dev/null | wc -l)
  if [ "$n_train" -ge 1 ] && [ "$n_test" -ge 1 ]; then
    ready_conds="$ready_conds $c"
  else
    missing_conds="$missing_conds $c"
    note "SKIP-DATA babylm_$c (train=$n_train test=$n_test) — run sync_from_gpu.sh"
  fi
done
note "queue pass: conditions=[$ready_conds ] seeds=[$SEEDS ] workers=$LSTM_WORKERS seq=$LSTM_SEQ_LEN steps=$LSTM_STEPS"
[ -z "${ready_conds// /}" ] && { note "no condition has local data — nothing to do"; exit 1; }

# ---------- run cells in parallel ----------
run_cell() {  # cond seed
  local cond=$1 seed=$2
  local out="$LSTM_RESULTS/babylm_${cond}_100M/seed${seed}/lstm_result.json"
  if [ -f "$out" ]; then
    echo "SKIP $cond seed$seed (done)"
    return 0
  fi
  if nice -n 10 "$PYTHON" "$SCRIPT_DIR/train_exp1_lstm.py" "$cond" --seed "$seed" --skip-if-done \
       >> "$LSTM_RESULTS/run_${cond}_seed${seed}.log" 2>&1; then
    echo "OK   $cond seed$seed"
  else
    echo "FAIL $cond seed$seed"
    return 1
  fi
}
export -f run_cell
export SCRIPT_DIR PYTHON LSTM_RESULTS

cells=""
for c in $ready_conds; do
  for s in $SEEDS; do
    cells="$cells$c $s"$'\n'
  done
done

fail=0
printf '%s' "$cells" | grep -v '^$' | xargs -P "$LSTM_WORKERS" -L1 bash -c 'run_cell "$@"' _ 2>&1 | tee -a "$LOG" || true

# ---------- verdict ----------
n_done=0
for c in $ready_conds; do
  for s in $SEEDS; do
    [ -f "$LSTM_RESULTS/babylm_${c}_100M/seed${s}/lstm_result.json" ] && n_done=$((n_done+1))
  done
done
expected=0
for c in $ready_conds; do for s in $SEEDS; do expected=$((expected+1)); done; done
note "queue pass done: $n_done/$expected cells complete (missing-data:${missing_conds:- none})"

# ---------- optional publish (weights excluded: LSTM ckpts are ~100-300 MB) ----
if [ "$PUBLISH" = "1" ] && [ -d "$LSTM_RESULTS" ]; then
  export GIT_INDEX_FILE="$LSTM_RESULTS/tmpindex"
  git read-tree HEAD 2>>"$LOG"
  git add -f -- \
    ":(exclude)experiments_v2/kallini_repro/results_lstm/**/*.pt" \
    ":(exclude)experiments_v2/kallini_repro/results_lstm/**/*.bin" \
    ":(exclude)experiments_v2/kallini_repro/results_lstm/**/*.safetensors" \
    experiments_v2/kallini_repro/results_lstm 2>>"$LOG" || true
  TREE=$(git write-tree 2>>"$LOG")
  unset GIT_INDEX_FILE
  if [ "$(git rev-parse "${TREE}^{tree}" 2>/dev/null)" != "$(git rev-parse 'HEAD^{tree}' 2>/dev/null)" ]; then
    COMMIT=$(git -c user.name="lstm-cpu2" -c user.email="lstm@cpu2.local" \
      commit-tree "$TREE" -p HEAD -m "results(lstm): $n_done/$expected cells ($(date -u +%FT%TZ))" 2>>"$LOG")
    if git push -f origin "$COMMIT:refs/heads/$RESULTS_BRANCH" >>"$LOG" 2>&1; then
      note "results pushed to $RESULTS_BRANCH"
    else
      note "results push FAILED"
      fail=1
    fi
  else
    note "no result changes to publish"
  fi
fi

if [ "$n_done" -eq "$expected" ]; then
  if [ -z "${missing_conds// /}" ]; then
    touch "$LSTM_RESULTS/ALL_LSTM_V3_DONE"
    note "ALL LSTM v3 CELLS DONE"
    exit 0
  fi
  note "ready conditions complete; ${missing_conds} still lack data -> retry after sync"
  exit 1
fi
note "incomplete: $((expected - n_done)) cell(s) missing"
exit 1
