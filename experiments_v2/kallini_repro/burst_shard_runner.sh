#!/usr/bin/env bash
# burst_shard_runner.sh — run one shard file on one GPU, serially, with resume.
#
# A shard file is a TSV produced by make_burst_shards.py with the columns
#   kind  condition  seed  steps  env  result_path  note
# and carries the *exact* launch recipe (env + args) of kallini_queue.sh, so this
# runner stays generic: it executes cells in order, skips finished ones, and
# never touches the registered protocol.
#
# Design rules (burst plan 2026-09-21):
#   * one runner process per GPU; CUDA_VISIBLE_DEVICES pins the card;
#   * a cell is "done" iff its result JSON exists -> resume is idempotent, and an
#     unfinished cell simply re-runs (the window is never wasted on partials,
#     because the cell is retried on the 3080 host afterwards);
#   * per-cell log + a status line in $STATE, so the burst timeline is auditable;
#   * nice + optional CPU affinity, so several GPUs can share one box safely.
#
# USAGE
#   burst_shard_runner.sh <shard.tsv> <gpu_index> [--dry-run] [--repo DIR]
#                         [--log-dir DIR] [--affinity 0-7]
set -uo pipefail

SHARD=${1:?usage: burst_shard_runner.sh <shard.tsv> <gpu_index> [--dry-run]}
GPU=${2:?usage: burst_shard_runner.sh <shard.tsv> <gpu_index> [--dry-run]}
shift 2 || true
DRY=0
REPO=${BURST_REPO:-/root/llm-impossible}
LOGDIR=${BURST_LOGDIR:-/root/burst/logs}
AFFINITY=${BURST_AFFINITY:-}
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)   DRY=1 ;;
    --repo)      REPO=$2; shift ;;
    --log-dir)   LOGDIR=$2; shift ;;
    --affinity)  AFFINITY=$2; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

PYTHON=${BURST_PYTHON:-/root/anaconda3/envs/llmimp/bin/python3}
[ -x "$PYTHON" ] || PYTHON=$(command -v python3)
mkdir -p "$LOGDIR"
STATE="$LOGDIR/state_gpu${GPU}.tsv"
[ -f "$STATE" ] || printf 'utc\tkind\tcondition\tseed\tstatus\trc\tseconds\n' > "$STATE"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export GIT_TERMINAL_PROMPT=0
export OMP_NUM_THREADS=${BURST_OMP_THREADS:-4}
export MKL_NUM_THREADS=${BURST_OMP_THREADS:-4}

cd "$REPO" || exit 3
echo "[burst] gpu=$GPU shard=$SHARD repo=$REPO python=$PYTHON dry=$DRY"

ok=0; skip=0; fail=0
# header is skipped by the tail below
# 2026-09-24 fix: bash `read` with IFS=$'\t' collapses runs of tabs (tab is
# IFS-whitespace) -> EMPTY columns were lost and later fields shifted left
# (env parsed as steps; 36/55 ss_B rows failed). Convert tabs to '|' first.
while IFS= read -r __raw; do
  IFS='|' read -r kind condition seed steps env result_path note <<< "$(printf '%s' "$__raw" | tr '\t' '|')"

  [ "$kind" = "kind" ] && continue
  [ -z "${kind:-}" ] && continue
  if [ -n "${result_path:-}" ] && [ -f "$result_path" ]; then
    skip=$((skip+1))
    printf '%s\t%s\t%s\t%s\tSKIP\t0\t0\n' "$(date -u +%FT%TZ)" "$kind" "$condition" "$seed" >> "$STATE"
    continue
  fi
  case "$kind" in
    lstm_gpu|capmatch) script="experiments_v2/kallini_repro/train_exp1_lstm.py" ;;
    *)                 script="experiments_v2/kallini_repro/train_exp1.py" ;;
  esac
  args=("$condition" --seed "$seed")
  [ -n "${steps:-}" ] && args+=(--steps "$steps")
  args+=(--skip-if-done)

  tag="${kind}_${condition}_seed${seed}${steps:+_steps$steps}"
  log="$LOGDIR/${tag}.log"
  cmd=(env ${env:-} "$PYTHON" "$script" "${args[@]}")
  if [ -n "$AFFINITY" ]; then cmd=(taskset -c "$AFFINITY" "${cmd[@]}"); fi
  cmd=(nice -n "${BURST_NICE:-10}" "${cmd[@]}")

  if [ "$DRY" = "1" ]; then
    echo "[dry] $tag :: ${cmd[*]} >> $log"
    continue
  fi
  t0=$(date +%s)
  echo "[burst] START $tag $(date -Is)"
  "${cmd[@]}" >> "$log" 2>&1
  rc=$?
  dt=$(( $(date +%s) - t0 ))
  if [ $rc -eq 0 ]; then
    ok=$((ok+1)); st=OK
  else
    fail=$((fail+1)); st=FAIL
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$kind" "$condition" "$seed" "$st" "$rc" "$dt" >> "$STATE"
  echo "[burst] $st $tag rc=$rc ${dt}s"
done < <(tail -n +2 "$SHARD")

echo "[burst] DONE gpu=$GPU ok=$ok skip=$skip fail=$fail"
[ "$fail" -eq 0 ]