#!/usr/bin/env bash
# finish_pool_v2_restart.sh — one-shot post-regeneration gate for the GPU box.
#
# Why: on 2026-09-20 the class-P datasets were regenerated under a new sentence
# pool (pool-v2-base-filter) and the queue/trainer gained the audit amendments
# (entropy-matched control, extension tier, evaluation hygiene, GPU LSTM arm).
# Those changes must be adopted by a *fresh* bash process: bash reads a running
# script incrementally, so editing or pulling under a live pass is the documented
# "half-new script" incident (RUNBOOK rule 13).
#
# What it does, exactly once, after the regeneration process exits:
#   1. pull origin/main (fixture pull; the loop would do it anyway)
#   2. verify the regenerated pool (data_integrity_check.py -> OK)
#   3. verify the pool-version marker
#   4. validate the new queue script end-to-end with QUEUE_DRY_RUN=1
#   5. only then rebuild the kallini-gpu unit (documented scenario-2 rebuild)
# On any failure it writes /root/pool_v2_ALERT.md and exits WITHOUT restarting, so
# the sentinel (and the queue's own pool-version gate) self-heals.
set -u
REPO=/root/llm-impossible
cd "$REPO" || { echo "no repo" > /root/pool_v2_ALERT.md; exit 1; }
export HOME=${HOME:-/root}
export PATH=/root/anaconda3/bin:/usr/local/bin:$PATH
export GIT_TERMINAL_PROMPT=0
LOG=/root/pool_v2_gate.log
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

# 1. wait for the regeneration to finish
while pgrep -f 'regenerate_conditions[.]py' >/dev/null 2>&1; do sleep 60; done
log "regeneration process gone"
sleep 10

# 2. adopt the new code
git pull --ff-only origin main >> "$LOG" 2>&1 || log "WARN git pull failed (continuing)"
git rev-parse --short HEAD >> "$LOG"

# 3. verify pool identity + version marker
python3 experiments_v2/kallini_repro/data_integrity_check.py --md5 aochildes >> "$LOG" 2>&1
check_rc=$?
pool_ver=$(cat /root/kallini_data/babylm_data_perturbed/.pool_version 2>/dev/null)
log "integrity rc=$check_rc pool_version='$pool_ver'"
if [ "$check_rc" -ne 0 ] || [ "$pool_ver" != "pool-v2-base-filter" ]; then
  printf '%s\n' "pool-v2 gate FAILED: integrity rc=$check_rc pool_version='$pool_ver' — see $LOG" \
    > /root/pool_v2_ALERT.md
  log "ABORT: not restarting the unit (sentinel will retry the gate)"
  exit 1
fi

# 4. end-to-end dry validation of the amended queue (no training, no marker)
if QUEUE_DRY_RUN=1 bash experiments_v2/kallini_repro/kallini_queue.sh >> "$LOG" 2>&1; then
  log "dry queue pass OK"
else
  printf '%s\n' "pool-v2 gate FAILED: QUEUE_DRY_RUN pass returned non-zero (see $LOG)" \
    > /root/pool_v2_ALERT.md
  log "ABORT: dry pass failed"
  exit 1
fi

# 5. rebuild the main-loop unit (RUNBOOK scenario 2; identical to the sentinel cmd)
# 5. wait for a cell boundary, then rebuild the main-loop unit
#    (killing a running cell wastes up to ~4 h; waiting for the next
#     exp1_result.json costs at most one cell and loses ~1 min instead)
n0=$(find experiments_v2/kallini_repro/results -name exp1_result.json 2>/dev/null | wc -l)
waited=0
while :; do
  n=$(find experiments_v2/kallini_repro/results -name exp1_result.json 2>/dev/null | wc -l)
  [ "$n" -gt "$n0" ] && { log "cell boundary: $n0 -> $n"; break; }
  systemctl is-active --quiet kallini-gpu || { log "unit already inactive -> restarting"; break; }
  [ "$waited" -ge 300 ] && { log "boundary wait 5h timeout -> restarting anyway"; break; }
  sleep 30; waited=$((waited + 1))
done
sleep 15

systemctl stop kallini-gpu >> "$LOG" 2>&1 || true
systemd-run --unit=kallini-gpu --property=Restart=on-failure --property=RestartSec=1min \
  --working-directory="$REPO" --setenv=HOME=/root --setenv=GIT_TERMINAL_PROMPT=0 \
  --setenv="RESULTS_BRANCH=v2-results-gpu" --setenv="SEEDS=0 14 41" \
  --setenv=RUN_V3=1 --setenv=RUN_V3_H7=1 \
  /usr/bin/bash experiments_v2/kallini_repro/kallini_loop.sh >> "$LOG" 2>&1
log "kallini-gpu unit rebuilt (rc=$?)"
rm -f /root/pool_v2_ALERT.md
exit 0
