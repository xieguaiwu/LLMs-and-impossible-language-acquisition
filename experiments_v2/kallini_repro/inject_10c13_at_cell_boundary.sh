#!/usr/bin/env bash
# inject_10c13_at_cell_boundary.sh — activate the §10c-13 amendments (positive-
# evidence arms G1_ppd/G2_cald/G3_ags, NoPE moved up, deferrals DL-1..4) on the
# GPU box without losing a training cell. Pattern: inject_10c (RUNBOOK 场景 6).
#
# Sequence (logged to /root/inject_10c13.log):
#   1. wait for the in-flight cell (parity_word/seed0) to be OK'd in queue_pass.log
#      — NOT while it trains, NOT while its final evaluation runs;
#   2. git pull --ff-only origin main (rename-safe for the running bash);
#   3. stop + reset-failed the loop unit;
#   4. dry-run the new queue (must exit 0; dry-run is side-effect free);
#   5. restart the unit with the registered environment;
#   6. verify + write status.
set -u

REPO=/root/llm-impossible
LOG=/root/inject_10c13.log
STATUS=/root/INJECT_10C13_STATUS.md
FAILED=/root/INJECT_10C13_FAILED.md
QUEUE_LOG="$REPO/experiments_v2/kallini_repro/results/queue_pass.log"
BOUNDARY_CELL="${1:-parity_word/seed0}"
MAX_WAIT_MIN="${MAX_WAIT_MIN:-420}"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }
fail() {
  log "ABORT: $*"
  {
    echo "# INJECT 10C-13 FAILED"
    echo
    echo "- time: $(date -Is)"
    echo "- reason: $*"
    echo "- action: loop unit left as-is; re-run after fixing or let the sentinel restart the pass."
  } > "$FAILED"
  exit 1
}

log "injector 10c-13 started (boundary cell: $BOUNDARY_CELL, max wait ${MAX_WAIT_MIN}m)"

# ---- 1. wait for the boundary ------------------------------------------------
waited=0
while ! grep -qa "OK   ${BOUNDARY_CELL}" "$QUEUE_LOG"; do
  waited=$((waited + 1))
  [ $((waited % 40)) -eq 0 ] && log "still waiting for 'OK   ${BOUNDARY_CELL}' (${waited}x15s)"
  if [ "$waited" -ge $((MAX_WAIT_MIN * 4)) ]; then
    fail "timeout waiting for the boundary cell ${BOUNDARY_CELL}"
  fi
  sleep 15
done
log "boundary reached — stopping the unit before the next cell starts training"
systemctl stop kallini-gpu 2>/dev/null || true
systemctl reset-failed kallini-gpu 2>/dev/null || true

# ---- 2. pull -----------------------------------------------------------------
git -C "$REPO" pull --ff-only origin main >> "$LOG" 2>&1 || fail "git pull failed"
HEAD=$(git -C "$REPO" rev-parse --short HEAD)
log "worktree at $HEAD (expect >= 10c-13 registration commit)"

# ---- 3. dry-run validation ---------------------------------------------------
cd "$REPO" || fail "cd $REPO failed"
if QUEUE_DRY_RUN=1 RUN_V3=1 RUN_V3_H7=1 SEEDS="0 14 41" EXT_SEEDS="53 96" \
   bash experiments_v2/kallini_repro/kallini_queue.sh \
     > /root/inject_10c13_dryrun.log 2>&1; then
  log "dry-run OK ($(grep -c 'dry\]' /root/inject_10c13_dryrun.log) planned actions)"
else
  fail "dry-run FAILED — see /root/inject_10c13_dryrun.log (unit left stopped)"
fi

# ---- 4. restart with the registered environment ------------------------------
systemctl reset-failed kallini-gpu 2>/dev/null || true
if ! systemd-run --unit=kallini-gpu \
      --property=Restart=on-failure --property=RestartSec=1min \
      --working-directory=/root/llm-impossible \
      --setenv=HOME=/root --setenv=GIT_TERMINAL_PROMPT=0 \
      --setenv="RESULTS_BRANCH=v2-results-gpu" --setenv="SEEDS=0 14 41" \
      --setenv="EXT_SEEDS=53 96" \
      --setenv=RUN_V3=1 --setenv=RUN_V3_H7=1 \
      /usr/bin/bash experiments_v2/kallini_repro/kallini_loop.sh >> "$LOG" 2>&1; then
  if systemctl is-active --quiet kallini-gpu; then
    log "unit already active (sentinel restart) — verifying it runs the new code"
  else
    fail "systemd-run failed and the unit is not active"
  fi
fi

# ---- 5. verify ---------------------------------------------------------------
sleep 45
active=$(systemctl is-active kallini-gpu 2>/dev/null || echo unknown)
pend=$(cd "$REPO" && python3 experiments_v2/kallini_repro/grid_status.py --one-line 2>/dev/null || echo "?")
{
  echo "# INJECT 10C-13 STATUS"
  echo
  echo "- completed: $(date -Is)"
  echo "- worktree HEAD: \`$HEAD\`"
  echo "- dry-run: OK (\`/root/inject_10c13_dryrun.log\`)"
  echo "- loop unit: **$active**"
  echo "- registered grid: \`$pend\`"
  echo
  echo "§10c-13 arms now live: NoPE base moved to [4b2]; nope_ext [4b3] gated by"
  echo "  .t1_smoke_ok; cald pilot [4b4] gated by data dirs; cald confirmatory gated"
  echo "  by .frozen_cald; datascale 12→4 / model_scale 6→2 / logo+ladder deferred."
  echo "Next expected: the pass continues the P block, then NoPE (T1), then the rest."
} > "$STATUS"
log "done: unit=$active"
cat "$STATUS"
