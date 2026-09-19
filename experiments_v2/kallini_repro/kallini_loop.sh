#!/usr/bin/env bash
# kallini_loop.sh — chained ralph driver for the Kallini et al. reproduction.
# Deployed on the GPU box AFTER the v2 unit (llm-ralph-gpu) drains: this unit
# waits for it, then runs the idempotent kallini_queue.sh in a retry loop and
# publishes results to the GPU results branch (same commit-tree mechanism).

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO" || exit 3

STATE=experiments_v2/kallini_repro/results
mkdir -p "$STATE"
LOG="$STATE/kallini_loop.log"
log() { echo "[$(date -Is)] $*" >> "$LOG"; }

export HOME=${HOME:-/root}
export PATH=/root/anaconda3/bin:/usr/local/bin:$PATH
export PYTHON=${PYTHON:-/root/anaconda3/bin/python3}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export GIT_TERMINAL_PROMPT=0
RESULTS_BRANCH=${RESULTS_BRANCH:-v2-results-gpu}
SEEDS=${SEEDS:-"0 14 41"}
command -v gh >/dev/null && gh auth setup-git >/dev/null 2>&1 || true

log "kallini loop starting (branch $RESULTS_BRANCH, seeds '$SEEDS')"

# wait for the v2 GPU unit to drain (it exits cleanly when all phases done)
waited=0
while systemctl is-active --quiet llm-ralph-gpu 2>/dev/null; do
  waited=$((waited + 1))
  [ $((waited % 12)) -eq 0 ] && log "still waiting for llm-ralph-gpu (${waited}x5m)"
  sleep 300
done
log "v2 unit drained -> starting kallini repro"

iter=0
while true; do
  iter=$((iter+1))
  log "=== kallini iteration $iter ==="
  git pull --ff-only origin main >> "$LOG" 2>&1 || log "git pull failed (continuing)"

  if bash experiments_v2/kallini_repro/kallini_queue.sh >> "$STATE/queue_pass.log" 2>&1; then
    log "kallini queue SUCCESS"
    qok=1
  else
    log "kallini queue incomplete -> will retry"
    qok=0
  fi

  # publish results (temp index + commit-tree; never touches worktree/HEAD)
  pushed=0
  export GIT_INDEX_FILE="$STATE/tmpindex"
  git read-tree HEAD 2>> "$LOG"
  git add -f experiments_v2/kallini_repro/results experiments_v2/kallini_repro/*.log \
    ":(exclude)experiments_v2/kallini_repro/results/**/final/*" \
    ":(exclude)experiments_v2/kallini_repro/results/**/*.safetensors" \
    ":(exclude)experiments_v2/kallini_repro/results/**/*.bin" \
    ":(exclude)experiments_v2/kallini_repro/results/tmpindex*" \
    ":(exclude)experiments_v2/kallini_repro/results/.publish.lock" \
    2>> "$LOG" || true
  TREE=$(git write-tree 2>> "$LOG")
  unset GIT_INDEX_FILE
  COMMIT=$(git -c user.name="kallini-repro" -c user.email="kallini@repro.local" \
    commit-tree "$TREE" -p HEAD -m "kallini-repro results: iteration $iter" 2>> "$LOG")
  if git push -f origin "$COMMIT:refs/heads/$RESULTS_BRANCH" >> "$LOG" 2>&1; then
    log "results pushed to $RESULTS_BRANCH"
    pushed=1
  else
    log "results push FAILED"
  fi

  if [ "$qok" -eq 1 ] && [ "$pushed" -eq 1 ]; then
    log "KALLINI REPRO COMPLETE — exiting cleanly"
    exit 0
  fi
  sleep 180
done
