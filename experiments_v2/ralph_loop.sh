#!/usr/bin/env bash
# ralph_loop.sh — non-stop, self-updating experiment driver.
#
# Deployed on the GPU/compute server via systemd-run (survives session
# logout; nohup gets killed on this infra). Each iteration:
#   1. git pull origin main        (the agent can steer by pushing code)
#   2. ensure python deps
#   3. bash experiments_v2/ralph_queue.sh   (idempotent, resumable)
#   4. push results to origin v2-results branch (observability channel)
#   5. if queue returned success AND results were pushed -> exit 0
#      otherwise sleep and retry forever ("ralph 不停跑").
#
# Never touches VERSION2.5 / Bitkrieg / any production cron data. Runs niced
# with systemd resource properties set at deployment time.

set -u
cd /root/llm-impossible || exit 3
STATE=experiments_v2/results
mkdir -p "$STATE"
LOG="$STATE/ralph.log"

log() { echo "[$(date -Is)] $*" >> "$LOG"; }

export PYTHON=${PYTHON:-python3}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export GIT_TERMINAL_PROMPT=0

# ---- one-time environment preparation ---------------------------------------
if ! $PYTHON -c "import torch, transformers, scipy, sklearn, pandas, matplotlib" 2>/dev/null; then
  log "installing python deps"
  pip install -q torch transformers datasets accelerate scipy scikit-learn \
      pandas matplotlib tqdm >> "$STATE/deps.log" 2>&1 \
    || pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple torch transformers \
       datasets accelerate scipy scikit-learn pandas matplotlib tqdm >> "$STATE/deps.log" 2>&1 \
    || { log "DEPS INSTALL FAILED (see deps.log)"; exit 4; }
fi
$PYTHON -c "import spacy" 2>/dev/null || pip install -q spacy >> "$STATE/deps.log" 2>&1 || true

iter=0
while true; do
  iter=$((iter+1))
  log "=== iteration $iter ==="

  # 1. self-update (agent steering channel)
  git pull --ff-only origin main >> "$LOG" 2>&1 || log "git pull failed (continuing)"

  # 2-3. run the idempotent queue
  if bash experiments_v2/ralph_queue.sh >> "$STATE/queue_pass.log" 2>&1; then
    log "queue SUCCESS"
    qok=1
  else
    log "queue incomplete -> will retry"
    qok=0
  fi

  # 4. publish results to the v2-results branch (best effort, every pass)
  pushed=0
  if [ -d "$STATE" ]; then
    git add -f experiments_v2/results experiments_v2/data_v2/conditions 2>> "$LOG"
    if ! git diff --cached --quiet 2>> "$LOG"; then
      git -c user.name="ralph-server" -c user.email="ralph@server.local" \
        commit -q -m "results: iteration $iter ($(date -u +%FT%TZ))" 2>> "$LOG" || true
    fi
    if git push -q origin HEAD:refs/heads/v2-results 2>> "$LOG"; then
      log "results pushed to v2-results"
      pushed=1
    else
      log "results push FAILED"
    fi
  fi

  if [ "$qok" -eq 1 ] && [ "$pushed" -eq 1 ]; then
    log "ALL WORK COMPLETE — ralph loop exiting cleanly"
    exit 0
  fi
  log "sleeping 120s before retry"
  sleep 120
done
