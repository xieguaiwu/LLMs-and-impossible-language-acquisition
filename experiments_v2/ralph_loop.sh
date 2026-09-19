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

export PYTHON=${PYTHON:-/root/anaconda3/bin/python3}   # systemd PATH lacks anaconda
[ -x "$PYTHON" ] || PYTHON=$(command -v python3)
export HOME=${HOME:-/root}   # systemd transient units lack HOME -> gh/git credentials dead
export PATH=/root/anaconda3/bin:/usr/local/bin:$PATH
command -v gh >/dev/null && gh auth setup-git >/dev/null 2>&1 || true
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export GIT_TERMINAL_PROMPT=0

# ---- one-time environment preparation ---------------------------------------
# Install ONLY missing packages; never list torch explicitly (it would upgrade
# the existing CPU torch to the latest CUDA wheel).
MISSING=""
for mod in transformers datasets accelerate scipy sklearn pandas matplotlib tqdm; do
  $PYTHON -c "import $mod" 2>/dev/null || MISSING="$MISSING $mod"
done
if [ "${MISSING:-}" ]; then
  log "installing missing python deps:$MISSING"
  pip install -q $MISSING >> "$STATE/deps.log" 2>&1 \
    || pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple $MISSING >> "$STATE/deps.log" 2>&1 \
    || { log "DEPS INSTALL FAILED (see deps.log)"; exit 4; }
fi
$PYTHON -c "import torch" 2>/dev/null || { log "torch missing and not auto-installed -- install CPU torch manually"; exit 4; }
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

  # 4. publish results to v2-results via temp index + commit-tree:
  #    never touches HEAD/branch/worktree, so a failed push cannot poison main
  pushed=0
  if [ -d "$STATE" ]; then
    export GIT_INDEX_FILE="$STATE/tmpindex"
    git read-tree HEAD 2>> "$LOG"
    # weights excluded (500MB/run x many runs would bloat the object db)
    git add -f -- \
      ":(exclude)experiments_v2/results/**/*.safetensors" \
      ":(exclude)experiments_v2/results/**/*.bin" \
      ":(exclude)experiments_v2/results/**/*.pt" \
      ":(exclude)experiments_v2/results/**/final/**" \
      experiments_v2/results 2>> "$LOG" || true
    TREE=$(git write-tree 2>> "$LOG")
    unset GIT_INDEX_FILE
    if [ "$(git rev-parse "${TREE}^{tree}" 2>/dev/null)" != "$(git rev-parse 'HEAD^{tree}' 2>/dev/null)" ]; then
      COMMIT=$(git -c user.name="ralph-server" -c user.email="ralph@server.local" \
        commit-tree "$TREE" -p HEAD -m "results: iteration $iter ($(date -u +%FT%TZ))" 2>> "$LOG")
      if git push -f origin "$COMMIT:refs/heads/v2-results" >> "$LOG" 2>&1; then
        log "results pushed to v2-results"
        pushed=1
      else
        log "results push FAILED"
      fi
    else
      log "no result changes to publish"
      pushed=1
    fi
  fi

  if [ "$qok" -eq 1 ] && [ "$pushed" -eq 1 ]; then
    # ---- phase 2: BabyLM (gated on RUN_BABYLM + raw-data fetch success) ----
    # RUN_BABYLM must gate the phase itself, not only the exit check below:
    # with RUN_BABYLM=0 (v2 BabyLM batch-4 arm retired, redteam #3 + prereg
    # deviation log) the loop used to start fetch_babylm.sh + babylm_queue.sh
    # anyway and then still exit as if only SVO had run.
    if [ "${RUN_BABYLM:-0}" = "1" ] && [ ! -f "$STATE/ALL_BABYLM_DONE" ]; then
      log "SVO complete -> starting BabyLM phase"
      if bash experiments_v2/fetch_babylm.sh >> "$STATE/fetch_babylm.log" 2>&1 \
         && bash experiments_v2/babylm_queue.sh >> "$STATE/queue_pass_babylm.log" 2>&1; then
        log "BabyLM phase SUCCESS"
      else
        log "BabyLM phase failed -> will retry next iteration"
      fi
      # publish again regardless (same commit-tree mechanism)
      export GIT_INDEX_FILE="$STATE/tmpindex_b"
      git read-tree HEAD 2>> "$LOG"
      git add -f -- \
        ":(exclude)experiments_v2/results/**/*.safetensors" \
        ":(exclude)experiments_v2/results/**/*.bin" \
        ":(exclude)experiments_v2/results/**/*.pt" \
        ":(exclude)experiments_v2/results/**/final/**" \
        experiments_v2/results experiments_v2/data_v2 2>> "$LOG" || true
      TREE=$(git write-tree 2>> "$LOG")
      unset GIT_INDEX_FILE
      COMMIT=$(git -c user.name="ralph-server" -c user.email="ralph@server.local" \
        commit-tree "$TREE" -p HEAD -m "results: babylm iteration $iter" 2>> "$LOG")
      if git push -f origin "$COMMIT:refs/heads/$RESULTS_BRANCH" >> "$LOG" 2>&1; then
        log "babylm results pushed"
      else
        log "babylm results push FAILED"
      fi
    fi
    if [ -f "$STATE/ALL_SVO_DONE" ] && { [ "${RUN_BABYLM:-0}" = "0" ] || [ -f "$STATE/ALL_BABYLM_DONE" ]; }; then
      log "ALL PHASES COMPLETE — ralph loop exiting cleanly"
      exit 0
    fi
    log "sleeping 120s before next phase/retry"
    sleep 120
  fi
  log "sleeping 120s before retry"
  sleep 120
done
