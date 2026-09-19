#!/usr/bin/env bash
# publish_results.sh — incremental results publisher for the kallini/v3 grid.
#
# Why it exists: kallini_loop.sh only publishes when a whole queue pass ends. One
# pass is 42 cells x ~4.1h (~8 days on one GPU), so nothing reaches the results
# branch for days and a machine loss would take the unpublished work with it.
# This script publishes whatever is on disk now, every 15 minutes.
#
# Safety:
#   * builds the commit from a TEMP INDEX + commit-tree -> never touches the
#     worktree or HEAD, so a failed push cannot poison main;
#   * WEIGHTS ARE EXCLUDED (final/ dirs, *.safetensors, *.bin): one GPT-2 cell
#     saves ~500 MB, GitHub rejects files > 100 MB, and a rejected push leaves
#     GBs of loose objects behind (observed 2026-09-19: .git grew to 31 GB).
#     Per-sentence perplexities (ppls_step*.pt, ~90 KB) ARE kept: the analysis
#     needs them for like-for-like subset metrics;
#   * flock-guarded, and it skips the push when the tree already matches the
#     branch, so the loop's own pass-end publish and this one do not fight.
#
# Usage: bash experiments_v2/kallini_repro/publish_results.sh [--dry-run]
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO" || exit 3

STATE=experiments_v2/kallini_repro/results
RESULTS_BRANCH=${RESULTS_BRANCH:-v2-results-gpu}
LOG="$STATE/publish.log"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

export HOME=${HOME:-/root}
export PATH=/root/anaconda3/bin:/usr/local/bin:$PATH
export GIT_TERMINAL_PROMPT=0
log() { echo "[$(date -Is)] $*" >> "$LOG"; }

mkdir -p "$STATE"
exec 9>"$STATE/.publish.lock" || exit 3
flock -n 9 || { echo "publish already running"; exit 0; }

# pathspecs: include the results tree and logs, exclude anything heavy
SPEC=(
  experiments_v2/kallini_repro/results
  experiments_v2/kallini_repro/*.log
  ":(exclude)experiments_v2/kallini_repro/results/**/final/*"
  ":(exclude)experiments_v2/kallini_repro/results/**/*.safetensors"
  ":(exclude)experiments_v2/kallini_repro/results/**/*.bin"
  ":(exclude)experiments_v2/kallini_repro/results/tmpindex*"
  ":(exclude)experiments_v2/kallini_repro/results/.publish.lock"
)

export GIT_INDEX_FILE="$STATE/tmpindex_pub"
git read-tree HEAD
git add -f "${SPEC[@]}" >/dev/null 2>&1 || true
TREE=$(git write-tree)
unset GIT_INDEX_FILE

cells=$(find "$STATE" -name exp1_result.json 2>/dev/null | wc -l)
if [ "$DRY" = "1" ]; then
  echo "tree: $TREE | completed cells: $cells"
  echo "--- staged files (heavy files must NOT appear) ---"
  git ls-tree -r --name-only "$TREE" | grep "kallini_repro/results" | head -20
  echo "--- heavy files staged: $(git ls-tree -r --name-only "$TREE" | grep -cE 'final/|\.safetensors$|\.bin$') (expect 0) ---"
  exit 0
fi

git fetch -q origin "$RESULTS_BRANCH" 2>/dev/null || true
REMOTE_TREE=$(git rev-parse "origin/$RESULTS_BRANCH^{tree}" 2>/dev/null || echo "")
if [ -n "$REMOTE_TREE" ] && [ "$REMOTE_TREE" = "$(git rev-parse "$TREE^{tree}")" ]; then
  echo "no changes to publish (cells=$cells)"; exit 0
fi

COMMIT=$(git -c user.name="kallini-repro" -c user.email="kallini@repro.local" \
  commit-tree "$TREE" -p HEAD -m "kallini-repro results (incremental): ${cells} cells done $(date -u +%FT%TZ)")
if git push -f origin "$COMMIT:refs/heads/$RESULTS_BRANCH" >> "$LOG" 2>&1; then
  log "incremental publish OK -> $RESULTS_BRANCH (cells=$cells)"
  echo "published ${cells} cells -> $RESULTS_BRANCH"
else
  log "incremental publish FAILED (cells=$cells)"
  echo "publish FAILED (see $LOG)"
  exit 1
fi
