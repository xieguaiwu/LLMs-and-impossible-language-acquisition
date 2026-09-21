#!/usr/bin/env bash
# bootstrap_burst_host.sh — prepare a second (Blackwell/5090) host for the v3 grid.
#
# WHAT IT DOES (idempotent; safe to re-run)
#   1. preflight: GPU model/capability, driver, CPU/RAM/disk, network reachability
#   2. conda env `llmimp` (python 3.12) with a torch that actually has sm_120
#      kernels — the preinstalled torch 2.5 has none, so it is NOT used
#   3. verification: sm_120 matmul + fp16 autocast + GradScaler smoke
#   4. repo clone at a pinned commit + data (rsync or regenerate) + md5/pool check
#   5. shard generation from the registered manifest (needs the 3080-side results
#      tree ONLY for the pending list; a fresh host can also start from
#      --shard-file copied over)
#   6. readiness report (and optionally two systemd runners, one per GPU)
#
# USAGE (on the new host)
#   bash bootstrap_burst_host.sh preflight
#   bash bootstrap_burst_host.sh env
#   bash bootstrap_burst_host.sh repo   --commit <hash>
#   bash bootstrap_burst_host.sh data   --from user@host:/root/kallini_data
#   bash bootstrap_burst_host.sh shards --shard-src /root/burst/shards   # generated on the 3080
#   bash bootstrap_burst_host.sh run    --shards shard_p_block.tsv,shard_arch.tsv
#   bash bootstrap_burst_host.sh bench  --cell parity_word --seed 0 --steps 300
#
# NOTES
#   * never install over the system python; everything lives in the conda env;
#   * the training code needs no GPU-specific edits — stack_compat.py handles the
#     torch 2.2 -> 2.7+ API move and pins the numerics explicitly;
#   * results go to per-host branches (see RESULTS_BRANCH below) so two hosts
#     never fight over one branch.
set -uo pipefail

REPO_DIR=${BURST_REPO:-/root/llm-impossible}
DATA_DIR=${KALLINI_DATA_PATH:-/root/kallini_data}
ENV_NAME=${BURST_ENV:-llmimp}
PIN_COMMIT=${BURST_COMMIT:-}
RESULTS_BRANCH=${BURST_RESULTS_BRANCH:-v2-results-burst}
CODE_URL=${BURST_CODE_URL:-https://github.com/xieguaiwu/LLMs-and-impossible-language-acquisition.git}
PY=()

log() { printf '\n=== %s ===\n' "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

ensure_env() {
  if [ ! -d "/root/anaconda3/envs/$ENV_NAME" ]; then
    log "creating conda env $ENV_NAME (python 3.12)"
    /root/anaconda3/bin/conda create -y -n "$ENV_NAME" python=3.12 >/dev/null 2>&1 \
      || die "conda create failed"
  fi
  PY=("/root/anaconda3/envs/$ENV_NAME/bin/python3")
  [ -x "${PY[0]}" ] || die "env python missing"
}

cmd_preflight() {
  log "preflight"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap \
             --format=csv || echo "nvidia-smi missing"
  echo "cores: $(nproc)   mem: $(free -g | awk 'NR==2{print $2}') GB   disk: $(df -h / | awk 'NR==2{print $4" free"}')"
  echo "cuda toolkit (if any): $(nvcc --version 2>/dev/null | tail -1 || echo none)"
  echo "python: $(python3 -V 2>&1)   conda: $(/root/anaconda3/bin/conda -V 2>/dev/null || echo none)"
  for url in https://pypi.org/simple/ https://download.pytorch.org/whl/ https://github.com; do
    code=$(curl -s -o /dev/null -m 8 -w '%{http_code}' "$url" || echo 000)
    echo "net $url -> $code"
  done
}

cmd_env() {
  ensure_env
  log "installing torch with sm_120 kernels (never the preinstalled 2.5)"
  # torch >= 2.7 ships sm_120; prefer the newest cu12x/cu13x wheel available.
  for spec in "torch --index-url https://download.pytorch.org/whl/cu130" \
              "torch --index-url https://download.pytorch.org/whl/cu128" \
              "torch"; do
    log "pip install $spec"
    # shellcheck disable=SC2086
    "${PY[0]}" -m pip install -q $spec && break
  done
  "${PY[0]}" -m pip install -q "transformers==4.57.6" "numpy<2.3" scikit-learn pandas tqdm requests \
    || die "dependency install failed"
  cmd_verify
}

cmd_verify() {
  ensure_env
  log "sm_120 + AMP verification"
  "${PY[0]}" - <<'PY' || die "stack verification failed"
import sys, torch
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA not available"
cap = torch.cuda.get_device_capability(0)
print("device", torch.cuda.get_device_name(0), "capability", cap, "count", torch.cuda.device_count())
if cap[0] < 12:
    print("NOTE: capability < 12.0 — this is not Blackwell; the sm_120 check is skipped")
x = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
y = (x @ x).float()
print("fp16 matmul OK:", float(y.abs().mean()))
with torch.autocast(device_type="cuda", dtype=torch.float16):
    z = torch.randn(256, 256, device="cuda") @ torch.randn(256, 256, device="cuda")
print("autocast OK:", z.dtype)
s = torch.amp.GradScaler("cuda") if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler()
print("GradScaler OK:", type(s).__module__)
PY
}

cmd_repo() {
  log "repo + pinned commit"
  if [ ! -d "$REPO_DIR/.git" ]; then
    git clone -q "$CODE_URL" "$REPO_DIR" || die "clone failed (check network/credentials)"
  fi
  cd "$REPO_DIR" || die "no repo dir"
  git fetch -q origin main
  if [ -n "$PIN_COMMIT" ]; then
    git checkout -q "$PIN_COMMIT" || die "cannot checkout $PIN_COMMIT"
  else
    git checkout -q main && git pull -q --ff-only origin main
  fi
  git log --oneline -1
  echo "results branch for this host: $RESULTS_BRANCH"
  echo -n "$RESULTS_BRANCH" > /root/burst_results_branch
}

cmd_data() {
  local from=""
  while [ $# -gt 0 ]; do case "$1" in --from) from=$2; shift ;; esac; shift; done
  log "data"
  if [ -f "$DATA_DIR/babylm_data_perturbed/.pool_version" ] && \
     [ -f "$DATA_DIR/babylm_data_perturbed/babylm_parity_word/babylm_test_affected" ]; then
    echo "data already present"
  elif [ -n "$from" ]; then
    echo "rsync from $from (15 GB, ~1 GB = perturbed pools + raw BabyLM)"
    rsync -a --info=progress2 "$from/" "$DATA_DIR/" || die "rsync failed"
  else
    log "regenerating locally via the queue's data sections (needs HF mirror access)"
    cd "$REPO_DIR" && RUN_V3=1 RUN_V3_LSTM_GPU=0 RUN_V3_LSTM_CAPMATCH=0 RUN_V3_NOPE=0 \
      RUN_V3_EXT=0 RUN_V3_STRETCH=0 RUN_V3_PROBE_SMOKE=0 QUEUE_DRY_RUN=0 \
      bash experiments_v2/kallini_repro/kallini_queue.sh >/dev/null 2>&1
  fi
  cd "$REPO_DIR" || die "no repo"
  echo -n "$(cat "$DATA_DIR/babylm_data_perturbed/.pool_version" 2>/dev/null)" ; echo
  "${PY[0]:-/root/anaconda3/bin/python3}" experiments_v2/kallini_repro/data_integrity_check.py --md5 aochildes \
    | tail -3
  echo "COPY the HF tokenizer cache from a working host if HF is blocked:"
  echo "  rsync -a <host>:/root/.cache/huggingface/hub/models--gpt2/ ~/.cache/huggingface/hub/"
}

cmd_shards() {
  local src="" out=/root/burst/shards
  while [ $# -gt 0 ]; do case "$1" in --shard-src) src=$2; shift ;; --out) out=$2; shift ;; esac; shift; done
  mkdir -p "$out"
  if [ -n "$src" ]; then
    rsync -a "$src/" "$out/" && ls -la "$out"
  else
    cd "$REPO_DIR" && "${PY[0]}" experiments_v2/kallini_repro/make_burst_shards.py --out-dir "$out" --gpus 4
  fi
  echo "shards ready in $out — run: $0 run --shards shard_p_block.tsv,shard_arch.tsv"
}

cmd_run() {
  local shards=""
  while [ $# -gt 0 ]; do case "$1" in --shards) shards=$2; shift ;; esac; shift; done
  [ -n "$shards" ] || die "--shards required"
  IFS=',' read -r -a files <<< "$shards"
  n=${#files[@]}
  for i in $(seq 0 $((n-1))); do
    local f="/root/burst/shards/${files[$i]}"
    [ -f "$f" ] || die "missing shard $f"
    log "launching runner gpu$i -> $f"
    systemd-run --unit="burst-gpu$i" --property=Restart=no \
      --property=StandardOutput=append:/root/burst/logs/runner_gpu$i.log \
      --property=StandardError=append:/root/burst/logs/runner_gpu$i.log \
      /usr/bin/bash "$REPO_DIR/experiments_v2/kallini_repro/burst_shard_runner.sh" \
        "$f" "$i" --repo "$REPO_DIR" --log-dir /root/burst/logs
  done
  echo "monitor: systemctl is-active burst-gpu0 burst-gpu1 ; tail -f /root/burst/logs/state_gpu0.tsv"
}

cmd_bench() {
  local cell=parity_word seed=0 steps=300
  while [ $# -gt 0 ]; do
    case "$1" in --cell) cell=$2; shift ;; --seed) seed=$2; shift ;; --steps) steps=$2; shift ;; esac; shift
  done
  cd "$REPO_DIR" || die "no repo"
  log "benchmark: $cell seed$seed $steps steps (compare with the 3080's 4.67 s/step)"
  /usr/bin/time -f "wall %e s" env CUDA_VISIBLE_DEVICES=0 REPRO_RESULTS=/root/burst/bench \
    "${PY[0]}" experiments_v2/kallini_repro/train_exp1.py "$cell" --seed "$seed" --steps "$steps" \
    2>&1 | grep -E '\[train\]|\[eval\]|wall|stack' | tail -8
}

main() {
  local cmd=${1:-preflight}; shift || true
  case "$cmd" in
    preflight) cmd_preflight "$@" ;;
    env) cmd_env "$@" ;;
    verify) cmd_verify "$@" ;;
    repo) cmd_repo "$@" ;;
    data) cmd_data "$@" ;;
    shards) cmd_shards "$@" ;;
    run) cmd_run "$@" ;;
    bench) cmd_bench "$@" ;;
    all)
      cmd_preflight; cmd_env; cmd_repo; cmd_data "$@"; cmd_shards
      echo; echo "READY — start with: $0 run --shards shard_p_block.tsv,shard_arch.tsv,shard_sr_panel.tsv,shard_stretch.tsv"
      ;;
    *) die "unknown command: $cmd" ;;
  esac
}
main "$@"