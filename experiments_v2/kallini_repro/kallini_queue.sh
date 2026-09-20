#!/usr/bin/env bash
# kallini_queue.sh — idempotent reproduction queue for Experiment 1 of
# Kallini et al. (2024), Shuffle + Reverse classes (9 languages x N seeds).
# Idempotent + failure-tolerant, same pattern as ralph_queue.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."   # repo root

PYTHON=${PYTHON:-/root/anaconda3/bin/python3}
[ -x "$PYTHON" ] || PYTHON=$(command -v python3)
SEEDS=${SEEDS:-"0 14 41"}
NICE_LEVEL=${NICE_LEVEL:-off}
NICE="nice -n $NICE_LEVEL"
[ "$NICE_LEVEL" = "off" ] && NICE=""

export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
# 10 GB card: reduce allocator fragmentation for the fp32-upcast GPT-2 loss path
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export KALLINI_REPO=${KALLINI_REPO:-/root/mission-impossible-language-models}
export KALLINI_DATA_PATH=${KALLINI_DATA_PATH:-/root/kallini_data}
export REPRO_RESULTS=${REPRO_RESULTS:-experiments_v2/kallini_repro/results}

LANGS="shuffle_control shuffle_nondeterministic shuffle_deterministic21 shuffle_local3 shuffle_local10 shuffle_even_odd reverse_control reverse_partial reverse_full"

# v3 class-P conditions (DESIGN_V3 §1.1) — trained with the SAME trainer/eval.
# Gated by RUN_V3 (GPU box enables); each has its own results tree.
V3_LANGS="parity_word parity_tok negtok fixed_start fixed_end bare_reverse word_shuffle"

fail=0
note() { echo "[$(date -Is)] $*"; }
mkdir -p experiments_v2/kallini_repro/results

# ---------- [1] their repo ------------------------------------------------------
if [ ! -d "$KALLINI_REPO" ]; then
  git clone -q https://github.com/jkallini/mission-impossible-language-models.git "$KALLINI_REPO" \
    || { note "KALLINI CLONE FAIL"; exit 9; }
fi
grep -q "^BABYLM_DATA_PATH = \"${KALLINI_DATA_PATH}\"" "$KALLINI_REPO/utils.py" || \
  sed -i "s|^BABYLM_DATA_PATH = .*|BABYLM_DATA_PATH = \"${KALLINI_DATA_PATH}\"|" "$KALLINI_REPO/utils.py"

# ---------- [2] BabyLM raw text from HF mirror (same source as fetch_babylm) ----
if [ ! -f "$KALLINI_DATA_PATH/babylm_data/babylm_100M/aochildes_parsed.json" ]; then
  note "fetching official BabyLM per-genre files via HF mirror (cambridge-climb/BabyLM)"
  mkdir -p "$KALLINI_DATA_PATH/babylm_data/babylm_100M" "$KALLINI_DATA_PATH/babylm_data/babylm_test"
  GENRES="aochildes bnc_spoken cbt children_stories gutenberg open_subtitles qed simple_wikipedia switchboard wikipedia"
  for g in $GENRES; do
    f="$KALLINI_DATA_PATH/babylm_data/babylm_100M/${g}.train"
    [ -f "$f" ] || curl -sL --retry 3 -o "$f" \
      "https://hf-mirror.com/datasets/cambridge-climb/BabyLM/resolve/main/clean/100M/${g}.txt" \
      || { note "HF FETCH FAIL $g"; exit 9; }
    t="$KALLINI_DATA_PATH/babylm_data/babylm_test/${g}.test"
    [ -f "$t" ] || curl -sL --retry 3 -o "$t" \
      "https://hf-mirror.com/datasets/cambridge-climb/BabyLM/resolve/main/clean/test/${g}.txt" \
      || { note "HF FETCH FAIL test $g"; exit 9; }
  done
  WORDS=$(cat "$KALLINI_DATA_PATH"/babylm_data/babylm_100M/*.train 2>/dev/null | wc -w || echo 0)
  echo "kallini corpus words: $WORDS"
  if [ "$WORDS" -lt 50000000 ]; then note "corpus too small ($WORDS)"; exit 9; fi
  note "shim-tagging train + test (per genre)"
  $PYTHON experiments_v2/kallini_repro/shim_tag.py \
    "$KALLINI_DATA_PATH"/babylm_data/babylm_100M/*.train \
    "$KALLINI_DATA_PATH"/babylm_data/babylm_test/*.test \
    >> experiments_v2/kallini_repro/data_prep.log 2>&1 \
    || { note "SHIM TAG FAIL"; exit 9; }
fi

# ---------- [3] perturbed datasets via THEIR perturb.py -------------------------
# Idempotent per language: regenerate only the languages whose per-genre files
# are missing. A partial pass (crash, OOM, disk) must not look complete: require
# all 10 BabyLM genres before a language counts as generated.
BABYLM_GENRES="aochildes bnc_spoken cbt children_stories gutenberg open_subtitles qed simple_wikipedia switchboard wikipedia"
perturb_missing=""
for l in $LANGS; do
  for g in $BABYLM_GENRES; do
    # upstream perturb.py writes {genre}.train / {genre}_affected.test
    [ -f "$KALLINI_DATA_PATH/babylm_data_perturbed/babylm_$l/babylm_100M/$g.train" ] \
      && [ -f "$KALLINI_DATA_PATH/babylm_data_perturbed/babylm_$l/babylm_test_affected/${g}_affected.test" ] \
      || { perturb_missing="$perturb_missing $l"; break; }
  done
done
if [ -n "$perturb_missing" ]; then
  note "perturbing train (100M) + test splits with their perturb.py:$perturb_missing"
  printf '%s\n' $perturb_missing | xargs -P 3 -I{} bash -c '
    NICE_LEVEL="'$NICE_LEVEL'"; NICE="nice -n $NICE_LEVEL"; [ "$NICE_LEVEL" = off ] && NICE=""
    # their perturb.py does sys.path.append("..") relative to the CWD. Run it
    # from <repo>/data (as their own data/perturb.sh does) so that ".." resolves
    # to the repo root, where utils.py lives. Running it from the repo root
    # raises ModuleNotFoundError: No module named utils.
    cd '"$KALLINI_REPO"'/data || exit 9
    '"$PYTHON"' perturb.py {} 100M >> '"$PWD"'/experiments_v2/kallini_repro/data_prep.log 2>&1 || true
    '"$PYTHON"' perturb.py {} test  >> '"$PWD"'/experiments_v2/kallini_repro/data_prep.log 2>&1 || true
  ' || note "WARN some perturb jobs failed (see data_prep.log)"
fi

# ---------- [3b] v3 class-P datasets (DESIGN_V3 §1.1, Kallini token-ID format) --
# Gate on the per-genre layout emitted since c77fe29. The old gate tested
# all.train, which the pre-c77fe29 writer produced as a single overwritten file
# (only the last genre survived) -> it never triggered a regeneration.
#
# 2026-09-20: the gate ALSO checks a pool-version marker, because the sentence
# filter changed semantics (v1: filter on the perturbed token count -> every
# markered condition kept a different sentence set; v2: filter on the BASE
# sentence, Kallini's filter_shuffle semantics -> all class-P conditions share
# one sentence set). Existence checks cannot detect a semantics change, and a
# stale pool would silently train the treatment arm on a different sentence set
# than its control (audit 2026-09-20, P0 / B5).
V3_LANGS_ALL="parity_word parity_tok negtok fixed_start fixed_end bare_reverse word_shuffle not_random"
POOL_VERSION="pool-v2-base-filter"
POOL_VERSION_FILE="$KALLINI_DATA_PATH/babylm_data_perturbed/.pool_version"
if [ "${RUN_V3:-0}" = "1" ]; then
  v3_missing=0
  [ "$(cat "$POOL_VERSION_FILE" 2>/dev/null)" = "$POOL_VERSION" ] || v3_missing=1
  for c in $V3_LANGS_ALL; do
    for g in $BABYLM_GENRES; do
      # v3_conditions.write_condition names files after the tagged json stem:
      # {genre}_parsed.train / {genre}_parsed_affected.test (NOT upstream's
      # {genre}.train) — checking the wrong name re-ran 2h of finished work
      # on every loop iteration.
      [ -f "$KALLINI_DATA_PATH/babylm_data_perturbed/babylm_$c/babylm_100M/${g}_parsed.train" ] \
        && [ -f "$KALLINI_DATA_PATH/babylm_data_perturbed/babylm_$c/babylm_test_affected/${g}_parsed_affected.test" ] \
        || { v3_missing=1; break; }
    done
    [ "$v3_missing" = "1" ] && break
  done
  if [ "$v3_missing" = "1" ]; then
    note "generating v3 class-P datasets (pool $POOL_VERSION)"
    # single source of truth for the generator (also runnable standalone on any
    # host that has the tagged shim JSONs)
    if FORCE_REGEN=1 $PYTHON experiments_v2/design_v3/regenerate_conditions.py --force \
        >> experiments_v2/kallini_repro/data_prep.log 2>&1; then
      printf '%s\n' "$POOL_VERSION" > "$POOL_VERSION_FILE"
      note "v3 datasets regenerated (pool $POOL_VERSION)"
    else
      note "V3 PERTURB FAIL"; exit 9
    fi
  fi
  # The loader globs *.train and *_affected.test: a stale pre-c77fe29 single-file
  # dump (all.train / all_affected.test) would be loaded on top of the per-genre
  # files. Quarantine such files outside babylm_data_perturbed (reversible).
  stale_dir="$KALLINI_DATA_PATH/_stale_all_splits"
  stale_hits=$(find "$KALLINI_DATA_PATH/babylm_data_perturbed" -maxdepth 3 \
    \( -name 'all.train' -o -name 'all_affected.test' \) -print 2>/dev/null)
  if [ -n "$stale_hits" ]; then
    mkdir -p "$stale_dir"
    printf '%s\n' "$stale_hits" | while IFS= read -r f; do mv -f "$f" "$stale_dir/"; done
    note "quarantined $(printf '%s\n' "$stale_hits" | wc -l) stale all.* split files -> $stale_dir"
  fi
fi

# ---------- [4] training queue --------------------------------------------------
run() {
  local lang=$1 seed=$2
  if [ "${QUEUE_DRY_RUN:-0}" = "1" ]; then note "[dry] would train $lang/seed$seed"; return 0; fi
  if $NICE $PYTHON experiments_v2/kallini_repro/train_exp1.py "$lang" --seed "$seed" --skip-if-done \
      >> experiments_v2/kallini_repro/queue.log 2>&1; then
    note "OK   $lang/seed$seed"
  else
    note "FAIL $lang/seed$seed"
    fail=$((fail+1))
  fi
}

run_steps() {  # lang seed steps  (H7 budget ladder; out_dir gets a steps<N>_ tag)
  local lang=$1 seed=$2 steps=$3
  if [ "${QUEUE_DRY_RUN:-0}" = "1" ]; then note "[dry] would train ${steps}step $lang/seed$seed"; return 0; fi
  if $NICE $PYTHON experiments_v2/kallini_repro/train_exp1.py "$lang" --seed "$seed" --steps "$steps" --skip-if-done \
      >> experiments_v2/kallini_repro/queue.log 2>&1; then
    note "OK   ${steps}step $lang/seed$seed"
  else
    note "FAIL ${steps}step $lang/seed$seed"
    fail=$((fail+1))
  fi
}

for seed in $SEEDS; do
  for lang in $LANGS; do
    run "$lang" "$seed"
  done
done

# ---------- [4b] v3 class-P training queue (DESIGN_V3 priority ladder) ---------
if [ "${RUN_V3:-0}" = "1" ]; then
  # P0: parity_word + fixed_start (the paper's central contrast)
  # P1: parity_tok + negtok
  # P2: H7 2x arm (natural + parity_word at 6000 steps, seed 0)
  # ladder for H7: {300,1000,2000,4000,6000} via STEPS env in train_exp1
  for seed in 0 14 41; do
    for lang in parity_word fixed_start parity_tok negtok not_random; do
      run "$lang" "$seed"
    done
  done
  if [ "${RUN_V3_H7:-1}" = "1" ]; then
    for lang in parity_word fixed_start shuffle_control; do
      run_steps "$lang" 0 6000
    done
  fi
fi

# ---------- [4c] GPU LSTM arm: equal-token-budget architecture axis -------------
# Audit B2 (2026-09-20). The cpu2 LSTM arm runs at 1/160 of the GPT-2 token
# budget, so it licenses no equal-budget architecture claim (F8). This arm runs
# the SAME trainer on the GPU with the GPT-2 protocol's shapes — seq 1024,
# effective batch 128, 3000 steps = 3000x128x1024 = 3.93e8 tokens, exactly the
# GPT-2 arm's budget — so F4 (H12) becomes a budget-matched contrast.
# Sub-protocol: seq 1024 / micro 8 x accum 16 / 3000 steps / 10k-sentence eval;
# the optimizer settings stay the frozen per-architecture v2 LSTM regime
# (AdamW, peak LR 1e-3, 10% warmup, dropout 0.3, clip 5.0), which is the
# documented per-architecture deviation (EXPDESIGN_V3 §2.2).
run_lstm_gpu() {  # condition seed
  local c=$1 s=$2
  if [ "${QUEUE_DRY_RUN:-0}" = "1" ]; then note "[dry] would train lstm_gpu $c/seed$s"; return 0; fi
  if $NICE env LSTM_DEVICE=cuda LSTM_RESULTS=experiments_v2/kallini_repro/results_lstm_gpu \
        LSTM_SEQ_LEN=1024 LSTM_EFF_BATCH=128 LSTM_MICRO_BATCH=8 LSTM_STEPS=3000 \
        LSTM_LR=1e-3 LSTM_EVAL_N=10000 LSTM_SAVE_CKPT=0 LSTM_PACK_VERSION=v2 \
        $PYTHON experiments_v2/kallini_repro/train_exp1_lstm.py "$c" --seed "$s" --skip-if-done \
        >> experiments_v2/kallini_repro/results_lstm_gpu/queue.log 2>&1; then
    note "OK   lstm_gpu $c/seed$s"
  else
    note "FAIL lstm_gpu $c/seed$s"
    fail=$((fail+1))
  fi
}

if [ "${RUN_V3:-0}" = "1" ] && [ "${RUN_V3_LSTM_GPU:-1}" = "1" ]; then
  mkdir -p experiments_v2/kallini_repro/results_lstm_gpu
  LSTM_GPU_CONDS="${LSTM_GPU_CONDS:-shuffle_control reverse_full parity_word not_random}"
  pending_lstm=$(find experiments_v2/kallini_repro/results_lstm_gpu -name lstm_result.json 2>/dev/null | wc -l)
  expected_lstm=$(( $(echo $LSTM_GPU_CONDS | wc -w) * 3 ))
  if [ "$pending_lstm" -lt "$expected_lstm" ] && [ "${QUEUE_DRY_RUN:-0}" != "1" ]; then
    # Smoke first: a broken CUDA path must fail here (~1 min) rather than after
    # the first 2 h cell. Writes into a quarantine tree, never the real arm.
    note "lstm_gpu smoke (1 step, quarantined tree)"
    if $NICE env LSTM_DEVICE=cuda \
          LSTM_RESULTS=experiments_v2/kallini_repro/results_smoke/_quarantine_lstm_gpu \
          LSTM_SEQ_LEN=1024 LSTM_EFF_BATCH=128 LSTM_MICRO_BATCH=8 \
          LSTM_SAVE_CKPT=0 LSTM_PACK_VERSION=v2smoke \
          $PYTHON experiments_v2/kallini_repro/train_exp1_lstm.py shuffle_control --seed 0 --steps 1 \
          >> experiments_v2/kallini_repro/results_lstm_gpu/queue.log 2>&1; then
      note "lstm_gpu smoke OK"
    else
      note "LSTM GPU SMOKE FAILED -> arm skipped this pass"
      fail=$((fail+1))
    fi
  fi
  if [ "$pending_lstm" -lt "$expected_lstm" ]; then
    for c in $LSTM_GPU_CONDS; do
      for s in 0 14 41; do
        run_lstm_gpu "$c" "$s"
      done
    done
  fi
  if [ "${QUEUE_DRY_RUN:-0}" != "1" ]; then
    RESULTS_DIR=experiments_v2/kallini_repro/results_lstm_gpu \
      RESULTS_BRANCH=v2-results-lstm-gpu RESULTS_KIND=lstm_result.json \
      bash experiments_v2/kallini_repro/publish_results.sh || note "WARN lstm_gpu publish failed"
  fi
fi

# ---------- [4d] rigor extension tier (2026-09-20 design audit) -- ---------------
# Adds every cell that a confirmatory family needs to reach the pre-registered
# sample size, plus the entropy-matched marker control (audit B1/B4):
#   * seeds 53/96 for the F4 reference conditions (shuffle_control, reverse_full)
#     -> F4a/F4b/F4c reach n=5 instead of n=3 (STATS_PLAN_V3 §2: headline claims
#     require n>=5 because exact rank tests cannot reach p<.05 at n=3)
#   * seeds 53/96 for parity_word / fixed_start -> F1 (H10, the paper's central
#     contrast) reaches n=5
#   * seeds 53/96 for parity_tok / negtok -> F2/F3 reach n=5
#   * fixed_end at seeds 0/14/41 -> F2's second control reaches n=3
#   * not_random at seeds 0/14/41 -> the entropy-matched control (B1)
#   * H7 (6000 steps) at seeds 14/41 for shuffle_control / parity_word -> F5 is
#     no longer blocked at n=1
EXT_SEEDS="${EXT_SEEDS:-53 96}"
if [ "${RUN_V3:-0}" = "1" ] && [ "${RUN_V3_EXT:-1}" = "1" ]; then
  for seed in $EXT_SEEDS; do
    for lang in shuffle_control reverse_full parity_word fixed_start parity_tok negtok; do
      run "$lang" "$seed"
    done
  done
  for seed in 0 14 41; do
    run fixed_end "$seed"
  done
  if [ "${RUN_V3_H7:-1}" = "1" ]; then
    for seed in 14 41; do
      for lang in shuffle_control parity_word; do
        run_steps "$lang" "$seed" 6000
      done
    done
  fi
fi

# ---------- [4e] BabyLM probe smoke (code-path check only, no conclusions) -----
# Runs the probe suite on the first available final/ checkpoint with 10 pairs and
# writes into a quarantine tree. It is a code-path check: probe numbers from
# shuffle-class checkpoints are not results (they must come from the P class).
if [ "${RUN_V3:-0}" = "1" ] && [ "${RUN_V3_PROBE_SMOKE:-1}" = "1" ]; then
  probe_ckpt=$(ls -d experiments_v2/kallini_repro/results/babylm_*_100M/seed0/final 2>/dev/null | head -1)
  if [ -n "$probe_ckpt" ]; then
    mkdir -p experiments_v2/kallini_repro/results_smoke/_quarantine_probe
    if $NICE $PYTHON experiments_v2/probes/probes_babylm.py \
         --model-dir "$probe_ckpt" --pairs 10 --smoke \
         --out experiments_v2/kallini_repro/results_smoke/_quarantine_probe/probe_smoke.json \
         >> experiments_v2/kallini_repro/results_lstm_gpu/queue.log 2>&1; then
      note "probe smoke OK ($probe_ckpt)"
    else
      note "WARN probe smoke failed (analysis-side, does not block the grid)"
    fi
  fi
fi

# ---------- [5] aggregate --------------------------------------------------------
$PYTHON experiments_v2/kallini_repro/aggregate_exp1.py >> experiments_v2/kallini_repro/queue.log 2>&1 \
  || note "WARN aggregation failed"

n_done=$(find experiments_v2/kallini_repro/results -name exp1_result.json 2>/dev/null | wc -l)
note "kallini pass done: $n_done runs complete, $fail failures"
if [ "$fail" -gt 0 ]; then exit 1; fi
if [ "${QUEUE_DRY_RUN:-0}" = "1" ]; then
  note "[dry] pass validated; not writing the completion marker"
  exit 0
fi
touch experiments_v2/kallini_repro/results/ALL_KALLINI_DONE
exit 0
