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

# Row-count equality gate (2026-09-20 evening, prereg §10c): the existence
# gate cannot see a truncated genre file — audit A0's exact failure mode
# (parity_word/simple_wikipedia was 37% short and passed every check).
#
# The invariant is CROSS-CONDITION per genre, i.e. "pool identity": for a fixed
# genre, all class-P conditions must have the same line count (they are generated
# from one shared sentence set). Genres legitimately differ from each other, so a
# within-condition comparison would be wrong — the first dry-run of this gate
# caught exactly that mistake before it could trigger a spurious full
# regeneration. Runs before any training starts (a pass begin has no training in
# flight, so the deterministic regeneration it may trigger is safe).
rowcount_gate() {
  local c g f n ref first hits=""
  for g in $BABYLM_GENRES; do
    ref=""; first=""
    for c in $V3_LANGS_ALL; do
      f="$KALLINI_DATA_PATH/babylm_data_perturbed/babylm_$c/babylm_100M/${g}_parsed.train"
      [ -f "$f" ] || { hits="$hits missing:${c}:${g}"; continue; }
      n=$(grep -c '' "$f")
      if [ -z "$ref" ]; then ref=$n; first=$c
      elif [ "$n" != "$ref" ]; then hits="$hits ${g}:${c}=${n}_vs_${ref}(${first})"; fi
    done
    ref=""; first=""
    for c in $V3_LANGS_ALL; do
      f="$KALLINI_DATA_PATH/babylm_data_perturbed/babylm_$c/babylm_test_affected/${g}_parsed_affected.test"
      [ -f "$f" ] || { hits="$hits testmissing:${c}:${g}"; continue; }
      n=$(grep -c '' "$f")
      if [ -z "$ref" ]; then ref=$n; first=$c
      elif [ "$n" != "$ref" ]; then hits="$hits test:${g}:${c}=${n}_vs_${ref}(${first})"; fi
    done
  done
  echo "$hits"
}

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
    # 2026-09-20 incident: a loop restart pulled this new gate while a manual
    # regeneration was already running, so a SECOND generator started writing the
    # same files (and gpu2 hit a global OOM that killed the training cell). Never
    # start a duplicate: wait for the running generator, then re-check the marker.
    while pgrep -f 'regenerate_conditions[.]py' >/dev/null 2>&1; do
      note "waiting for a running dataset regeneration to finish (no duplicate generator)"
      sleep 60
    done
    [ "$(cat "$POOL_VERSION_FILE" 2>/dev/null)" = "$POOL_VERSION" ] && v3_missing=0
  fi
  if [ "$v3_missing" = "1" ]; then
    if [ "${QUEUE_DRY_RUN:-0}" = "1" ]; then
      note "[dry] v3 class-P datasets incomplete -> would regenerate (skipped in dry-run)"
    else
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
  fi
  # Row-count equality gate (audit A0 prevention, §10c): a mismatch means a
  # genre file is truncated or from a different generation pass. The pools are
  # deterministic, so one regeneration attempt is the correct repair; a
  # persistent mismatch is a hard failure (the training arm must not start on
  # an unequal pool). Dry-run mode only reports (no writes).
  gate_hits=$(rowcount_gate)
  if [ -n "$gate_hits" ] && [ "${QUEUE_DRY_RUN:-0}" = "1" ]; then
    note "[dry] DATA GATE row-count mismatch would be repaired by regeneration:$gate_hits"
    gate_hits=""
  fi
  if [ -n "$gate_hits" ]; then
    note "DATA GATE row-count mismatch:$gate_hits -> one deterministic regen pass"
    if FORCE_REGEN=1 $PYTHON experiments_v2/design_v3/regenerate_conditions.py --force \
        >> experiments_v2/kallini_repro/data_prep.log 2>&1; then
      printf '%s\n' "$POOL_VERSION" > "$POOL_VERSION_FILE"
    else
      note "V3 REGEN FAIL (row-count gate)"; exit 9
    fi
    gate_hits=$(rowcount_gate)
    if [ -n "$gate_hits" ]; then
      note "DATA GATE STILL MISMATCHED AFTER REGEN:$gate_hits"; exit 9
    fi
    note "DATA GATE row-count OK after regen"
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
  if [ "${QUEUE_DRY_RUN:-0}" = "1" ]; then note "[dry] would train $lang/seed$seed${RUN_EXTRA_ENV:+ [$RUN_EXTRA_ENV]}"; return 0; fi
  # env ${RUN_EXTRA_ENV:-} lets a tier arm its cells (e.g. LADDER_PROBE=1 for
  # the class-P blocks) without duplicating the runner; empty = unchanged.
  if env ${RUN_EXTRA_ENV:-} $NICE $PYTHON experiments_v2/kallini_repro/train_exp1.py "$lang" --seed "$seed" --skip-if-done \
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

# ---------- [4b] v3 class-P training queue (DESIGN_V3 priority ladder) ---------
if [ "${RUN_V3:-0}" = "1" ]; then
  # P0: parity_word + fixed_start (the paper's central contrast)
  # P1: parity_tok + negtok
  # P2: H7 2x arm (natural + parity_word at 6000 steps, seed 0)
  # ladder for H7: {300,1000,2000,4000,6000} via STEPS env in train_exp1
  # 2026-09-20 evening (§10c-3): every class-P cell carries the in-process
  # ladder probe (P1 branch-matched minimal pairs at each eval checkpoint),
  # so rule-acquisition dynamics come out of the same cells.
  RUN_EXTRA_ENV="LADDER_PROBE=1"
  for seed in 0 14 41; do
    for lang in parity_word fixed_start parity_tok negtok not_random; do
      run "$lang" "$seed"
    done
  done
  RUN_EXTRA_ENV=""
  if [ "${RUN_V3_H7:-1}" = "1" ]; then
    for lang in parity_word fixed_start shuffle_control; do
      run_steps "$lang" 0 6000
    done
  fi
fi

# ---------- [4b2] NoPE position-ablation arm (§10c-4, moved up per §10c-13 E) ----
# 2026-09-24 (§10c-13 approval): moved BEFORE [4c]/[4c2] so the causal position-
# channel ablation (T1) lands inside the 12-day window. Block body unchanged.
if [ "${RUN_V3:-0}" = "1" ] && [ "${RUN_V3_NOPE:-1}" = "1" ]; then
  NOPE_DIR=experiments_v2/kallini_repro/results_nope
  mkdir -p "$NOPE_DIR"
  NOPE_CONDS="${NOPE_CONDS:-parity_word shuffle_control}"
  pending_nope=$(find "$NOPE_DIR" -name exp1_result.json 2>/dev/null | wc -l)
  expected_nope=$(( $(echo $NOPE_CONDS | wc -w) * 3 ))
  if [ "$pending_nope" -lt "$expected_nope" ]; then
    for c in $NOPE_CONDS; do
      for s in 0 14 41; do
        if [ "${QUEUE_DRY_RUN:-0}" = "1" ]; then note "[dry] would train nope $c/seed$s"; continue; fi
        if $NICE env REPRO_RESULTS=$NOPE_DIR GPT2_NOPE=1 LADDER_PROBE=1 \
            $PYTHON experiments_v2/kallini_repro/train_exp1.py "$c" --seed "$s" --skip-if-done \
            >> experiments_v2/kallini_repro/results_nope/queue.log 2>&1; then
          note "OK   nope $c/seed$s"
        else
          note "FAIL nope $c/seed$s"
          fail=$((fail+1))
        fi
      done
    done
  fi
  if [ "${QUEUE_DRY_RUN:-0}" != "1" ]; then
    RESULTS_DIR=$NOPE_DIR RESULTS_BRANCH=v2-results-nope RESULTS_KIND=exp1_result.json \
      bash experiments_v2/kallini_repro/publish_results.sh || note "WARN nope publish failed"
  fi
fi

# ---------- [4b3] NoPE extension (§10c-13 A1; T1 smoke gate) ---------------------
# fixed_start + not_random under NoPE. Runs only after the T1 smoke gate: the
# first base-NoPE content-penalty measurement <= 0.15 nats writes .t1_smoke_ok
# (analysis side). Otherwise the arm is skipped and its 23.7 GPU-h re-purposed.
if [ "${RUN_V3:-0}" = "1" ] && [ "${RUN_V3_NOPE_EXT:-1}" = "1" ]; then
  NOPE_DIR=${NOPE_DIR:-experiments_v2/kallini_repro/results_nope}
  mkdir -p "$NOPE_DIR"
  if [ -f "$NOPE_DIR/.t1_smoke_ok" ]; then
    for c in fixed_start not_random; do
      for s in 0 14 41; do
        if [ "${QUEUE_DRY_RUN:-0}" = "1" ]; then note "[dry] would train nope-ext $c/seed$s"; continue; fi
        if $NICE env REPRO_RESULTS=$NOPE_DIR GPT2_NOPE=1 LADDER_PROBE=1 \
            $PYTHON experiments_v2/kallini_repro/train_exp1.py "$c" --seed "$s" --skip-if-done \
            >> experiments_v2/kallini_repro/results_nope/queue.log 2>&1; then
          note "OK   nope-ext $c/seed$s"
        else
          note "FAIL nope-ext $c/seed$s"
          fail=$((fail+1))
        fi
      done
    done
  else
    note "nope-ext waiting for .t1_smoke_ok (T1 smoke gate, §10c-13 A1)"
  fi
fi

# ---------- [4b4] CALD positive-evidence family (§10c-13 A2) ---------------------
# cald_local / cald_long / cald_shuf. Pilot cells (calibration input) write to
# results_cald_pilot/ and are DISCARDED after calibration (prereg A2); the
# confirmatory 9 cells run only after .frozen_cald exists (written by the
# §10c-13a amendment step after the pilot readout; mirrors capmatch .frozen_lr).
if [ "${RUN_V3:-0}" = "1" ] && [ "${RUN_V3_CALD:-1}" = "1" ]; then
  CALD_DIR=experiments_v2/kallini_repro/results_cald
  CALD_PILOT_DIR=experiments_v2/kallini_repro/results_cald_pilot
  mkdir -p "$CALD_DIR" "$CALD_PILOT_DIR"
  cald_data_root="$KALLINI_DATA_PATH/babylm_data_perturbed"
  for c in cald_long cald_shuf; do
    if [ ! -d "$cald_data_root/babylm_${c}/babylm_100M" ]; then
      note "cald data for $c missing -> run design_v3/make_cald_conditions.py; pilot skipped this pass"
      continue
    fi
    if [ "${QUEUE_DRY_RUN:-0}" = "1" ]; then note "[dry] would train cald-pilot $c/seed0"; continue; fi
    if $NICE env REPRO_RESULTS=$CALD_PILOT_DIR REPRO_DATA_SUBDIR="babylm_${c}" \
        $PYTHON experiments_v2/kallini_repro/train_exp1.py "$c" --seed 0 --skip-if-done \
        >> experiments_v2/kallini_repro/results_cald_pilot/queue.log 2>&1; then
      note "OK   cald-pilot $c/seed0"
    else
      note "FAIL cald-pilot $c/seed0"
      fail=$((fail+1))
    fi
  done
  if [ -f "$CALD_DIR/.frozen_cald" ]; then
    for c in cald_local cald_long cald_shuf; do
      for s in 0 14 41; do
        if [ "${QUEUE_DRY_RUN:-0}" = "1" ]; then note "[dry] would train cald $c/seed$s"; continue; fi
        if $NICE env REPRO_RESULTS=$CALD_DIR REPRO_DATA_SUBDIR="babylm_${c}" \
            $PYTHON experiments_v2/kallini_repro/train_exp1.py "$c" --seed "$s" --skip-if-done \
            >> experiments_v2/kallini_repro/results_cald/queue.log 2>&1; then
          note "OK   cald $c/seed$s"
        else
          note "FAIL cald $c/seed$s"
          fail=$((fail+1))
        fi
      done
    done
  else
    note "cald confirmatory waiting for .frozen_cald (§10c-13a amendment after pilot)"
  fi
  if [ "${QUEUE_DRY_RUN:-0}" != "1" ]; then
    RESULTS_DIR=$CALD_DIR RESULTS_BRANCH=v2-results-cald RESULTS_KIND=exp1_result.json \
      bash experiments_v2/kallini_repro/publish_results.sh || note "WARN cald publish failed"
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

# ---------- [4c2] capacity-matched LSTM arm (§10c-2, registered 2026-09-20) -----
# The §[4c] arm is equal-TOKEN-budget but not equal-capacity (~40M vs 124M), so
# the confirmatory architecture family F4 is redefined onto THIS arm:
#   EMB = HIDDEN = 1620 with the tied output head -> 50257x1620 + 16x1620^2
#   = 123.4M params (99.5% of GPT-2-small 124M).
# Same protocol shapes as §[4c] (seq 1024 / eff batch 128 / 3000 steps = 3.93e8
# tokens). Registration status: pre-data amend (0 cells at registration).
# The LR is frozen by a cheap probe on the natural condition ONLY (REDTEAM #4(i):
# 3 LRs x 1 seed x 600 steps, quarantined tree, no inferential claim). The
# frozen value is cached in .frozen_lr so a re-armed pass skips the probe.
CAPMATCH_RESULTS=experiments_v2/kallini_repro/results_lstm_gpu_capmatch
run_lstm_capmatch() {  # condition seed
  local c=$1 s=$2
  if [ "${QUEUE_DRY_RUN:-0}" = "1" ]; then note "[dry] would train lstm_capmatch $c/seed$s (LR $CAPMATCH_LR)"; return 0; fi
  # P1-1 guard (2026-09-24): never train this confirmatory arm with an unfrozen LR.
  # §10c-2 requires the probe to freeze it; the old code silently fell back to 1e-3.
  case "${CAPMATCH_LR:-}" in ""|FAIL)
    note "SKIP lstm_capmatch $c/seed$s (no frozen LR; probe incomplete -> arm deferred)"
    return 0
    ;;
  esac
  if $NICE env LSTM_DEVICE=cuda LSTM_RESULTS=$CAPMATCH_DIR \
        LSTM_SEQ_LEN=1024 LSTM_EFF_BATCH=128 LSTM_MICRO_BATCH=8 LSTM_STEPS=3000 \
        LSTM_LR="$CAPMATCH_LR" LSTM_EVAL_N=10000 LSTM_SAVE_CKPT=0 LSTM_PACK_VERSION=v2 \
        LSTM_ARCH_TAG=lstm_capmatch124 \
        LSTM_BUDGET_NOTE="capacity-matched arm: EMB=HIDDEN=1620 (tied head) = 123.4M params vs GPT-2-small 124M; token budget identical to the GPT-2 arm (3000x128x1024)" \
        $PYTHON experiments_v2/kallini_repro/train_exp1_lstm.py "$c" --seed "$s" --skip-if-done \
        >> experiments_v2/kallini_repro/results_lstm_gpu_capmatch/queue.log 2>&1; then
    note "OK   lstm_capmatch $c/seed$s"
  else
    note "FAIL lstm_capmatch $c/seed$s"
    fail=$((fail+1))
  fi
}
if [ "${RUN_V3:-0}" = "1" ] && [ "${RUN_V3_LSTM_CAPMATCH:-1}" = "1" ]; then
  CAPMATCH_DIR=experiments_v2/kallini_repro/results_lstm_gpu_capmatch
  mkdir -p "$CAPMATCH_DIR"
  pending_cap=$(find "$CAPMATCH_DIR" -name lstm_result.json 2>/dev/null | wc -l)
  CAPMATCH_CONDS="${CAPMATCH_CONDS:-shuffle_control reverse_full parity_word}"
  expected_cap=$(( $(echo $CAPMATCH_CONDS | wc -w) * 3 ))
  if [ "$pending_cap" -lt "$expected_cap" ]; then
    # --- LR probe (no inference; quarantined tree; skip-if-done per tree) ---
    CAPMATCH_LR=$(cat "$CAPMATCH_DIR/.frozen_lr" 2>/dev/null || true)
    if [ -z "$CAPMATCH_LR" ]; then
      for lr in 5e-4 1e-3 2e-3; do
        lr_dir=experiments_v2/kallini_repro/results_smoke/_quarantine_lstm_capmatch_lr/lr$lr
        mkdir -p "$lr_dir"
        if [ "${QUEUE_DRY_RUN:-0}" = "1" ]; then note "[dry] would run capmatch LR probe lr=$lr"; continue; fi
        if $NICE env LSTM_DEVICE=cuda LSTM_RESULTS="$lr_dir" \
              LSTM_SEQ_LEN=1024 LSTM_EFF_BATCH=128 LSTM_MICRO_BATCH=8 LSTM_STEPS=600 \
              LSTM_LR="$lr" LSTM_EVAL_N=2000 LSTM_SAVE_CKPT=0 LSTM_PACK_VERSION=v2smoke \
              LSTM_ARCH_TAG=lstm_capmatch124_lrprobe \
              $PYTHON experiments_v2/kallini_repro/train_exp1_lstm.py shuffle_control --seed 0 --skip-if-done \
              >> experiments_v2/kallini_repro/results_smoke/lr_probe.log 2>&1; then
          note "capmatch LR probe OK ($lr)"
        else
          note "WARN capmatch LR probe failed ($lr)"
          fail=$((fail+1))
        fi
      done
      # P1-1 fix (2026-09-24): the trainer writes
      #   <LSTM_RESULTS>/babylm_{cond}_{TRAIN_SET}/seed{seed}   (train_exp1_lstm.py:469)
      # — there is NO "steps" path component — and a 600-step run evaluates at
      # checkpoints [100, 300, 500] (train_exp1.eval_checkpoints_for), so the old
      # glob `lr*/.../steps600_seed0` and the key `eval_gmean["600"]` could never match:
      # best stayed None and the hardcoded "1e-3" was frozen instead of the probe's best
      # LR (prereg §10c-2 / REDTEAM #4(i)). The old label `d.parent.name.split("lr")[1]`
      # would also have raised IndexError had the glob ever matched.
      CAPMATCH_LR=$($PYTHON - <<'PYEOF'
import json, pathlib, sys
base = pathlib.Path("experiments_v2/kallini_repro/results_smoke/_quarantine_lstm_capmatch_lr")


def diag(msg):
    print(msg, file=sys.stderr)


best, best_v = None, None
for d in sorted(base.glob("lr*/babylm_shuffle_control_100M/seed0")):
    name = d.parent.parent.name            # lr<value>
    label = name[2:] if name.startswith("lr") else name
    r = d / "lstm_result.json"
    if not r.exists():
        diag(f"[probe] {name}: no lstm_result.json")
        continue
    try:
        res = json.loads(r.read_text())
    except Exception as exc:
        diag(f"[probe] {name}: unreadable JSON ({exc})")
        continue
    if res.get("max_steps") != 600:
        diag(f"[probe] {name}: max_steps={res.get('max_steps')} != 600 -> skip")
        continue
    trace = res.get("eval_gmean") or {}
    keys = sorted(int(k) for k in trace if str(k).isdigit())
    if not keys:
        diag(f"[probe] {name}: empty eval_gmean")
        continue
    last = keys[-1]
    v = trace[str(last)]
    diag(f"[probe] lr={label}: eval_gmean[{last}]={v} (max_steps=600, ckpts={keys})")
    if v is not None and (best_v is None or v < best_v):
        best, best_v = label, v
print(best or "FAIL")
PYEOF
)
      if [ "$CAPMATCH_LR" = "FAIL" ]; then
        note "WARN capmatch LR probe produced NO usable result -> arm SKIPPED this pass; .frozen_lr NOT written (no silent 1e-3 fallback; prereg 10c-2 requires the probe)"
      elif [ "${QUEUE_DRY_RUN:-0}" = "1" ]; then
        note "[dry] would freeze capmatch LR to $CAPMATCH_LR (not written in dry-run)"
      else
        printf '%s\n' "$CAPMATCH_LR" > "$CAPMATCH_DIR/.frozen_lr"
        note "capmatch LR frozen: $CAPMATCH_LR"
      fi
    fi
    # smoke first (fail fast before the first ~4 h cell), then the cells. Both
    # live in one guard; run_lstm_capmatch prints a [dry] line per cell, so a
    # dry-run still enumerates the arm through the guard-free call below.
    if [ "${QUEUE_DRY_RUN:-0}" != "1" ] && [ "$pending_cap" -lt "$expected_cap" ] \
       && ! ls "$CAPMATCH_DIR"/babylm_* >/dev/null 2>&1; then
      note "lstm_capmatch smoke (1 step, quarantined tree)"
      if $NICE env LSTM_DEVICE=cuda \
            LSTM_RESULTS=experiments_v2/kallini_repro/results_smoke/_quarantine_lstm_capmatch \
            LSTM_SEQ_LEN=1024 LSTM_EFF_BATCH=128 LSTM_MICRO_BATCH=8 LSTM_STEPS=1 \
            LSTM_EMB=1620 LSTM_HIDDEN=1620 LSTM_SAVE_CKPT=0 LSTM_PACK_VERSION=v2smoke \
            LSTM_ARCH_TAG=lstm_capmatch124 \
            $PYTHON experiments_v2/kallini_repro/train_exp1_lstm.py shuffle_control --seed 0 \
            >> experiments_v2/kallini_repro/results_lstm_gpu_capmatch/queue.log 2>&1; then
        note "lstm_capmatch smoke OK"
      else
        note "LSTM CAPMATCH SMOKE FAILED -> arm skipped this pass"
        fail=$((fail+1))
      fi
    fi
    for c in $CAPMATCH_CONDS; do
      for s in 0 14 41; do
        run_lstm_capmatch "$c" "$s"
      done
    done
  fi
  if [ "${QUEUE_DRY_RUN:-0}" != "1" ]; then
    RESULTS_DIR=$CAPMATCH_DIR RESULTS_BRANCH=v2-results-lstm-gpu-capmatch \
      RESULTS_KIND=lstm_result.json \
      bash experiments_v2/kallini_repro/publish_results.sh || note "WARN capmatch publish failed"
  fi
fi

# ---------- [4c3] NoPE position-ablation arm — MOVED to [4b2] (§10c-13, 2026-09-24)
# The block body now runs directly after the class-P queue so T1 lands inside the
# 12-day window; the nope extension lives in [4b3], gated by the T1 smoke flag.

# ---------- [4f] Kallini S/R replication panel (T0) ----------------------------
# Runs AFTER the paper-critical blocks (2026-09-20 ordering decision): the class-P
# grid + H7 carry the paper's central contrast (H10) and the probes depend on its
# checkpoints, so they go first; the 27-cell replication panel follows. Total
# compute is unchanged.
for seed in $SEEDS; do
  for lang in $LANGS; do
    run "$lang" "$seed"
  done
done

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
  RUN_EXTRA_ENV="LADDER_PROBE=1"
  for seed in $EXT_SEEDS; do
    for lang in shuffle_control reverse_full parity_word fixed_start parity_tok negtok; do
      run "$lang" "$seed"
    done
  done
  for seed in 0 14 41; do
    run fixed_end "$seed"
  done
  RUN_EXTRA_ENV=""
  if [ "${RUN_V3_H7:-1}" = "1" ]; then
    for seed in 14 41; do
      for lang in shuffle_control parity_word; do
        run_steps "$lang" "$seed" 6000
      done
    done
  fi
  # H7 3x (9000 steps = 1.18e9 tokens ~= 9 epochs): the design's first extension
  # priority — it doubles as the Kallini-token-budget fidelity arm (EXPDESIGN §5).
  if [ "${RUN_V3_H7_3X:-1}" = "1" ]; then
    for lang in shuffle_control parity_word; do
      run_steps "$lang" 0 9000
    done
  fi
fi

# ---------- [4d2] stretch tier (§10c, registered 2026-09-20 evening) ------------
# Registered exploratory arms, queued AFTER every paper-critical block so they
# can only ever add days, never delay a confirmatory family:
#   * datascale axis (§10c-5): PoS analog — fixed 3000-step budget over
#     deterministic 1M/10M-token sentence subsets; the full-data cell is the
#     existing 1x cell, so the axis needs only the two reduced scales.
#   * ladder-probe replay (§10c-3): the two class-P cells that finished BEFORE
#     the in-process ladder probe existed (parity_word s0, fixed_start s0),
#     re-run under LADDER_PROBE=1 so the acquisition-dynamics panel is seed-
#     complete. 1x budget, exploratory.
#   * LOGO generalization (§10c-8): train WITHOUT simple_wikipedia, evaluate
#     the same frozen draw; the analysis slices per-genre transfer deltas.
#   * model-scale axis (§10c-6): GPT-2 medium (355M) x 3 conditions x 2 seeds;
#     answers the paper's own Limitations ("bigger models may memorize away
#     the bias"). Slowest cells last.
if [ "${RUN_V3:-0}" = "1" ] && [ "${RUN_V3_STRETCH:-1}" = "1" ]; then
  # --- data-scale subsets (deterministic; runs once, cheap, CPU-only) --------
  if [ "${QUEUE_DRY_RUN:-0}" != "1" ] \
     && [ ! -f "$KALLINI_DATA_PATH/babylm_data_perturbed/.datascale_v1" ]; then
    note "generating data-scale subsets (1M/10M tokens x 3 conditions)"
    if $PYTHON experiments_v2/design_v3/make_datascale_subsets.py \
        >> experiments_v2/kallini_repro/data_prep.log 2>&1; then
      printf '%s\n' "datascale-v1" > "$KALLINI_DATA_PATH/babylm_data_perturbed/.datascale_v1"
    else
      note "WARN datascale subset generation failed -> datascale cells skipped this pass"
    fi
  fi
  DATASCALE_CONDS="shuffle_control parity_word"   # §10c-13 DL-3: 12→4 cells (sub10M + fixed_start deferred)
  DS_DIR=experiments_v2/kallini_repro/results_datascale
  mkdir -p "$DS_DIR"
  for scale in sub1M; do
    pending_ds=$(find "$DS_DIR" -name exp1_result.json 2>/dev/null | grep -c "$scale" || true)
    if [ "$pending_ds" -lt 4 ]; then
      for c in $DATASCALE_CONDS; do
        for s in 0 14; do
          if [ "${QUEUE_DRY_RUN:-0}" = "1" ]; then note "[dry] would train datascale $scale $c/seed$s"; continue; fi
          if $NICE env REPRO_RESULTS=$DS_DIR \
                REPRO_DATA_SUBDIR="babylm_${c}_${scale}" REPRO_DIR_TAG="_${scale}" \
                $PYTHON experiments_v2/kallini_repro/train_exp1.py "$c" --seed "$s" --skip-if-done \
                >> experiments_v2/kallini_repro/results_datascale/queue.log 2>&1; then
            note "OK   datascale $scale $c/seed$s"
          else
            note "FAIL datascale $scale $c/seed$s"
            fail=$((fail+1))
          fi
        done
      done
    fi
  done
  if [ "${QUEUE_DRY_RUN:-0}" != "1" ]; then
    RESULTS_DIR=$DS_DIR RESULTS_BRANCH=v2-results-datascale RESULTS_KIND=exp1_result.json \
      bash experiments_v2/kallini_repro/publish_results.sh || note "WARN datascale publish failed"
  fi

  # --- ladder-probe replay — DEFERRED (§10c-13 DL-4) --------------------------
  RP_DIR=experiments_v2/kallini_repro/results_ladder_probe
  mkdir -p "$RP_DIR"
  pending_rp=$(find "$RP_DIR" -name exp1_result.json 2>/dev/null | wc -l)
  if [ "${RUN_V3_RP:-0}" = "1" ] && [ "$pending_rp" -lt 2 ]; then
    for c in parity_word fixed_start; do
      if [ "${QUEUE_DRY_RUN:-0}" = "1" ]; then note "[dry] would train ladder-replay $c/seed0"; continue; fi
      if $NICE env REPRO_RESULTS=$RP_DIR LADDER_PROBE=1 \
          $PYTHON experiments_v2/kallini_repro/train_exp1.py "$c" --seed 0 --skip-if-done \
          >> experiments_v2/kallini_repro/results_ladder_probe/queue.log 2>&1; then
        note "OK   ladder-replay $c/seed0"
      else
        note "FAIL ladder-replay $c/seed0"
        fail=$((fail+1))
      fi
    done
  fi
  if [ "${QUEUE_DRY_RUN:-0}" != "1" ]; then
    RESULTS_DIR=$RP_DIR RESULTS_BRANCH=v2-results-ladder-probe RESULTS_KIND=exp1_result.json \
      bash experiments_v2/kallini_repro/publish_results.sh || note "WARN ladder replay publish failed"
  fi

  # --- LOGO generalization — DEFERRED (§10c-13 DL-1) --------------------------
  LOGO_DIR=experiments_v2/kallini_repro/results_logo
  mkdir -p "$LOGO_DIR"
  pending_logo=$(find "$LOGO_DIR" -name exp1_result.json 2>/dev/null | wc -l)
  if [ "${RUN_V3_LOGO:-0}" = "1" ] && [ "$pending_logo" -lt 2 ]; then
    if [ "${QUEUE_DRY_RUN:-0}" = "1" ]; then note "[dry] would generate LOGO subsets"; fi
    if [ "${QUEUE_DRY_RUN:-0}" != "1" ] \
       && $PYTHON experiments_v2/design_v3/make_datascale_subsets.py --logo \
            >> experiments_v2/kallini_repro/data_prep.log 2>&1; then
      for c in shuffle_control parity_word; do
        if $NICE env REPRO_RESULTS=$LOGO_DIR \
              REPRO_DATA_SUBDIR="babylm_${c}_logo7sw" REPRO_DIR_TAG="_logo7sw" \
              $PYTHON experiments_v2/kallini_repro/train_exp1.py "$c" --seed 0 --skip-if-done \
              >> experiments_v2/kallini_repro/results_logo/queue.log 2>&1; then
          note "OK   logo7sw $c/seed0"
        else
          note "FAIL logo7sw $c/seed0"
          fail=$((fail+1))
        fi
      done
    else
      [ "${QUEUE_DRY_RUN:-0}" != "1" ] && note "WARN LOGO subset generation failed -> skipped this pass"
    fi
  fi
  if [ "${QUEUE_DRY_RUN:-0}" != "1" ]; then
    RESULTS_DIR=$LOGO_DIR RESULTS_BRANCH=v2-results-logo RESULTS_KIND=exp1_result.json \
      bash experiments_v2/kallini_repro/publish_results.sh || note "WARN logo publish failed"
  fi

  # --- capmatch n=5 extension (§10c-2; owner ruling 2026-09-21: unconditional) --
  # The confirmatory architecture family F4 is registered at n=5 (STATS_PLAN_V3 §2
  # headline rule). The GPT-2 side already has seeds 53/96 for all three capmatch
  # conditions (extension tier), so the paired contrast can reach n=5; without
  # these cells F4 stays at n=3 and only the parametric test can reach p<.05.
  # Placed in the stretch tier on purpose: it can only extend the campaign, never
  # delay a paper-critical block.
  if [ "${RUN_V3_CAPMATCH_EXT:-1}" = "1" ]; then
    # The tier may run with §[4c2] disabled, so re-establish its globals rather
    # than relying on that block having executed (LSTM_LR would be empty -> crash).
    CAPMATCH_DIR=${CAPMATCH_DIR:-experiments_v2/kallini_repro/results_lstm_gpu_capmatch}
    CAPMATCH_CONDS=${CAPMATCH_CONDS:-shuffle_control reverse_full parity_word}
    mkdir -p "$CAPMATCH_DIR"
    CAPMATCH_LR=${CAPMATCH_LR:-$(cat "$CAPMATCH_DIR/.frozen_lr" 2>/dev/null || true)}
    if [ -z "$CAPMATCH_LR" ] || [ "$CAPMATCH_LR" = "FAIL" ]; then
      note "WARN capmatch ext: no frozen LR (probe incomplete) -> stretch-tier capmatch cells are skipped (no silent 1e-3 fallback)"
    fi
    pending_cap_ext=$(find "$CAPMATCH_DIR" -name lstm_result.json 2>/dev/null | wc -l)
    if [ "$pending_cap_ext" -lt 15 ]; then
      for s in 53 96; do
        for c in $CAPMATCH_CONDS; do
          run_lstm_capmatch "$c" "$s"
        done
      done
    fi
    if [ "${QUEUE_DRY_RUN:-0}" != "1" ]; then
      RESULTS_DIR=$CAPMATCH_DIR RESULTS_BRANCH=v2-results-lstm-gpu-capmatch \
        RESULTS_KIND=lstm_result.json \
        bash experiments_v2/kallini_repro/publish_results.sh || note "WARN capmatch ext publish failed"
    fi
  fi

  # --- model-scale axis — REDUCED 6→2 cells (§10c-13 DL-2) --------------------
  MS_DIR=experiments_v2/kallini_repro/results_model_scale
  mkdir -p "$MS_DIR"
  MS_CONDS="shuffle_control parity_word"
  pending_ms=$(find "$MS_DIR" -name exp1_result.json 2>/dev/null | wc -l)
  if [ "$pending_ms" -lt 2 ]; then
    for c in $MS_CONDS; do
      for s in 0; do
        if [ "${QUEUE_DRY_RUN:-0}" = "1" ]; then note "[dry] would train model-scale $c/seed$s (gpt2_medium)"; continue; fi
        if $NICE env REPRO_RESULTS=$MS_DIR REPRO_MODEL_SIZE=gpt2_medium REPRO_MICRO_BATCH=2 \
            $PYTHON experiments_v2/kallini_repro/train_exp1.py "$c" --seed "$s" --skip-if-done \
            >> experiments_v2/kallini_repro/results_model_scale/queue.log 2>&1; then
          note "OK   model-scale $c/seed$s"
        else
          note "FAIL model-scale $c/seed$s"
          fail=$((fail+1))
        fi
      done
    done
  fi
  if [ "${QUEUE_DRY_RUN:-0}" != "1" ]; then
    RESULTS_DIR=$MS_DIR RESULTS_BRANCH=v2-results-model-scale RESULTS_KIND=exp1_result.json \
      bash experiments_v2/kallini_repro/publish_results.sh || note "WARN model-scale publish failed"
  fi
fi

# ---------- [4e] BabyLM probe smoke (code-path check only, no conclusions) -----
# Runs the probe suite on the first available final/ checkpoint with 10 pairs and
# writes into a quarantine tree. It is a code-path check: probe numbers from
# shuffle-class checkpoints are not results (they must come from the P class).
if [ "${RUN_V3:-0}" = "1" ] && [ "${RUN_V3_PROBE_SMOKE:-1}" = "1" ] \
   && [ "${QUEUE_DRY_RUN:-0}" != "1" ]; then
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
