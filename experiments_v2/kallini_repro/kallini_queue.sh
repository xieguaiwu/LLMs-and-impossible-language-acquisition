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
# are missing. A half-finished pass must not mask the missing languages.
perturb_missing=""
for l in $LANGS; do
  ls "$KALLINI_DATA_PATH/babylm_data_perturbed/babylm_$l/babylm_100M"/*.train >/dev/null 2>&1 \
    || perturb_missing="$perturb_missing $l"
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
V3_LANGS_ALL="parity_word parity_tok negtok fixed_start fixed_end bare_reverse word_shuffle"
if [ "${RUN_V3:-0}" = "1" ]; then
  v3_missing=0
  for c in $V3_LANGS_ALL; do
    ls "$KALLINI_DATA_PATH/babylm_data_perturbed/babylm_$c/babylm_100M"/*.train >/dev/null 2>&1 \
      || v3_missing=1
  done
  if [ "$v3_missing" = "1" ]; then
    note "generating v3 class-P datasets"
    $PYTHON - <<'PYEOF' >> experiments_v2/kallini_repro/data_prep.log 2>&1 \
      || { note "V3 PERTURB FAIL"; exit 9; }
import sys
sys.path.insert(0, "experiments_v2/design_v3")
from v3_conditions import write_condition, CONDITIONS
from pathlib import Path
import glob, os
base = Path(os.environ.get("KALLINI_DATA_PATH", "/root/kallini_data"))
tagged_train = sorted(glob.glob(str(base / "babylm_data" / "babylm_100M" / "*_parsed.json")))
tagged_test  = sorted(glob.glob(str(base / "babylm_data" / "babylm_test" / "*_parsed.json")))
assert tagged_train and tagged_test, "shim tag the corpus first"
for lang in "parity_word parity_tok negtok fixed_start fixed_end bare_reverse word_shuffle".split():
    for tf in tagged_train:
        write_condition(lang, Path(tf), base / "babylm_data_perturbed", "100M")
    for tf in tagged_test:
        write_condition(lang, Path(tf), base / "babylm_data_perturbed", "test")
print("v3 P-class datasets done", len(tagged_train), "genres")
PYEOF
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
  if $NICE $PYTHON experiments_v2/kallini_repro/train_exp1.py "$lang" --seed "$seed" --skip-if-done \
      >> experiments_v2/kallini_repro/queue.log 2>&1; then
    note "OK   $lang/seed$seed"
  else
    note "FAIL $lang/seed$seed"
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
    for lang in parity_word fixed_start parity_tok negtok; do
      run "$lang" "$seed"
    done
  done
  if [ "${RUN_V3_H7:-1}" = "1" ]; then
    for lang in parity_word fixed_start shuffle_control; do
      if $NICE $PYTHON experiments_v2/kallini_repro/train_exp1.py "$lang" --seed 0 \
          --steps 6000 --skip-if-done >> experiments_v2/kallini_repro/queue.log 2>&1; then
        note "OK   2x $lang/seed0"
      else
        note "FAIL 2x $lang/seed0"; fail=$((fail+1))
      fi
    done
  fi
fi

# ---------- [5] aggregate --------------------------------------------------------
$PYTHON experiments_v2/kallini_repro/aggregate_exp1.py >> experiments_v2/kallini_repro/queue.log 2>&1 \
  || note "WARN aggregation failed"

n_done=$(find experiments_v2/kallini_repro/results -name exp1_result.json 2>/dev/null | wc -l)
note "kallini pass done: $n_done runs complete, $fail failures"
if [ "$fail" -gt 0 ]; then exit 1; fi
touch experiments_v2/kallini_repro/results/ALL_KALLINI_DONE
exit 0
