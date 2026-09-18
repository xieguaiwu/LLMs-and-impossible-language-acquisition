#!/usr/bin/env bash
# SVO full suite (Plan B, 5 seeds) -- the cheap, decisive experiments.
# Total: 3 conditions x 5 seeds x (gpt2 + lstm_matched + gpt2_tiny) = 45 runs
#        + control conditions on gpt2 (fixed_start/fixed_end/negtok/tok/shuffle)
# Run inside tmux/systemd-run on a GPU box. Each gpt2 run ~30-60 min on T4;
# lstm/lstm_matched ~10-20 min; gpt2_tiny ~20-30 min.

set -euo pipefail
cd "$(dirname "$0")/../.."          # repo root

SEEDS=(42 43 44 45 46)
PYTHON=${PYTHON:-python3}

echo "== [1/4] data =="
$PYTHON experiments_v2/data_v2/generate_svo.py --count 10000 --seed 42
$PYTHON experiments_v2/data_v2/conditions.py

echo "== [2/4] main conditions x 5 seeds =="
for seed in "${SEEDS[@]}"; do
  for cond in natural reversed parity_negation; do
    for model in gpt2 lstm_matched gpt2_tiny; do
      echo "--- $model / $cond / seed$seed"
      $PYTHON experiments_v2/training/train_lm.py \
        --model "$model" --dataset svo --condition "$cond" --seed "$seed"
    done
  done
done

echo "== [3/4] control conditions (gpt2, 5 seeds) =="
for seed in "${SEEDS[@]}"; do
  for cond in fixed_start_neg fixed_end_neg parity_negation_negtok word_shuffle; do
    echo "--- gpt2 / $cond / seed$seed"
    $PYTHON experiments_v2/training/train_lm.py \
      --model gpt2 --dataset svo --condition "$cond" --seed "$seed"
  done
done
# token-unit parity needs the gpt2 tokenizer; run it after the standard set
for seed in "${SEEDS[@]}"; do
  echo "--- gpt2 / parity_negation_tok / seed$seed"
  $PYTHON experiments_v2/training/train_lm.py \
    --model gpt2 --dataset svo --condition parity_negation_tok --seed "$seed"
done

echo "== [4/4] probes + statistics =="
for seed in "${SEEDS[@]}"; do
  dir=experiments_v2/results/svo/gpt2/parity_negation_seed$seed
  $PYTHON experiments_v2/probes/probes.py --model-dir "$dir" --model gpt2 \
    --n-pairs 500 --extrapolation --probe-diagnostic || true
done
$PYTHON experiments_v2/analysis/aggregate_seeds.py
$PYTHON experiments_v2/analysis/stats_tests.py
$PYTHON experiments_v2/analysis/plots.py

echo "DONE. See experiments_v2/results/aggregated/holm_corrected_tests.csv"
