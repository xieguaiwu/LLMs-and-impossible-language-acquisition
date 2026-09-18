#!/usr/bin/env bash
# BabyLM replication (Plan B minimum): natural / reversed / parity_negation
# x 5 seeds on gpt2 (the cell the original paper left at n=1 and where the
# parity-vs-natural question is most contested). LSTM BabyLM optional via
# RUN_LSTM=1.
#
# Data: download BabyLM 100M (devig? use babylm_100M) from babylm.github.io,
# place the raw text at experiments_v2/data_v2/babylm/raw/ then run the
# preparation script once:
#   $PYTHON experiments_v2/data_v2/prepare_babylm.py --raw experiments_v2/data_v2/babylm/raw
#
# Wall-clock: ~15 runs x 3-6 GPU-h (A100) interleaved => 2-3 days on one A100.

set -euo pipefail
cd "$(dirname "$0")/../.."

SEEDS=(42 43 44 45 46)
PYTHON=${PYTHON:-python3}
RUN_LSTM=${RUN_LSTM:-0}

for seed in "${SEEDS[@]}"; do
  for cond in natural reversed parity_negation; do
    echo "--- gpt2 / $cond / seed$seed"
    $PYTHON experiments_v2/training/train_lm.py \
      --model gpt2 --dataset babylm --condition "$cond" --seed "$seed"
  done
done

if [[ "$RUN_LSTM" == "1" ]]; then
  for seed in "${SEEDS[@]}"; do
    for cond in natural reversed parity_negation; do
      echo "--- lstm_matched / $cond / seed$seed"
      $PYTHON experiments_v2/training/train_lm.py \
        --model lstm_matched --dataset babylm --condition "$cond" --seed "$seed"
    done
  done
fi

$PYTHON experiments_v2/analysis/aggregate_seeds.py
$PYTHON experiments_v2/analysis/stats_tests.py
echo "DONE."
