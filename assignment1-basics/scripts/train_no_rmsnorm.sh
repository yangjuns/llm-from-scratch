#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

max_iters=40000
# Wall-clock cap per run (override: TIME_LIMIT=10m bash scripts/train_no_rmsnorm.sh)
time_limit="${TIME_LIMIT:-30m}"

for settings in "1e-3 1e-4" "1e-4 1e-5"; do
    read -r max_lr min_lr <<< "$settings"
    run_name="no-rmsnorm-lr-${max_lr}"

    # timeout exits with 124 when the cap is hit; that is expected, not a failure
    timeout "$time_limit" uv run python -m cs336_basics.train \
        --train-data data/tokenized/tinystories_train.npy \
        --valid-data data/tokenized/tinystories_valid.npy \
        --no-rmsnorm \
        --seed 0 \
        --batch-size 32 \
        --max-iters "$max_iters" \
        --max-lr "$max_lr" \
        --min-lr "$min_lr" \
        --log-interval 10 \
        --eval-interval 500 \
        --checkpoint-interval 1000 \
        --checkpoint-dir "checkpoints/$run_name" \
        --run-name "$run_name" \
        --wandb-project cs336-assignment1 \
        --wandb-entity yangjuns-cmu-google || [ $? -eq 124 ]
done
