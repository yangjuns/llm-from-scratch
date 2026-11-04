#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

max_iters=40000
# Stop early without changing the LR schedule, e.g.
# STOP_STEP=5000 bash scripts/sweep_learning_rate_higher.sh
stop_step="${STOP_STEP:-$max_iters}"

for settings in "1e-2 1e-3" "3e-2 3e-3"; do
    read -r max_lr min_lr <<< "$settings"
    run_name="lr-${max_lr}"

    uv run python -m cs336_basics.train \
        --train-data data/tokenized/tinystories_train.npy \
        --valid-data data/tokenized/tinystories_valid.npy \
        --seed 0 \
        --batch-size 32 \
        --max-iters "$max_iters" \
        --stop-step "$stop_step" \
        --max-lr "$max_lr" \
        --min-lr "$min_lr" \
        --log-interval 10 \
        --eval-interval 500 \
        --checkpoint-interval 5000 \
        --checkpoint-dir "checkpoints/$run_name" \
        --run-name "$run_name"
done
