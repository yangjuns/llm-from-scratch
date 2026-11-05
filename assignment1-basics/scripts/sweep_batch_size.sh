#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root/assignment1-basics"

# Memory estimate from notes/a1-llm-implementation.md ("AdamW peak memory"):
#   total bytes = 16P + 4A
#   P = 2Vd + L(4d^2 + 3d*d_ff + 2d) + d
#   A = L((56/3)BTd + 2BhT^2) + BTd + 2BTV
# Here B=batch size, T=context length, L=layers, d=model width, h=heads,
# V=vocabulary size, and d_ff=feed-forward width. With this trainer's defaults
# (V=10000, T=256, L=4, d=512, h=16, d_ff=1344), this estimates
# 363,143,168 + 93,705,557*B bytes. A nominal 24 GiB budget suggests B<=271,
# but this component-count estimate is not the actual CUDA peak; runtime
# workspaces, allocator overhead, and other GPU use can lower the fit limit.
# Treat the sweep's first CUDA OOM (if any) as the empirical limit.

# Match the token budget of 40,000 steps at batch size 32 and context length 256.
baseline_steps=40000
baseline_batch_size=32
context_length=256
baseline_warmup_iters=200
baseline_checkpoint_interval=500
run_time_limit=1h
target_tokens=$((baseline_steps * baseline_batch_size * context_length))

uv run python -c \
    'import torch; assert torch.cuda.is_available(), "CUDA is unavailable; this sweep requires a working GPU."'

batch_sizes=(1 2 4 8 16 32 64 128 256 512)
log_dir="logs/batch-size-sweep-lr-1e-3"
mkdir -p "$log_dir"

for batch_size in "${batch_sizes[@]}"; do
    tokens_per_step=$((batch_size * context_length))
    max_iters=$((target_tokens / tokens_per_step))
    warmup_iters=$(((baseline_warmup_iters * baseline_batch_size + batch_size / 2) / batch_size))
    log_interval=$(((10 * baseline_batch_size + batch_size / 2) / batch_size))
    eval_interval=$(((500 * baseline_batch_size + batch_size / 2) / batch_size))
    checkpoint_interval=$(((baseline_checkpoint_interval * baseline_batch_size + batch_size / 2) / batch_size))

    (( max_iters > 0 )) || max_iters=1
    (( warmup_iters > 0 )) || warmup_iters=1
    (( log_interval > 0 )) || log_interval=1
    (( eval_interval > 0 )) || eval_interval=1
    (( checkpoint_interval > 0 )) || checkpoint_interval=1

    echo "Batch size ${batch_size}: ${max_iters} steps (about $((max_iters * tokens_per_step)) tokens)"
    log_file="$log_dir/batch-${batch_size}.log"
    if timeout --signal=TERM --kill-after=30s "$run_time_limit" \
        uv run python -m cs336_basics.train \
        --device cuda \
        --train-data data/tokenized/tinystories_train.npy \
        --valid-data data/tokenized/tinystories_valid.npy \
        --seed 0 \
        --batch-size "$batch_size" \
        --max-iters "$max_iters" \
        --warmup-iters "$warmup_iters" \
        --max-lr 1e-3 \
        --min-lr 1e-4 \
        --log-interval "$log_interval" \
        --eval-interval "$eval_interval" \
        --checkpoint-interval "$checkpoint_interval" \
        --checkpoint-dir "checkpoints/batch-${batch_size}-lr-1e-3" \
        --run-name "batch-${batch_size}-lr-1e-3" \
        --wandb-project cs336-assignment1 \
        --wandb-entity yangjuns-cmu-google \
        2>&1 | tee "$log_file"; then
        echo "Completed batch size ${batch_size}."
    else
        status=$?
        if (( status == 124 )); then
            echo "Batch size ${batch_size} reached the ${run_time_limit} limit; continuing to the next size."
        else
            echo "Run failed at batch size ${batch_size} (exit ${status}). See ${log_file}; stopping the sweep."
            exit "$status"
        fi
    fi
done
