#!/usr/bin/env bash
set -euo pipefail

WARMUP_STEPS="${1:-5}"
if [[ $# -gt 1 || ! "$WARMUP_STEPS" =~ ^[0-9]+$ ]]; then
    printf 'Usage: %s [nonnegative-warmup-steps]\n' "$0" >&2
    exit 1
fi

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p results
LOG_FILE="results/benchmark_model_sizes.log"
: > "$LOG_FILE"

configs=(
    "small 768 3072 12 12"
    "medium 1024 4096 24 16"
    "large 1280 5120 36 20"
    "xl 2560 10240 32 32"
    "10B 4608 12288 50 36"
)
modes=(
    "forward"
    "forward-backward"
    "forward-backward-optimizer"
)

for config in "${configs[@]}"; do
    read -r size d_model d_ff num_layers num_heads <<< "$config"
    for mode in "${modes[@]}"; do
        {
            printf '\n=== %s | %s ===\n' "$size" "$mode"
            uv run python cs336_systems/benchmark.py \
                --vocab-size 10000 \
                --context-length 512 \
                --d-model "$d_model" \
                --d-ff "$d_ff" \
                --num-layers "$num_layers" \
                --num-heads "$num_heads" \
                --batch-size 4 \
                --warmup-steps "$WARMUP_STEPS" \
                --steps 10 \
                --mode "$mode" \
                --device cuda
        } 2>&1 | tee -a "$LOG_FILE"
    done
done

printf '\nSaved results to %s\n' "$LOG_FILE"
