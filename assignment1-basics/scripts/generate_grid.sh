#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Generate one sample per (temperature, top-p) pair and write markdown, e.g.
# CHECKPOINT=checkpoints/batch-64-lr-1e-3/latest.pt \
#     bash scripts/generate_grid.sh
export CHECKPOINT="${CHECKPOINT:-checkpoints/batch-64-lr-1e-3/latest.pt}"
export PROMPT="${PROMPT:-<|endoftext|>}"
export MAX_TOKENS="${MAX_TOKENS:-256}"
export SEED="${SEED:-0}"
export TEMPERATURES="${TEMPERATURES:-0.5 0.8 1.0}"
export TOP_PS="${TOP_PS:-0.7 0.9 1.0}"
export OUTPUT="${OUTPUT:-logs/generation_grid.md}"

mkdir -p "$(dirname "$OUTPUT")"

uv run python - <<'EOF' | tee "$OUTPUT"
import os

import torch

from cs336_basics.llm import TransformerLM
from cs336_basics.llm_decode import decode
from cs336_basics.tokenizer import Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = Tokenizer.from_files(
    "artifacts/tinystories_bpe/vocab.json",
    "artifacts/tinystories_bpe/merges.json",
    special_tokens=["<|endoftext|>"],
)
# Model settings match the defaults in cs336_basics/train.py.
model = TransformerLM(
    vocab_size=10_000,
    context_length=256,
    d_model=512,
    num_layers=4,
    num_heads=16,
    d_ff=1344,
    rope_theta=10_000.0,
    device=torch.device(device),
)
ckpt = torch.load(os.environ["CHECKPOINT"], map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()

num_params = sum(p.numel() for p in model.parameters())
print(f"Checkpoint: `{os.environ['CHECKPOINT']}` (step {ckpt['iteration']})")
print(f"Parameters: {num_params:,}")
print(f"Prompt: `{os.environ['PROMPT']}` | seed {os.environ['SEED']}")

for temperature in map(float, os.environ["TEMPERATURES"].split()):
    for top_p in map(float, os.environ["TOP_PS"].split()):
        torch.manual_seed(int(os.environ["SEED"]))
        text = decode(
            model,
            tok,
            os.environ["PROMPT"],
            int(os.environ["MAX_TOKENS"]),
            temperature=temperature,
            top_p_threshold=top_p,
        )
        print(f"\n### temperature {temperature}, top-p {top_p}\n")
        print(f"```text\n{text}\n```")
EOF
