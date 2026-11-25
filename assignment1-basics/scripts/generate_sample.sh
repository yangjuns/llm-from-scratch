#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Generate text from a trained TinyStories checkpoint, e.g.
# CHECKPOINT=checkpoints/batch-64-lr-1e-3/latest.pt TEMPERATURE=0.7 \
#     bash scripts/generate_sample.sh
export CHECKPOINT="${CHECKPOINT:-checkpoints/batch-64-lr-1e-3/latest.pt}"
export PROMPT="${PROMPT:-<|endoftext|>}"
export MAX_TOKENS="${MAX_TOKENS:-256}"
export TEMPERATURE="${TEMPERATURE:-0.8}"
export TOP_P="${TOP_P:-0.9}"
export SEED="${SEED:-0}"
export NOPE="${NOPE:-0}"

case "$NOPE" in
    0|1) ;;
    *) echo "NOPE must be 0 or 1" >&2; exit 1 ;;
esac

uv run python - <<'EOF'
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
    enable_rope=os.environ["NOPE"] != "1",
)
ckpt = torch.load(os.environ["CHECKPOINT"], map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"checkpoint step: {ckpt['iteration']} | device: {device}")

torch.manual_seed(int(os.environ["SEED"]))
print(
    decode(
        model,
        tok,
        os.environ["PROMPT"],
        int(os.environ["MAX_TOKENS"]),
        temperature=float(os.environ["TEMPERATURE"]),
        top_p_threshold=float(os.environ["TOP_P"]),
    )
)
EOF
