import argparse

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from cs336_basics.model import RotaryEmbedding, TransformerBlock


NUM_LAYERS = 32
D_MODEL = 2560
D_FF = 10240
NUM_HEADS = 16
CONTEXT_LENGTH = 2048
BATCH_SIZE = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure XL checkpoint-group peak CUDA memory."
    )
    parser.add_argument("--checkpoint-block-size", type=int, default=1)
    args = parser.parse_args()
    if args.checkpoint_block_size not in (1, 2, 4):
        parser.error("--checkpoint-block-size must be 1, 2, or 4")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This experiment requires CUDA.")

    block = TransformerBlock(
        d_model=D_MODEL,
        d_ff=D_FF,
        num_heads=NUM_HEADS,
        positional_encoder=RotaryEmbedding(
            context_length=CONTEXT_LENGTH,
            dim=D_MODEL // NUM_HEADS,
        ),
    ).to("cuda")
    block = torch.compile(block, fullgraph=True)

    def run_group(x: Tensor) -> Tensor:
        for _ in range(args.checkpoint_block_size):
            x = block(x)
        return x

    x = torch.randn(
        BATCH_SIZE,
        CONTEXT_LENGTH,
        D_MODEL,
        device="cuda",
        requires_grad=True,
    )

    torch.cuda.reset_peak_memory_stats()
    y = x
    for _ in range(NUM_LAYERS // args.checkpoint_block_size):
        y = checkpoint(run_group, y, use_reentrant=False)
    y.sum().backward()
    torch.cuda.synchronize()

    print(f"Checkpoint blocks per group: {args.checkpoint_block_size}")
    print(
        f"Peak allocated: {torch.cuda.max_memory_allocated() / 1024**3:.3f} GiB"
    )
    print(
        f"Peak reserved: {torch.cuda.max_memory_reserved() / 1024**3:.3f} GiB"
    )


if __name__ == "__main__":
    main()
