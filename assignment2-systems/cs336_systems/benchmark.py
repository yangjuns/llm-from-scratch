import argparse
import statistics
import timeit
from typing import cast

import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW

MODES = ("forward", "forward-backward", "forward-backward-optimizer")


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize a Transformer model for benchmarking."
    )
    parser.add_argument("--vocab-size", type=positive_int, default=10_000)
    parser.add_argument("--context-length", type=positive_int, default=256)
    parser.add_argument("--d-model", type=positive_int, default=512)
    parser.add_argument("--num-layers", type=positive_int, default=4)
    parser.add_argument("--num-heads", type=positive_int, default=16)
    parser.add_argument("--d-ff", type=positive_int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--warmup-steps", type=positive_int, default=10)
    parser.add_argument("--steps", type=positive_int, default=100)
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="forward",
        help="steps to time: forward, forward-backward, or forward-backward-optimizer",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.d_model % args.num_heads != 0:
        parser.error("--d-model must be divisible by --num-heads")
    if (args.d_model // args.num_heads) % 2 != 0:
        parser.error("--d-model / --num-heads must be even for RoPE")
    if not 0 < args.rope_theta < float("inf"):
        parser.error("--rope-theta must be finite and positive")
    return args


def main() -> None:
    args = parse_args()
    # Initialize model
    model = cast(
        torch.nn.Module,
        BasicsTransformerLM(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            rope_theta=args.rope_theta,
        ),
    ).to(args.device)
    # Initialize optimizer
    optimizer = cast(torch.optim.Optimizer, AdamW(model.parameters()))
    num_params = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Initialized {num_params:,} parameters on {args.device} with batch "
        f"size {args.batch_size}"
    )
    device = next(model.parameters()).device
    print(f"Running on CUDA: {device.type == 'cuda'}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    # Generate random batch of data (batch_size, seq)
    x: torch.Tensor = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.context_length),
        device=args.device,
    )

    # Define each step as a lambda for timing
    def step():
        # check the mode
        if args.mode == "forward":
            model(x)
        elif args.mode == "forward-backward":
            y = model(x)
            y.sum().backward()
        elif args.mode == "forward-backward-optimizer":
            y = model(x)
            optimizer.zero_grad()
            y.sum().backward()
            optimizer.step()
        torch.cuda.synchronize() if args.device == "cuda" else None

    # Run w steps of warm up
    for _ in range(args.warmup_steps):
        step()

    # Run timed steps
    timings = timeit.repeat(step, number=1, repeat=args.steps)
    elapsed = sum(timings)
    print(f"Completed {args.steps} steps in {elapsed:.9f}s")
    print(
        f"Mean: {statistics.mean(timings):.9f}s/step, "
        f"Std (population): {statistics.pstdev(timings):.9f}s/step"
    )


if __name__ == "__main__":
    main()
