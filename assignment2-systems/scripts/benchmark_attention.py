import argparse
import csv
import gc
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from cs336_basics.model import scaled_dot_product_attention


BATCH_SIZE = 8
DEFAULT_DIMENSIONS = (16, 32, 64, 128)
DEFAULT_SEQUENCE_LENGTHS = (256, 1024, 4096, 8192, 16384)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch scaled dot-product attention."
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/benchmark_attention.csv"),
    )
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    return args


def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    return scaled_dot_product_attention(q, k, v)


def elapsed_ms(fn: Any) -> tuple[float, Any]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), result


def run_case(
    dimension: int,
    sequence_length: int,
    iterations: int,
    warmup: int,
) -> dict[str, Any]:
    shape = (BATCH_SIZE, sequence_length, dimension)
    q = torch.randn(shape, device="cuda", requires_grad=True)
    k = torch.randn(shape, device="cuda", requires_grad=True)
    v = torch.randn(shape, device="cuda", requires_grad=True)
    upstream = torch.randn(shape, device="cuda")

    for _ in range(warmup):
        output = attention(q, k, v)
        output.backward(upstream)
        q.grad = k.grad = v.grad = None
    torch.cuda.synchronize()

    forward_times = []
    backward_times = []
    memory_before_backward = 0

    for _ in range(iterations):
        forward_ms, output = elapsed_ms(lambda: attention(q, k, v))
        forward_times.append(forward_ms)
        memory_before_backward = max(
            memory_before_backward,
            torch.cuda.memory_allocated(),
        )
        backward_ms, _ = elapsed_ms(lambda: output.backward(upstream))
        backward_times.append(backward_ms)
        q.grad = k.grad = v.grad = None

    return {
        "dimension": dimension,
        "sequence_length": sequence_length,
        "forward_ms": mean(forward_times),
        "backward_ms": mean(backward_times),
        "memory_before_backward_gib": (memory_before_backward / 1024**3),
        "status": "ok",
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires CUDA.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []

    for dimension in DEFAULT_DIMENSIONS:
        for sequence_length in DEFAULT_SEQUENCE_LENGTHS:
            label = f"d_model={dimension}, sequence_length={sequence_length}"
            print(f"Benchmarking {label}", flush=True)
            try:
                result = run_case(
                    dimension,
                    sequence_length,
                    args.iterations,
                    args.warmup,
                )
                print(
                    f"  forward={result['forward_ms']:.3f} ms, "
                    f"backward={result['backward_ms']:.3f} ms, "
                    "memory before backward="
                    f"{result['memory_before_backward_gib']:.3f} GiB",
                    flush=True,
                )
            except torch.OutOfMemoryError:
                result = {
                    "dimension": dimension,
                    "sequence_length": sequence_length,
                    "forward_ms": "",
                    "backward_ms": "",
                    "memory_before_backward_gib": "",
                    "status": "oom",
                }
                print("  OOM", flush=True)
            finally:
                gc.collect()
                torch.cuda.empty_cache()

            results.append(result)

    fieldnames = (
        "dimension",
        "sequence_length",
        "forward_ms",
        "backward_ms",
        "memory_before_backward_gib",
        "status",
    )
    with args.output.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
