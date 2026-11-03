import argparse
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.llm import TransformerLM
from cs336_basics.llm_train import (
    AdamW,
    cosine_annealing_lr_schedule,
    cross_entropy,
    gradiant_clip,
)
from cs336_basics.load_data import load_checkpoint, load_data, save_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Transformer LM.")
    # Data
    p.add_argument("--train-data", type=Path, required=True)
    p.add_argument("--valid-data", type=Path, required=True)
    # Model
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--d-ff", type=int, default=1344)
    p.add_argument("--rope-theta", type=float, default=10_000.0)
    # Optimizer
    p.add_argument("--max-lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--warmup-iters", type=int, default=200)
    p.add_argument(
        "--cosine-iters",
        type=int,
        default=None,
        help="Iterations for cosine annealing (default: --max-iters)",
    )
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # Loop
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-iters", type=int, default=5000)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=20)
    # Checkpointing
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--checkpoint-interval", type=int, default=500)
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint to resume training from",
    )
    # Misc
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--run-name", default=None)
    return p.parse_args()


@torch.no_grad()
def evaluate(
    model: TransformerLM,
    data: np.ndarray,
    args: argparse.Namespace,
) -> float:
    model.eval()
    losses = []
    for _ in range(args.eval_batches):
        x, y = load_data(data, args.batch_size, args.context_length, args.device)
        losses.append(cross_entropy(model(x), y.long()).item())
    model.train()
    return float(np.mean(losses))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Memory-map so the multi-GB token arrays are never fully loaded in RAM.
    train_data = np.load(args.train_data, mmap_mode="r")
    valid_data = np.load(args.valid_data, mmap_mode="r")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=torch.device(args.device),
    )
    optimizer = AdamW(
        list(model.parameters()),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_iter = 0
    if args.resume is not None:
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from {args.resume} at iteration {start_iter}")

    use_wandb = args.wandb_project is not None
    if use_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project, name=args.run_name, config=vars(args)
        )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cosine_iters = args.cosine_iters or args.max_iters
    # gradiant_clip iterates its argument twice, so it needs a list.
    params = list(model.parameters())
    num_params = sum(p.numel() for p in params)
    print(f"Training {num_params:,} parameters on {args.device}")

    start_time = time.time()
    for it in range(start_iter, args.max_iters):
        lr = cosine_annealing_lr_schedule(
            it, args.max_lr, args.min_lr, args.warmup_iters, cosine_iters
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = load_data(
            train_data, args.batch_size, args.context_length, args.device
        )
        optimizer.zero_grad()
        loss = cross_entropy(model(x), y.long())
        loss.backward()
        gradiant_clip(params, args.grad_clip)
        optimizer.step()

        step = it + 1
        metrics: dict[str, float] = {}
        if step % args.log_interval == 0:
            elapsed = time.time() - start_time
            metrics.update(train_loss=loss.item(), lr=lr, elapsed_s=elapsed)
            print(
                f"step {step} | loss {loss.item():.4f} | lr {lr:.2e} "
                f"| {elapsed:.0f}s",
                flush=True,
            )
        if step % args.eval_interval == 0 or step == args.max_iters:
            val_loss = evaluate(model, valid_data, args)
            metrics["val_loss"] = val_loss
            print(f"step {step} | val loss {val_loss:.4f}", flush=True)
        if use_wandb and metrics:
            wandb.log(metrics, step=step)
        if step % args.checkpoint_interval == 0 or step == args.max_iters:
            save_checkpoint(
                model, optimizer, step, args.checkpoint_dir / "latest.pt"
            )

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
