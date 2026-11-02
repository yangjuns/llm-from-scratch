import torch
import math

from collections.abc import Callable, Iterable


def cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    logits: (batch, seq, vocab_size)
    target: (batch, seq) A batch of integers that indicate the target token id
    """
    # Adding a constant to logits doesn't change the loss
    stable_logits = (
        logits - logits.max(dim=-1, keepdim=True).values
    )  # shape(batch, seq, 1)
    # Cross Entropy L = logsumexp normalization - correct logits
    # Use gather to compute the correct logits
    indices = target.unsqueeze(-1)
    correct_logits = stable_logits.gather(dim=-1, index=indices)

    loss = (
        stable_logits.logsumexp(dim=-1) - correct_logits
    )  # shape (batch, seq)

    return loss.mean()


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
    ):
        defaults = {  # type: ignore
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)  # type: ignore

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None):
        with torch.enable_grad():
            loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:  # This is the model parameters
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                # Compute the gradient of the loss
                grad = p.grad
                # Compute adjusted learning rate
                adjusted_lr = lr * (math.sqrt(1 - beta2**t)) / (1 - beta1**t)
                # Apply weight decay
                p.sub_(p * weight_decay * lr)
                # Update the first moment estimate
                m = state.get("m", torch.zeros_like(p))
                m = beta1 * m + (1 - beta1) * grad
                state["m"] = m
                # Update second moment
                v = state.get("v", torch.zeros_like(p))
                v = beta2 * v + (1 - beta2) * grad**2
                state["v"] = v
                # Apply moment-adjusted weight update
                p.sub_(adjusted_lr * m / (torch.sqrt(v) + eps))
                state["t"] = t + 1
        return loss


def cosine_annealing_lr_schedule(
    t: int, lr_max: float, lr_min: float, warm_up_iters: int, cycle_iters: int
) -> float:
    # Warmup phase
    if t < warm_up_iters:
        return lr_max * t / warm_up_iters
    # Cosine annealing phase
    if warm_up_iters <= t and t <= cycle_iters:
        return lr_min + 0.5 * (lr_max - lr_min) * (
            1
            + math.cos(
                math.pi * (t - warm_up_iters) / (cycle_iters - warm_up_iters)
            )
        )
    # Post annealing phase
    return lr_min


def gradiant_clip(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    eps = 1e-6
    grads = [p.grad.flatten() for p in parameters if p.grad is not None]
    all_grads = torch.cat(grads)
    # Compute the l2 norm for all paramaters
    total_norm: float = float(all_grads.norm())  # type: ignore
    for p in parameters:
        if p.grad is not None:
            if total_norm > max_l2_norm:
                p.grad.mul_(max_l2_norm / (total_norm + eps))
