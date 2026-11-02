import torch


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
