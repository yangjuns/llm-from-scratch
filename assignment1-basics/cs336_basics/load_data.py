import numpy as np
import numpy.typing as npt
import torch


def load_data(
    x: npt.NDArray[np.integer],
    batch_size: int,
    context_length: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load data into batches for training.
    x: one-dimensional integer array of token IDs
    batch_size: number of sequences in the batch
    context_length: number of input tokens in each sequence
    device: torch device to load the batch onto

    Returns:
    The sampled input sequences and their next-token targets, both with
    shape (batch_size, context_length).
    """
    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional array of token IDs")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if context_length <= 0:
        raise ValueError("context_length must be positive")

    num_start_indices = len(x) - context_length
    if num_start_indices <= 0:
        raise ValueError("x must contain at least context_length + 1 tokens")

    start_indices = np.random.randint(0, num_start_indices, size=batch_size)
    window_indices = start_indices[:, None] + np.arange(context_length + 1)
    windows = x[window_indices]

    inputs = torch.as_tensor(windows[:, :-1], dtype=torch.int, device=device)
    targets = torch.as_tensor(windows[:, 1:], dtype=torch.int, device=device)
    return inputs, targets
