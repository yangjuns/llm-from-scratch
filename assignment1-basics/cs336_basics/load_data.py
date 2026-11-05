import os
from typing import BinaryIO, IO

import numpy as np
import numpy.typing as npt
import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike[str] | BinaryIO | IO[bytes],
) -> None:
    """
    Serialize the model, optimizer, and completed training iteration.

    Args:
        model: Model whose state should be saved.
        optimizer: Optimizer whose state should be saved.
        iteration: Number of training iterations completed.
        out: Filesystem path or binary file-like object to write the checkpoint to.
    """
    # create a dictionary to save
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike[str] | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Restore the model and optimizer state from a serialized checkpoint.

    Args:
        src: Filesystem path or binary file-like object containing the checkpoint.
        model: Model whose state should be restored.
        optimizer: Optimizer whose state should be restored.

    Returns:
        The number of training iterations completed when the checkpoint was saved.
    """
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]


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
