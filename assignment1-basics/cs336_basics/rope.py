import torch
from einops import einsum


class RoPE(torch.nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.d_model = d_k * 2  # Typically, d_model is twice d_k for RoPE
        # construct the rotary positional embedding matrix
        seq = torch.arange(max_seq_len, device=device, dtype=dtype)
        freq = 1.0 / (
            theta ** (torch.arange(0, d_k, 2, device=device, dtype=dtype) / d_k)
        )
        # This is the outer product. Storing the angles for each position and frequency.
        angles = einsum(seq, freq, "seq, freq -> seq freq")
        self.cos_cached = torch.cos(angles)
        self.sin_cached = torch.sin(angles)

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Process an input tensor of shape (..., seq_len, d_k) and return a
        tensor of the same shape.

        Note that you should tolerate x with an arbitrary number of batch
        dimensions. You should assume that the token positions are a tensor of
        shape (..., seq_len) specifying the token positions of x along the
        sequence dimension.

        x : torch.Tensor
            The input tensor of shape (..., seq_len, d_k).
        token_positions : torch.Tensor
            The tensor specifying the token positions of x along the sequence
            dimension, of shape (..., seq_len).

        Returns
        -------
        torch.Tensor
            The rotated tensor of the same shape as x.
        """
        cos = self.cos_cached[token_positions]  # (..., seq_len, d_k/2)
        sin = self.sin_cached[token_positions]  # (..., seq_len, d_k/2)
        x_even = x[..., 0::2]  # (..., seq_len, d_k/2)
        x_odd = x[..., 1::2]  # (..., seq_len, d_k/2)
        x_rotated_even = x_even * cos - x_odd * sin  # (..., seq_len, d_k/2)
        x_rotated_odd = x_even * sin + x_odd * cos  # (..., seq_len, d_k/2)
        x_rotated = torch.stack(
            [x_rotated_even, x_rotated_odd], dim=-1
        )  # (..., seq_len, d_k/2, 2)
        x_rotated = x_rotated.flatten(-2)  # (..., seq_len, d_k)

        return x_rotated
