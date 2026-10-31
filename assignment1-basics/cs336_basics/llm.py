import torch
import math

from einops import einsum


class Linear(torch.nn.Module):
    """
    A simple implementation of a linear (fully connected) layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialize the layer.

        Args:
            in_features (int): Final dimension of the input.
            out_features (int): Final dimension of the output.
            device (torch.device | None): Device to store the parameters on.
            dtype (torch.dtype | None): Data type of the parameters.
        """
        super().__init__()
        self.weights = torch.nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        std_dev = (2 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(
            self.weights, mean=0, std=std_dev, a=-3 * std_dev, b=3 * std_dev
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.
        """
        return einsum(
            x,
            self.weights,
            "batch_size sequence_length d_model, d_ff d_model -> batch_size sequence_length d_ff",
        )


class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Initialize the embedding layer with the given parameters.
        num_embeddings: int  Size of the vocabulary
        embedding_dim: int  Dimension of the embedding vectors, i.e., 𝑑model
        device: torch.device | None = None  Device to store the parameters on
        dtype: torch.dtype | None = None  Data type of the parameters
        """
        super().__init__()
        self.weights = torch.nn.Parameter(
            torch.empty(
                num_embeddings, embedding_dim, device=device, dtype=dtype
            )
        )
        torch.nn.init.trunc_normal_(self.weights, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Args:
            d_model (int): The dimension of the model.
            eps (float, optional): A small value to avoid division by zero.
            Defaults to 1e-5.
            device (optional): The device to store the parameters. Defaults to
            None.
            dtype (optional): The data type of the parameters. Defaults to None.
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weights = torch.nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        return (x_normed * self.weights).to(in_type)


class SwiGLUFF(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model

        if d_ff is None:
            # round up to the nearest multiple of 64
            d_ff = (round(8 / 3 * d_model) + 63) // 64 * 64
        self.d_ff: int = d_ff
        self.linear1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.linear2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)
        self.linear3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.linear1(x)
        return self.linear2(a * torch.sigmoid(a) * self.linear3(x))


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
        # This is the outer product. Storing the angles for each position and
        # frequency.
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


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Apply softmax along the given dimension. This doens't change the shape of
    the tensor.
    """
    x_reduced = x - x.max(dim=dim, keepdim=True).values
    return torch.exp(x_reduced) / torch.sum(
        torch.exp(x_reduced), dim=dim, keepdim=True
    )


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """
    query: shape (batch_size, ..., seq_len, dk)
    key: (batch_size, ..., seq_len, dk)
    value: (batch, ..., seq_len, dv)
    mask: (seq_len, seq_len) - A matrix of True and False. Row i of the matrix
    indiciates which key the query i should attend to. [i,j] = True means query
    i should attend to key j.
    """
    dk: int = query.shape[-1]
    # dot product shape (batch, ..., queries_len, keys_len)
    dot_product: torch.Tensor = einsum(
        query, key, "... queries dk, ... keys dk -> ... queries keys"
    )
    pre_softmax = dot_product / math.sqrt(dk)
    # Apply mask
    if mask is not None:
        pre_softmax = pre_softmax.masked_fill(~mask, -torch.inf)

    sm = softmax(pre_softmax, dim=-1)
    attention = einsum(
        sm, value, "... queries keys, ... keys values -> ... queries values"
    )
    return attention
