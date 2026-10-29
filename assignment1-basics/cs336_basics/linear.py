import torch

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
