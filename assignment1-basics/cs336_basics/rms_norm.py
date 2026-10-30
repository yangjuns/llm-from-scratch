import torch


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
