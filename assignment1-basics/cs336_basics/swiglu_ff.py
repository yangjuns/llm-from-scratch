import torch

from cs336_basics.linear import Linear


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
