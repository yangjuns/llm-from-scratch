# LLM Implementation Learning Notes

## RMSNorm and tensor arithmetic

### Computing RMS

- Tensors support `+ - * / **` elementwise, with broadcasting.
- RMS over the last dim:

  ```python
  rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
  x_normed = x / rms
  ```

  - `keepdim=True` keeps shape `(..., 1)` so the division broadcasts.
  - `torch.rsqrt(...)` gives `1/sqrt(...)`, so `x * torch.rsqrt(ms + eps)` works.
- `eps` goes inside the sqrt (per the assignment formula), not added to the
  RMS afterwards. `x.norm() / sqrt(d) + eps` is close but not identical.
- Upcast to float32 for the computation, then cast back to the input dtype.
- Pass `device`/`dtype` to `torch.ones(d_model, ...)` for the weight.

### Multiplying by `self.weight`

- `x * self.weight` is elementwise multiplication with broadcasting, not a
  matrix multiply.
- `x: (batch, seq, d_model)`, `weight: (d_model,)`. Broadcasting aligns shapes
  from the right, so weight acts as `(1, 1, d_model)`. Each feature is scaled
  by a learned per-feature gain; output shape is unchanged.
- Fails if trailing dims are neither equal nor 1, e.g. `(2, 3) * (2,)`.

| Form | Meaning |
| --- | --- |
| `a * b`, `torch.mul` | elementwise, broadcasts |
| `a @ b`, `torch.matmul` | matrix multiply, leading batch dims broadcast |
| `torch.einsum("...i,oi->...o", x, W)` | explicit-index form, handy for Linear |

### What `eps` does

- Numerical stability: the denominator is `sqrt(mean(x²) + eps)`.
- Prevents division by zero when all features are 0 (output stays 0, not NaN).
- Keeps gradients finite, since the derivative of sqrt blows up near 0.
- Guards against underflow of `x²` in fp16/bf16 (another reason to upcast).
- Negligible for normal-sized activations.

## Linear layer: why no bias

- Modern LLMs (LLaMA, PaLM, the assignment architecture) omit biases.
- Normalization makes a constant offset largely redundant: RMSNorm has a
  learned scale but no shift.
- Saves parameters, optimizer state, and memory traffic at essentially no
  quality cost.
- PaLM reported more stable training at large scale without biases.
- Offsets can still be learned elsewhere (embeddings, residual stream, SwiGLU
  gating).
- So `y = x @ W.T` with `W: (d_out, d_in)`, or
  `torch.einsum("...i,oi->...o", x, W)`.
