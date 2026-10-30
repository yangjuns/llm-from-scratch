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

## Feed-forward: regular FFN vs gated (SwiGLU)

- Regular FFN: `W2 · act(W1 x)`. Each hidden unit is a fixed pointwise
  function of a single projection.
- Gated (GLU): `W2 · (act(W1 x) ⊙ (W3 x))`. With `act = SiLU` this is SwiGLU.
  - Gate = `SiLU(W1 x)`, content = `W3 x`. `W1` learns when to pass features,
    `W3` learns what to pass.
  - SiLU is `a * sigmoid(a)`, not plain `sigmoid(a)`. The original GLU uses a
    sigmoid, so its gate lies in (0, 1); SiLU is a softer, unbounded gate.
- The elementwise multiply is cheap, but its operands are two different
  projections of `x`. The product is a second-order (multiplicative)
  interaction that `act(Wx)` can't express. A purely linear layer has no
  nonlinearity, and stacked linear layers collapse to one linear map.
- Empirically better (Shazeer 2020; adopted by LLaMA, PaLM) with no
  theoretical explanation in the paper.
- 3 matrices instead of 2, so shrink `d_ff` from `4·d_model` to ~`8/3·d_model`
  (rounded to a multiple of 64) to keep the parameter count equal.
- Gate vs attention: both are input-dependent multiplicative weights, but the
  gate does not mix across tokens or across units, and it is not normalized
  (no softmax). Think of it as a per-feature, per-token volume knob, not routing.

## RoPE (rotary positional embedding)

### Building blocks

- `torch.arange(n)` (one "r") returns `0..n-1` like `range`; end is exclusive.
  Dtype is int64 for int args, float32 if any arg is a float.
- Outer product: `a[:, None] * b[None, :]` (or `torch.outer(a, b)`) gives a
  table with `[i, k] = a[i] * b[k]`. `None` inserts a size-1 axis, so shapes
  `(m, 1) * (1, n)` broadcast to `(m, n)`.
- `.cos()` / `.sin()` are elementwise on a tensor, same as `torch.cos(t)`.

### Terminology and notation

| Symbol | Meaning |
| --- | --- |
| $d_k$ | Dimension of a query/key vector (even). Grouped into $d_k/2$ pairs. |
| $L$ | `max_seq_len`; only sizes the tables. |
| $i$ (or $m,n$) | Token position, $0\le i<L$. $m$ = query position, $n$ = key position. |
| $k$ | Integer index of a dimension pair, $k\in\{0,\dots,d_k/2-1\}$. Pair $k$ is $(x_{2k},x_{2k+1})$. |
| $\Theta$ | Base constant (e.g. $10000$). Not a frequency; it sets the range of frequencies. |
| $\varphi_k$ | **Frequency** of pair $k$: $\varphi_k=\Theta^{-2k/d_k}=1/\Theta^{2k/d_k}$. Radians added per token step. |
| $\theta_{i,k}$ | **Angle** at position $i$ for pair $k$: $\theta_{i,k}=i\,\varphi_k$. |
| $R(a)$ | $2\times2$ rotation matrix by angle $a$. |
| $q,u$ | Unrotated 2-vector slices of the query / key for one pair. |
| $q',u'$ | Rotated versions: $q'=R(m\varphi)q$, $u'=R(n\varphi)u$. |
| $n-m$ | Relative offset (distance) between key and query. |

- **Frequency** $\varphi_k$: how fast the angle grows per token position.
  Range: $\varphi_0=1$ (fastest) down to $\approx1/\Theta$ (slowest); never
  exactly $1/\Theta$ since $k<d_k/2$.
- **Period / wavelength**: $2\pi/\varphi_k$ tokens per full turn. $1/\varphi_k$
  alone is tokens per radian.
- High frequency distinguishes nearby tokens but wraps around; low frequency
  tracks long range (clock hands: second / minute / hour).
- Frequency depends on the pair index $k$, not the token position. Every token
  uses all frequencies. Fast-first ordering is just a convention.
- In code, name it `frequency = 1.0 / theta ** (k / d_k)`; it is $\varphi_k$
  itself, not $1/\varphi_k$.
- The assignment writes $\theta_{i,k}=i/\Theta^{(2k-2)/d}$ with $k$ starting at
  $1$; same thing as $i\,\varphi_k$ with $k$ starting at $0$.

### Implementation

- Precompute `cos[i, k]` and `sin[i, k]` tables once, shape
  `(max_seq_len, d_k/2)`, then index them with `token_positions`.
- The rotation matrix per position is block-diagonal with 2x2 blocks
  `[[c, -s], [s, c]]`. Don't build it (O(d_k^2), mostly zeros); apply it
  elementwise in O(d_k):

  ```python
  x_even, x_odd = x[..., 0::2], x[..., 1::2]
  out_even = x_even * cos - x_odd * sin
  out_odd  = x_even * sin + x_odd * cos
  ```

  then interleave back into the last dim.
- Apply to queries and keys only, not values.

### Rotation matrix

The rotation matrix $R(a)$ is the $2\times 2$ matrix that rotates any 2D
vector counterclockwise by angle $a$ (in radians), preserving its length:

$$
R(a) = \begin{pmatrix} \cos a & -\sin a \\ \sin a & \cos a \end{pmatrix}.
$$

- Its columns are where the basis vectors land: $(1,0)\mapsto(\cos a,\sin a)$
  and $(0,1)\mapsto(-\sin a,\cos a)$.
- Applying it to $v=(v_1,v_2)^\top$:
  $R(a)v = (v_1\cos a - v_2\sin a,\; v_1\sin a + v_2\cos a)^\top$.
- RoPE applies $R(i\,\varphi_k)$ to pair $k$ of the token at position $i$.
  Over the whole vector this is a block-diagonal matrix with blocks
  $R(i\,\varphi_1),\dots,R(i\,\varphi_{d_k/2})$.

### Two key facts about rotations

**Fact 1 (inverse is transpose).**
$$R(a)^\top = R(-a) = R(a)^{-1}.$$

Proof: $\cos(-a)=\cos a$ and $\sin(-a)=-\sin a$, so
$R(-a)=\begin{pmatrix}\cos a & \sin a\\ -\sin a & \cos a\end{pmatrix}=R(a)^\top$.
Rotating by $a$ then by $-a$ returns to the start.

**Fact 2 (rotations compose by adding angles).**
$$R(a)\,R(b) = R(a+b).$$

Proof: multiply the matrices and use the angle-addition formulas
$\cos(a+b)=\cos a\cos b-\sin a\sin b$ and
$\sin(a+b)=\sin a\cos b+\cos a\sin b$:
$$
R(a)R(b)=\begin{pmatrix}
\cos a\cos b-\sin a\sin b & -(\sin a\cos b+\cos a\sin b)\\
\sin a\cos b+\cos a\sin b & \cos a\cos b-\sin a\sin b
\end{pmatrix}=R(a+b).
$$

Consequence: $R(a)^\top R(b) = R(-a)R(b) = R(b-a)$.

### Why only relative position matters

Fix pair $k$ and let $\varphi=\varphi_k$. Let $q,u\in\mathbb{R}^2$ be the
unrotated pair-$k$ slices of the query (position $m$) and key (position $n$).
RoPE gives $q'=R(m\varphi)\,q$ and $u'=R(n\varphi)\,u$. This pair's
contribution to the attention score is

$$
\begin{aligned}
q'^\top u' &= \big(R(m\varphi)q\big)^\top R(n\varphi)u \\
&= q^\top R(m\varphi)^\top R(n\varphi)\,u \\
&= q^\top R(-m\varphi)\,R(n\varphi)\,u && \text{(Fact 1)}\\
&= q^\top R\big((n-m)\varphi\big)\,u && \text{(Fact 2)}.
\end{aligned}
$$

- The absolute positions cancel; only the offset $n-m$ remains. Shifting both
  tokens by the same amount leaves the score unchanged.
- The full score is the sum over pairs:
  $$\langle q',k'\rangle=\sum_{k=0}^{d_k/2-1} q_k^\top R\big((n-m)\varphi_k\big)u_k,$$
  which equals the ordinary dot product of the full rotated vectors.
- Intuition: a dot product depends on the angle between vectors; rotating
  both leaves only the difference of the rotations.

### Why the angle grows linearly with position

- The angle must be linear in position, $\theta_i = i\,\varphi$. Then
  $-m\varphi + n\varphi = (n-m)\varphi$ depends only on the offset. For a
  general $g(i)$ you would get $g(n)-g(m)$, which is not a function of $n-m$.
  "Larger angle at larger positions" is a side effect, not a goal.
- The sign doesn't matter ($-i\varphi$ just relabels the offset as $m-n$).
- Two tokens $d$ apart have their pair vectors turned by $d\,\varphi_k$
  relative to each other, which changes the dot product. That is how the model
  senses distance.

### Counting: frequencies vs. angles

- Frequencies: $d_k/2$ numbers $\varphi_k$, shared by all positions. Fixed
  constants, not learned.
- Angles: $L\cdot d_k/2$ numbers $\theta_{i,k}=i\,\varphi_k$ (one per position
  and pair), with $L$ = `max_seq_len`. This is the outer-product table.
- Rotation matrices: one $R(\theta_{i,k})$ per angle, stored as the `cos` and
  `sin` tables of shape $(L, d_k/2)$.
- `max_seq_len` only sizes the tables. The frequencies depend only on
  $\Theta$ and $d_k$.

### Periodicity and multiple frequencies

- A single pair is periodic: distances $d$ and $d+2\pi/\varphi_k$ give the same
  rotation, so it is ambiguous alone.
- Combined, the angle vector $(d\varphi_0,\dots,d\varphi_{d_k/2-1})$ identifies
  $d$ unambiguously, like clock hands: the second hand repeats, but together
  with the minute and hour hands the time is unique. Two distinct distances
  would need $(d-d')\varphi_k\in 2\pi\mathbb{Z}$ for every $k$ at once, which
  essentially never happens.
- Slowest frequency: $\varphi_{\min}=\Theta^{-(d_k-2)/d_k}\approx 1/\Theta$, so
  the longest wavelength is about $2\pi\Theta$ ($\approx 63{,}000$ tokens for
  $\Theta=10000$). Choose $\Theta$ so this is on the order of the context
  length or more, so the slow pairs give a clean long-range signal.
- Not a hard requirement, since the other pairs still disambiguate. But the
  model only works well on distances seen in training; extrapolating past the
  trained length degrades quality. Long-context models raise $\Theta$ (e.g.
  LLaMA 3 uses $500{,}000$) or use RoPE scaling (position interpolation, NTK,
  YaRN).
