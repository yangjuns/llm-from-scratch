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

### RMSNorm vs. LayerNorm

Both normalize each token's feature vector over `d_model`. RMSNorm skips the
mean-centering and the bias.

LayerNorm:

$$y = \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}\odot g + b,\quad
\mu=\tfrac1d\sum x_i,\ \sigma^2=\tfrac1d\sum (x_i-\mu)^2$$

RMSNorm:

$$y = \frac{x}{\sqrt{\tfrac1d\sum x_i^2+\epsilon}}\odot g$$

| | LayerNorm | RMSNorm |
| --- | --- | --- |
| Subtracts mean | yes | no |
| Learned parameters | gain and bias (`2d`) | gain only (`d`) |
| Statistics computed | mean and variance | mean of squares only |
| Output mean | ~0 (before `b`) | not forced to 0 |

- RMSNorm is cheaper (fewer ops and parameters) and performs about as well.
  The RMSNorm paper argues re-scaling is what stabilizes training and
  re-centering adds little. LLaMA and most recent LLMs use it.
- If the mean of `x` is already near 0, the two are almost identical, since
  then variance ≈ mean of squares.

### Why RMSNorm is named `ln`

- `ln` = layer norm. GPT-2 and the original Transformer use LayerNorm, with
  names like `ln_1`, `ln_2`, `ln_f`. Many models kept `ln` after switching to
  RMSNorm, since it is a simplified layer norm.
- The assignment fixtures use `ln1.weight`, `ln2.weight`, `ln_final.weight`.
- The name has no effect on behavior. It only matters that it matches the
  state-dict keys the adapter loads.

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

### Why `token_positions` is an input (custom positions)

Positions are separate data: `x` only tells you `seq_len`, not which positions
its tokens occupy. So `RoPE.forward(x, token_positions)` takes them as input,
with shape `(..., seq_len)` and integer values in $[0, L)$ (any order, repeats
allowed).

Use cases where tokens are not "positions $0..n-1$ of one sequence":

1. **Generation with a KV cache** (most common). Only the new token is fed, so
   `x` has `seq=1`, but its position is $t$ (the cache length), not 0. Cached
   keys already carry their own rotations, so the new query must use $t$.
2. **Chunked or continued processing.** A second chunk of a long text starts
   at its offset, e.g. `[512, 513, ...]`.
3. **Packed sequences.** Several short documents share one row to avoid
   padding. Positions restart at each boundary, e.g. `[0,1,2,3, 0,1,2, 0,1]`
   (usually with a mask blocking cross-document attention).
4. **Left-padded batches.** Positions are shifted so the first real token gets
   position 0, e.g. `[0,0,0,0,1,2,3]`.
5. **Context-extension tricks.** Position interpolation, skipping, or
   remapping use scaled or non-consecutive positions on purpose.
6. **Per-example layouts.** Each batch row can have its own positions, which
   is why the shape is `(..., seq)` and not just `(seq,)`.

Plain pretraining (full sequences starting at 0) needs none of these, so
`torch.arange(seq_len)` is enough. A reasonable design is
`token_positions=None` defaulting to `arange(seq_len)`; a passed-in value
always wins.

Multi-head does not change the position **values**: every head's $q$ and $k$
are rotated by the same position of its token. It only changes the **shape**:
after splitting heads, $q$ is `(batch, heads, seq, d_k)`, so positions
`(batch, seq)` need a head axis (`unsqueeze(-2)`) to broadcast. A `(seq,)`
tensor broadcasts without it.

Why not store positions on the module at construction: they describe the
input, not the model. A stored tensor fixes the sequence length and batch
shape, breaks on truncated or longer inputs, and can't express any case above.
Pass them in `forward` (or default them there).

## Numerical precision: what needs float32

Rule of thumb: matmuls and elementwise activations can run in the model
`dtype` (bf16/fp16); anything that builds index-like values, sums many terms,
or exponentiates should be computed in float32.

**RoPE tables.** Positions, frequencies, angles, cos and sin are computed in
float32, and only the finished tables are cast to `dtype`. bf16 has about 8
bits of mantissa, so it cannot distinguish integer positions 256 and 257
(fp16 has the same problem at larger positions). Both tokens would get the
same rotation. Rotations only need relative accuracy, so casting the finished
cos/sin down is fine.

**Needs float32:**
- RoPE position/angle computation (above).
- RMSNorm mean of squares: a sum over `d_model` loses precision in bf16, and
  `x^2` can overflow in fp16. Upcast, then cast the output back.
- Softmax: `exp` and the sum over a long sequence lose accuracy. Upcast, then
  cast back. Subtracting the max already prevents overflow.
- Cross-entropy / log-sum-exp.
- AdamW moments and master weights: small updates vanish in bf16 (about 3
  significant digits).

**Fine in `dtype`:** linear/embedding weights (hardware accumulates matmuls in
float32 anyway), activations (SiLU, SwiGLU products), and the cos/sin tables
after they are computed.

### Type promotion

When two tensors of different dtypes are combined, PyTorch takes the wider
type. In RMSNorm, `x_normed` is float32 and `self.weights` may be bf16, so
`x_normed * self.weights` is computed in float32. The weights are stored in
bf16 but used in float32.

### Should norm weights be stored in `dtype`?

Mostly no. They are only `d_model` values per norm, so float32 storage costs
nothing in memory. A learned parameter stored in bf16 can't represent small
updates, and upcasting in the forward pass doesn't recover bits already lost
in storage. Common practice (LLaMA-style, mixed precision) is to keep norm
weights in float32. Caveat: `model.to(torch.bfloat16)` casts them too, so
keeping them float32 needs autocast or a manual exception.

## Parameter and memory accounting (transformer_accounting a)

Config (GPT-2 XL, from the assignment): vocab 50,257, 48 layers,
d_model 1,600, 25 heads, d_ff 4,288 (nearest multiple of 64 to 8/3 x 1,600).

**Attention parameters.** Each head's $W_q$ maps `d_model -> d_k` with
`d_k = d_model / num_heads`, so it has `d_model x d_k` parameters (not
`d_k x d_k`). Across all heads each of $W_q, W_k, W_v, W_o$ has `d_model^2`
parameters, so attention totals `4 * d_model^2`, independent of head count.

**Per block**
- Attention: 4 x 1600^2 = 10,240,000
- SwiGLU ($W_1, W_2, W_3$): 3 x 1600 x 4288 = 20,582,400
- Two RMSNorms: 2 x 1600 = 3,200
- Block total: 30,825,600

**Whole model**
- Embedding: 50,257 x 1600 = 80,411,200
- 48 blocks: 1,479,628,800
- Final norm: 1,600
- LM head: 1600 x 50,257 = 80,411,200
- **Total: 1,640,452,800 (about 1.64B)**

(Weight tying would subtract one 80,411,200 table; the assignment model does
not tie.)

**Memory to load the weights**
- Single precision (float32) is **4 bytes** per parameter:
  1,640,452,800 x 4 = 6,561,811,200 bytes, about 6.56 GB or 6.11 GiB.
- Half precision (fp16/bf16) is 2 bytes: about 3.28 GB or 3.06 GiB.

Easy to mix up: "single precision" is 32-bit (4 bytes), not 2 bytes. Two bytes
is half precision. Training needs far more than the weights alone (gradients,
AdamW moments, activations).

### Weight tying

Weight tying uses **one matrix for both the input embedding and the output
(LM head) projection**.

- Embedding: token id -> vector, a lookup into a `(vocab_size, d_model)` table
  `E`.
- LM head: final hidden vector -> logits over the vocabulary, a
  `(d_model, vocab_size)` matrix.

The two have the same shape up to a transpose, so with tying the logits are
`h @ E.T`. In PyTorch: `lm_head.weight = embedding.weight`.

Why do it:
- Saves `vocab_size x d_model` parameters (80,411,200 for GPT-2 XL, about 5%
  of the model; a larger share for small models with big vocabularies).
- Mild regularization: the same vector must work for both reading a token in
  and predicting it out (Press & Wolf, 2017).
- GPT-2 and many small LMs tie weights; many large modern models (e.g. LLaMA)
  do not.

Trade-offs: gradients from both uses flow into one matrix, the initialization
must suit both roles, and the input and output sides can't be tuned
independently. The assignment model does **not** tie, so both tables are
counted in the parameter total.

## Forward-pass FLOPs (transformer_accounting b, c)

Rule: a matmul `(m, k) @ (k, n)` costs `2 * m * k * n` FLOPs (one multiply
and one add per term). Only matmuls are counted.

Config: seq 1,024, d_model 1,600, 25 heads (d_k = 64), d_ff 4,288, 48 layers,
vocab 50,257.

**Per block**
| Matmul | Formula | FLOPs |
|---|---|---|
| Q, K, V projections | 3 x 2 seq d^2 | 15,728,640,000 |
| QK^T and scores @ V | 4 seq^2 d | 6,710,886,400 |
| W_o projection | 2 seq d^2 | 5,242,880,000 |
| SwiGLU (W1, W2, W3) | 3 x 2 seq d d_ff | 42,152,755,200 |
| **Block total** | | **69,835,161,600** |

**Whole model**
- 48 blocks: 3,352,087,756,800
- LM head: 2 x seq x d x vocab = 164,682,137,600
- **Total: 3,516,769,894,400 FLOPs (about 3.52 TFLOPs)**

**Common mistakes**
- SwiGLU has **three** matmuls (W1, W3: `d -> d_ff`; W2: `d_ff -> d`), not two.
- Attention scores are per head `seq x seq`. `QK^T` is
  `(seq, d_k) @ (d_k, seq)` = `2 seq^2 d_k`, and `scores @ V` is
  `(seq, seq) @ (seq, d_k)` = `2 seq^2 d_k`. Over `h` heads that sums to
  `4 seq^2 d`.
- Per-head Q/K/V cost `2 seq d d_k` each; times `h` heads gives `2 seq d^2`,
  so head count does not change the projection FLOPs.

**Which parts dominate (c)**
| Component | FLOPs | Share |
|---|---|---|
| SwiGLU FFN | 2,023,332,249,600 | 57.5% |
| Q/K/V/O projections | 1,006,632,960,000 | 28.6% |
| Attention scores (QK^T, @V) | 322,122,547,200 | 9.2% |
| LM head | 164,682,137,600 | 4.7% |

The FFN dominates, followed by the attention projections. The `seq^2` score
matmuls are small at seq 1,024, but they grow quadratically with context
length while everything else grows linearly.

## FLOPs across model sizes and context length (transformer_accounting d, e)

`d_ff` is the nearest multiple of 64 to 8/3 x d_model; seq 1,024, vocab
50,257.

| Model | Layers | d_model | Heads | d_ff | Total FLOPs |
|---|---|---|---|---|---|
| Small | 12 | 768 | 12 | 2,048 | 291,648,307,200 (0.29T) |
| Medium | 24 | 1,024 | 16 | 2,752 | 830,172,299,264 (0.83T) |
| Large | 36 | 1,280 | 20 | 3,392 | 1,768,530,903,040 (1.77T) |
| XL | 48 | 1,600 | 25 | 4,288 | 3,516,769,894,400 (3.52T) |

**Share of total FLOPs (d)**
| Model | FFN (SwiGLU) | Q/K/V/O projections | Attention scores | LM head |
|---|---|---|---|---|
| Small | 39.8% | 19.9% | 13.3% | 27.1% |
| Medium | 50.1% | 24.8% | 12.4% | 12.7% |
| Large | 54.3% | 27.3% | 10.9% | 7.5% |
| XL | 57.5% | 28.6% | 9.2% | 4.7% |

Takeaway: as the model grows, the FFN and projection shares rise because
their cost scales with `d_model^2` and with layer count. The LM head share
falls sharply because it scales only with `d_model`. The attention-score
share falls slowly because `seq^2 * d_model` grows more slowly than
`d_model^2` when seq is fixed.

**GPT-2 XL at context length 16,384 (e)**

Total rises from 3.52T to about 133.6T FLOPs, roughly 38x for a 16x longer
context. Everything except the attention scores is linear in seq (16x); the
scores are quadratic (256x), so they take over:

| Component | Share at 1,024 | Share at 16,384 |
|---|---|---|
| Attention scores | 9.2% | 61.7% |
| SwiGLU FFN | 57.5% | 24.2% |
| Q/K/V/O projections | 28.6% | 12.1% |
| LM head | 4.7% | 2.0% |

Long contexts make attention the bottleneck, which is why FlashAttention and
sparse or linear attention variants matter.
