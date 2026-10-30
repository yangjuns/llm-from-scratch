# Tensor Operations Learning Notes

## How to read tensor code

- Track the **shape**, not the values. Write the shape after every line:
  `x_even = x[..., 0::2]  # (..., seq_len, d_k/2)`.
- For each op ask: what shape goes in, what shape comes out?
- Test on a tiny example and print values and shapes.

### What a shape comment means

A shape is a tuple with one entry per axis, in order. Each entry is the
**actual size of that axis at that moment** (never a "maximum").

- Symbolic names (`seq_len`, `d_k`, `d_k/2`) stand for a size set elsewhere.
- Literal numbers (`2`) are for fixed sizes.
- `...` means "any number of leading axes I don't care about".
- Names and numbers are two ways to write the same thing. In einops, a name
  labels an axis and must mean the same size everywhere.
- `max_seq_len` is an upper bound that sizes a cache; the real `seq_len` of an
  input can be anything up to it.

## Axes and `dim`

`dim` is the axis number. Count from the left starting at 0; negative counts
from the right. For shape `(2, 3, 4)`:

| dim | meaning | size |
| --- | --- | --- |
| `0` | first axis | 2 |
| `1` | second axis | 3 |
| `2` or `-1` | last axis | 4 |
| `-2` | second-to-last | 3 |

- Negative dims don't depend on the number of leading batch axes, so `dim=-1`
  is always the feature axis.
- Effect of `dim` depends on the op: reductions (`sum`, `mean`, `norm`)
  collapse that axis; `stack` inserts a new axis there; `cat` joins along it.

## Printing and memory order

- Printed tensor: the outermost bracket is dim 0, the innermost bracket (one
  row of numbers) is the **last** dim. Count the leading `[` to get the number
  of dims.
- Row-major order: the last axis varies fastest. `flatten` and printing both
  read in this order.

```python
t = torch.arange(12).reshape(2, 3, 2)
# [[[ 0,  1], [ 2,  3], [ 4,  5]],
#  [[ 6,  7], [ 8,  9], [10, 11]]]
```

## Creating tensors

- `torch.arange(n)`: `0..n-1`, end exclusive. int64 for int args, float32 if
  any arg is a float. `torch.arange(2, 10, 2)` gives `[2, 4, 6, 8]`.
- `torch.ones`, `torch.zeros`: pass `device=` and `dtype=`.

## Indexing, slicing, and lookup (and how they change shape)

Rule of thumb: each axis is handled by one index expression, left to right.
What you put on an axis decides what happens to that axis:

| On an axis you write | Effect on that axis |
| --- | --- |
| an int, `3` | axis is **removed** |
| a slice, `:n`, `1:3`, `::2` | axis is **kept**, size shrinks |
| an int tensor, `idx` | axis is **replaced** by `idx`'s shape |
| `None` | a new size-1 axis is **inserted** |
| `:` | axis is kept unchanged |
| `...` | all remaining axes kept unchanged |

Axes you don't mention are kept unchanged (trailing axes).

### Int and slice (shape examples, `t: (L=5, D=4)`)

| Expression | Result shape | Note |
| --- | --- | --- |
| `t[3]` | `(4,)` | row 3 only, axis 0 removed |
| `t[:3]` | `(3, 4)` | rows 0, 1, 2 |
| `t[1:3]` | `(2, 4)` | rows 1, 2 (end exclusive) |
| `t[::2]` | `(3, 4)` | every second row |
| `t[:, 0]` | `(5,)` | column 0, axis 1 removed |
| `t[:, :2]` | `(5, 2)` | first two columns |

- `[n]` is **not** `[:n]`: one row (dimension removed) vs. the first `n` rows
  (dimension kept).
- For `x: (2, 3, 6)`: `x[0]` is `(3, 6)`, `x[:, 0]` is `(2, 6)`,
  `x[..., 0]` is `(2, 3)`, `x[..., 0::2]` is `(2, 3, 3)`, `x[:, None]` is
  `(2, 1, 3, 6)`.
- Slices are views of the same memory (no copy); writing to a slice changes
  the original.

### Lookup (indexing with an int tensor)

Indexing an axis with an int tensor `idx` **gathers** entries, and the axis
is replaced by `idx`'s whole shape:

```
result.shape = idx.shape + remaining axes
```

| Expression (`t: (5, 4)`) | `idx` shape | Result shape |
| --- | --- | --- |
| `t[idx]` | `(3,)`, e.g. `[4, 0, 4]` | `(3, 4)` |
| `t[idx]` | `(2, 3)` | `(2, 3, 4)` |

- `t[idx][a]` equals `t[idx[a]]`: each index value picks one whole row.
- Repeated and out-of-order indices are fine; values must be in `[0, L)`.
- Gathers return a copy.
- Value of an index decides **which** row; shape of `idx` decides the
  **output shape**.

### Slice vs. lookup

| | Slice `cos[:seq_len]` | Lookup `cos[token_positions]` |
| --- | --- | --- |
| Uses | how many rows (a count) | which rows (actual values) |
| Result shape (`cos: (L, D)`) | `(seq_len, D)` | `(*token_positions.shape, D)` |
| Assumes positions are | `0..seq_len-1` in order | anything in `[0, L)` |
| Copy? | view | copy |

- RoPE: `cos_cached` is `(max_seq_len, d_k/2)`. With
  `token_positions: (..., seq_len)`,
  `cos_cached[token_positions]` is `(..., seq_len, d_k/2)`, which lines up
  with `x`'s batch and sequence axes. A slice would drop the batch dims and
  ignore the actual position values.
- `[:token_positions]` (slice with a tensor bound) is a bug; use
  `[token_positions]`.

## Elementwise ops and broadcasting

- `+ - * / **` and `.cos()`, `.sin()`, `.sqrt()`, `.pow()` act per element.
- `a * b` is elementwise, **not** a matrix multiply.

### Broadcasting rule

To combine two tensors elementwise:

1. **Align shapes from the right** (last axes together).
2. If one shape is shorter, pad it with 1s on the **left**.
3. For each axis, the two sizes must be **equal**, or one of them must be
   **1**. A size-1 axis is stretched (virtually copied) to match the other.
4. The result size on each axis is the larger of the two.

Examples (`->` is the result shape):

| A | B | Result | Why |
| --- | --- | --- | --- |
| `(2, 3)` | `(3,)` | `(2, 3)` | B becomes `(1, 3)` |
| `(8, 5, 4)` | `(4,)` | `(8, 5, 4)` | B becomes `(1, 1, 4)` (RMSNorm weight) |
| `(3, 1)` | `(1, 4)` | `(3, 4)` | both stretch: outer product |
| `(8, 5, 4)` | `(5, 1)` | `(8, 5, 4)` | B becomes `(1, 5, 1)` |
| `(2, 1, 3)` | `(4, 1)` | `(2, 4, 3)` | B becomes `(1, 4, 1)` |
| `(2, 3)` | `(2,)` | **error** | aligned right: 3 vs 2 |
| `(8, 5, 4)` | `(5,)` | **error** | aligned right: 4 vs 5 |
| `(2, 3)` | `(4, 3)` | **error** | 2 vs 4, neither is 1 |

- Check a pair of shapes quickly with `torch.broadcast_shapes(a, b)`.
- Broadcasting aligns from the **right**, so a per-feature vector `(d,)` works
  against `(..., d)`, but a per-row vector `(n,)` against `(n, d)` does not.
  Fix with `v[:, None]` (shape `(n, 1)`) or `keepdim=True`.

### Making shapes line up

- `None` (or `unsqueeze`) inserts a size-1 axis: `p[:, None]` is `(m, 1)`,
  `f[None, :]` is `(1, n)`, so `p[:, None] * f[None, :]` is `(m, n)`.
- `keepdim=True` on a reduction keeps the reduced axis as size 1:
  `x.mean(-1)` on `(8, 5, 4)` is `(8, 5)` and `x - x.mean(-1)` **fails**;
  `x.mean(-1, keepdim=True)` is `(8, 5, 1)` and `x - ...` works.
- Broadcasting never changes data; it just reuses it. Output shape comes from
  the rule above, so you can predict it before running.

### Debugging shape errors

1. Write both shapes, right-aligned, one above the other.
2. Check each column: equal or one of them 1?
3. The first bad column is the axis to fix (`None`, `keepdim`, `permute`).

## 7-day tensor practice plan

30-45 minutes a day, tiny tensors on CPU (no GPU or big model needed). Print
shapes and values after every step.

| Day | Topic | You should be able to |
| --- | --- | --- |
| 1 | Shape, indexing, slicing | Understand `x[:, 0, :]`, `x[..., -1]` |
| 2 | Reshape and permute | Understand `reshape`, `view`, `transpose`, `permute` |
| 3 | Combining tensors | Understand `stack`, `cat`, `unsqueeze`, `squeeze` |
| 4 | Broadcasting | Predict how two shapes interact in elementwise operations |
| 5 | Reductions and matrix multiplication | Understand `sum`, `mean`, `@`, `matmul` |
| 6 | Tensor puzzles | Implement simple functions without explicit loops |
| 7 | Mini attention implementation | Reason about `(B, H, T, D)` transformations |

## Reductions

- `x.mean(dim=-1)` takes `(2, 3, 4)` to `(2, 3)`.
- `keepdim=True` keeps the axis with size 1 (`(2, 3, 1)`), so the result still
  broadcasts against `x` (used in RMSNorm).

## Matrix multiply and einsum

- `a @ b` / `torch.matmul`: matrix multiply; leading batch dims broadcast.
- `einsum` labels each axis with a letter. A letter that appears in the inputs
  but **not** in the output is summed over.

| Expression | Meaning |
| --- | --- |
| `"i,k->ik"` | outer product |
| `"i,i->"` | dot product |
| `"...i,oi->...o"` | Linear layer $y = xW^\top$ |

## Outer product

- For $a\in\mathbb{R}^m$, $b\in\mathbb{R}^n$: the $m\times n$ table with
  entries $a_i b_k$, i.e. $ab^\top$ (rank 1).
- Three equivalent ways:

  ```python
  torch.einsum("i,k->ik", a, b)
  torch.outer(a, b)
  a[:, None] * b[None, :]
  ```

- `outer` and `einsum` need matching dtypes; broadcasting promotes
  automatically.

## `stack`, `cat`, `flatten`

### `torch.stack(tensors, dim)`

Joins same-shape tensors along a **new** axis inserted at position `dim`.

- Rule: with new-axis index $j$, `result[..., j, ...] = tensors[j][...]`.
  $j$ is the index along the new axis and says which input to take from.
- Example, `even = [10, 11]`, `odd = [20, 21]`:

  ```
  stack(dim=0)  -> [[10, 11],    each input is a row
                    [20, 21]]
  stack(dim=-1) -> [[10, 20],    row k = (even[k], odd[k])
                    [11, 21]]
  ```

- With `dim=-1`: `result[k, j] = (even, odd)[j][k]`, so corresponding
  elements of the inputs end up side by side.

### `torch.cat(tensors, dim)`

Joins along an **existing** axis; the number of dims is unchanged.
`cat((a, b))` of two `(3,)` gives `(6,)`; `stack((a, b))` gives `(2, 3)`.

### `tensor.flatten(start_dim, end_dim=-1)`

Merges consecutive axes into one (size = product), in row-major order. The
data order is unchanged; only the shape changes.

```python
t = torch.arange(12).reshape(3, 2, 2)
t.flatten()      # (12,)
t.flatten(-2)    # (3, 4)   last two axes merged
t.flatten(0, 1)  # (6, 2)
```

### Interleaving trick (RoPE)

Pair `(2k, 2k+1)` is rotated, then written back to the same slots:

```python
out = torch.stack([out_even, out_odd], dim=-1)  # (..., seq_len, d_k/2, 2)
out = out.flatten(-2)                           # (..., seq_len, d_k)
```

- `stack(dim=-1)` puts even/odd side by side within each pair; `flatten` reads
  the last axis fastest, so the result is `e0, o0, e1, o1, ...`.
- With `dim=0` you would get `e0, e1, ..., o0, o1, ...` instead, which is the
  wrong layout.
- The order matters: later layers and weights assume each feature index keeps
  its meaning.

## `nn.Module`

- `M(x)` calls `Module.__call__`, which runs hooks and then `forward`.
  `M.forward(x)` skips the hooks, so always use `M(x)`.
