# Building an OpenWebText model on one RTX 4090

## Goal and status

Find a good OpenWebText language model that can be trained on one 24 GB
RTX 4090. "Best" depends on the available training time and token budget,
not just the largest model that fits in memory. This note records decisions
and short memory pilots, not a completed full OpenWebText training run.

The TinyStories baseline used 22.7M parameters, a 256-token context, batch
size 64, FP32, and a peak learning rate of `1e-3`. Its results are described
in the [experiment notes](a1-llm-experiments.md). OpenWebText is a different,
more varied corpus, so its best model size and learning rate still need to
be measured.

## Decision 1: a 1,024-token context

Use **1,024 tokens** as the initial context length. Longer context lets a
prediction use information from farther back in the current window than the
256-token TinyStories model. This may help with longer articles and other web
documents, but does not guarantee that the model will learn to use distant
information or memorize a document. Context is temporary input, not a way to
store facts permanently in model weights.

### What the data says

The existing OpenWebText tokenizer has a 32,000-token vocabulary. On 10
sampled validation documents it encoded about 4.245 UTF-8 bytes per token;
this is only a rough average and varies by document. See the
[tokenizer comparison](a1-bpe-tinystories-vs-owt.md).

The table below was measured from `assignment1-basics/data/tokenized/owt_valid.npy`.
The array contains 66,401,098 tokens. The existing tokenizer assigns
`<|endoftext|>` ID 0. Lengths are the number of tokens between consecutive
separator IDs, excluding the separators and the partial segments before the
first and after the last separator. There are 59,058 such complete segments.

| Percentile | Document length in tokens |
|---|---:|
| 10th | 271 |
| 25th | 415 |
| 50th | 714 |
| 75th | 1,240 |
| 90th | 2,125 |
| 95th | 3,179 |

About **66.8%** of these documents are no longer than 1,024 tokens. The
75th percentile is **1,240**, not 1,024; the earlier suggestion that 1,024
was the 75th percentile was incorrect. We still choose 1,024 as a practical
compromise between longer context and GPU cost. These are validation-set
document statistics, not a measurement of every training document.

A document shorter than the context does not necessarily appear whole in a
training example. The current
[data loader](../assignment1-basics/cs336_basics/load_data.py) samples random
windows from a continuous token stream, and a window can cross a document
separator. For any prediction, the causal mask limits attention to earlier
tokens in that window. Longer documents must be seen in separate windows;
the model cannot attend to tokens outside the current window.

### Cost of the longer context

The current attention implementation explicitly builds attention scores with
shape `(batch, heads, context, context)` for each layer. Doubling the context
from 512 to 1,024 therefore makes those score tensors four times as large for
the same physical batch size. Other activations and output logits also consume
memory. A longer context can require a smaller physical batch and more time.

RoPE uses precomputed position tables rather than a learned position embedding
per context position, so increasing context does not itself increase the
number of learned parameters in this model. It does increase computation and
activation memory. At generation time, this window is also a limit on the
tokens available to a prediction, not a promise of reliable long-range recall.

## Decision 2: explore BF16 mixed precision

Use **BF16 mixed precision as a candidate** for reducing activation memory and
accelerating suitable matrix operations on the 4090. BF16 uses 2 bytes per
element instead of FP32's 4, with a wide exponent range but less precision.
That does **not** halve total GPU memory: FP32 parameters, gradients,
optimizer states, temporary tensors, and allocator overhead still matter.

The proposed approach is mixed precision rather than simply converting every
parameter and operation to BF16:

- Keep model parameters and AdamW moment estimates in FP32 so that small
  updates and accumulated statistics retain precision.
- Use BF16 autocast for eligible forward operations; retain FP32 for sensitive
  reductions such as normalization, softmax, and loss calculation. RoPE
  positions and angle construction also need sufficient precision.
- BF16's exponent range generally avoids the need for the gradient scaling
  commonly used with FP16, but finite losses and gradients still need checking.

The [training script](../assignment1-basics/cs336_basics/train.py) supports
`--dtype bfloat16` for training and evaluation, while defaulting to FP32.
The model computes RMSNorm, attention softmax, and RoPE angles in FP32 in
relevant places; the training script also converts logits to FP32 for the
custom cross-entropy. More detail on precision is in the
[implementation notes](a1-llm-implementation.md).

Before a full run, compare FP32 and BF16 on a small fixed batch: check finite
loss and gradients, compare losses and updates within reasonable tolerances,
and measure peak allocated and reserved GPU memory, tokens per second, and
validation loss. A successful short test is not proof of long-run stability
or a measurement of BF16 speedup over FP32.

## Decision 3: model shape and training budget

We compared short BF16 pilots at context 1,024 on the 4090. Reported memory
figures below are the user's observed PyTorch peak allocated memory in decimal
GB, not reserved or total device memory. Pilot results are not validation-loss
comparisons, and the reported training peak may not include final evaluation.

| Layers | Width | Heads | `d_ff` | Parameters | Physical batch | Peak allocated |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 768 | 12 | 2,048 | 134,105,856 | 8 | 18.60 GB |
| 12 | 768 | 12 | 2,048 | 134,105,856 | 16 | Out of memory |
| 16 | 768 | 12 | 2,048 | 162,423,552 | 8 | 23.11 GB |
| 12 | 864 | 12 | 2,304 | 162,813,024 | 8 | 19.72 GB |
| 12 | 960 | 12 | 2,560 | 194,174,400 | 8 | 20.84 GB |
| **12** | **1,056** | **12** | **2,816** | **228,189,984** | **8** | **about 22 GB** |

The 16-layer and 864-wide models have nearly the same parameter count, but
the wider pilot used less peak allocated memory. We then tested wider
12-layer models while keeping 12 heads and the `8/3` SwiGLU width. We chose
**12 layers and width 1,056** as a practical candidate that fit the short
pilot, not as a demonstrated optimum. Compare validation loss at matched
token and time budgets before claiming it is better than another shape.

[CS336 lecture 3](https://github.com/stanford-cs336/lectures/blob/main/lecture_03.pdf)
defines aspect ratio as `d_model / num_layers` and summarizes roughly 100-200
as a wide range of reasonable values, subject to systems considerations.
The chosen model has aspect ratio 88; this heuristic comes from other models
and is not a requirement. Its 88-dimensional heads are even for RoPE.
Width 1,056 is divisible by 32, and its feed-forward width 2,816 by 128;
dimensions need not be powers of two, but actual throughput must be measured.

The chosen full-run plan is BF16 autocast, a micro-batch of 8 and 8
accumulation steps, yielding 65,536 tokens per optimizer update.
The 70,000-update schedule processes **4,587,520,000 sampled tokens**,
about 20.1 times the model's parameter count and 1.68 times the tokenized
training-array size. Twenty tokens per parameter is a planning heuristic,
not a requirement; random window sampling does not guarantee complete
coverage. At the observed approximately 3 seconds per update, training
takes roughly 58 hours 20 minutes, excluding evaluation, checkpoints,
startup, and interruptions. Peak learning rate `1e-3` remains a candidate
based on TinyStories rather than a verified OpenWebText optimum.

For a long run, checkpoint space must be planned before starting. The
current script saves a numbered checkpoint and a separate `latest.pt` copy;
it retains numbered checkpoints. For 228M FP32 parameters and two FP32 Adam
moments, one checkpoint is roughly 2.74 GB before serialization overhead,
and each new numbered checkpoint plus latest copy needs roughly 5.48 GB
initially; additional numbered checkpoints add roughly 2.74 GB each.
`--no-checkpoint` avoids those writes but produces no saved model and offers
no resume after interruption. Check available disk or a mounted volume first.

### Full-run command

The selected plan saves every 5,000 updates and logs to the existing W&B
project. With 70,000 updates, 14 numbered checkpoints plus `latest.pt` will
occupy roughly 41 GB in total, before serialization overhead and other
files; W&B metric logging alone does not save the model. Confirm enough disk
space before starting or point `--checkpoint-dir` at a suitable volume.
Authenticate W&B on the VM before launching; do not put credentials in the
command or notes.

From `assignment1-basics/`:

```sh
uv run python -m cs336_basics.train \
  --train-data data/tokenized/owt_train.npy \
  --valid-data data/tokenized/owt_valid.npy \
  --device cuda --dtype bfloat16 \
  --vocab-size 32000 --context-length 1024 \
  --d-model 1056 --num-layers 12 --num-heads 12 --d-ff 2816 \
  --batch-size 8 --gradient-accumulation-steps 8 \
  --max-lr 1e-3 --min-lr 1e-4 --warmup-iters 200 \
  --max-iters 70000 --log-interval 10 \
  --eval-interval 250 --eval-batches 20 \
  --checkpoint-interval 5000 \
  --checkpoint-dir checkpoints/owt-228m-bf16 \
  --wandb-project cs336-assignment1 \
  --wandb-entity yangjuns-cmu-google \
  --run-name owt-228m-bf16
```

This command has not yet been run as a full training job. It uses the
trainer's defaults for AdamW betas, epsilon, weight decay, gradient clipping,
RoPE, RMSNorm, and seed, as listed in the earlier training table. To resume,
keep the same model, optimizer and schedule arguments, add `--resume` with
the checkpoint path, and use the W&B run ID with `--wandb-run-id` if resuming
the same run. Checkpoint resume does not restore random generator states.

## Initial 134M candidate and memory estimates

The initial candidate was 12 layers, width 768, 12 heads,
feed-forward width 2,048, and the 32,000-token OpenWebText vocabulary. With
the current separate input embeddings and output head, this is **134,105,856
parameters**. The tokenized training set has 2,727,120,453 tokens, roughly
20 times that parameter count. This ratio is a rough planning heuristic, not
a guarantee of optimal training; random window sampling also means that
processing that many tokens does not imply exactly one pass over the corpus.

### Proposed model parameters

| Hyperparameter | Value | Rationale |
|---|---:|---|
| Vocabulary size | 32,000 | Existing OpenWebText BPE tokenizer |
| Context length | 1,024 tokens | Longer context at a manageable estimated cost |
| Transformer layers | 12 | Initial size candidate |
| Model width (`d_model`) | 768 | Initial size candidate |
| Attention heads | 12 | 64 dimensions per head |
| Feed-forward width (`d_ff`) | 2,048 | Exactly `8/3 * d_model` |
| Activation | SwiGLU | Existing architecture |
| Normalization | RMSNorm | Existing architecture |
| Position encoding | RoPE, theta 10,000 | Existing default |
| Input and output embeddings | Separate | Existing architecture; not weight-tied |
| Total learned parameters | 134,105,856 | Includes both embedding matrices |

With vocabulary size `V`, model width `d`, feed-forward width `d_ff`, and
layer count `L`, the parameter count is
`2*V*d + L*(4*d^2 + 3*d*d_ff + 2*d) + d`. Context length does not add
learned parameters to this RoPE architecture.

### Proposed training parameters

| Hyperparameter | Initial value | Status |
|---|---:|---|
| Precision | BF16 autocast | Candidate; parameters and optimizer state stay FP32 |
| Physical micro-batch | 8 sequences | Initial pilot configuration |
| Gradient accumulation | 8 micro-batches | Effective batch of 64 sequences |
| Tokens per optimizer update | 65,536 | `8 * 8 * 1024` |
| Optimizer | AdamW | Existing implementation |
| Peak / minimum learning rate | `1e-3` / `1e-4` | Starting candidates, not an OpenWebText sweep result |
| Learning-rate schedule | Linear warmup, then cosine decay | Existing implementation |
| Warmup | 200 optimizer updates | Existing default |
| Adam beta1 / beta2 | `0.9` / `0.95` | Existing defaults |
| Adam epsilon | `1e-8` | Existing default |
| Weight decay | `0.1` | Existing default |
| Gradient clipping | `1.0` | Existing default |
| Schedule horizon | 40,000 optimizer updates | Provisional, not an optimized budget |
| Random seed | 0 | Existing default |

The pilot below stops after 10 updates, logs every update, and evaluates on
two micro-batches at the end. Its schedule horizon is still 40,000 updates.

### Batch size and memory estimate

The TinyStories batch of 64 at context 256 represented 16,384 tokens per
update. At context 1,024, an effective batch of 64 is 65,536 tokens per
update, so the TinyStories result does not establish that it is optimal.
We will start with 64 as a hypothesis and use gradient accumulation to keep
the physical batch smaller.

The following illustrative subtotals use the simplified activation accounting
in the [implementation notes](a1-llm-implementation.md#adamw-peak-memory-adamw_accounting-a).
They add FP32 model parameters, gradients, and two Adam moments (1.998 GiB),
transformer-layer activations at an assumed 2 to 4 bytes per counted value,
an FP32 final-normalization output, and a BF16 logits tensor plus two FP32
loss-sized tensors. Layer activations and output-related terms scale with
micro-batch; the parameter-state estimate does not.

For the 134M candidate at micro-batch 4, context 1,024, 12 heads, and
vocabulary 32,000, the calculation breaks down as follows (GiB = 2^30 bytes):

| Component | Calculation | Approximate size |
|---|---|---:|
| FP32 parameters | `134,105,856 * 4` bytes | 0.50 GiB |
| FP32 gradients | `134,105,856 * 4` bytes | 0.50 GiB |
| FP32 AdamW moments | `134,105,856 * 2 * 4` bytes | 1.00 GiB |
| Transformer-layer activations | `12 * ((56/3)*B*T*d + 2*B*h*T^2)` values, at 2-4 bytes each | 3.56-7.13 GiB |
| Final normalization output | `B*T*d * 4` bytes | 0.01 GiB |
| BF16 logits and two FP32 loss tensors | `B*T*V * (2 + 4 + 4)` bytes | 1.22 GiB |
| **Illustrative subtotal** | Sum of above components | **6.79-10.36 GiB** |

Here `B=4`, `T=1024`, `d=768`, `h=12`, and `V=32000`. The activation
formula uses `d_ff = 8d/3`; using 2-4 bytes per counted value illustrates
the range of element widths, not a measured mixed-precision allocation or
a formal memory bound. These components need not all be live at once.

Scaling the same accounting by micro-batch gives:

| Micro-batch | Accumulation steps | Effective batch | Illustrative subtotal |
|---:|---:|---:|---:|
| 4 | 16 | 64 | 6.8-10.4 GiB |
| 8 | 8 | 64 | 11.6-18.7 GiB |
| 16 | 4 | 64 | 21.2-35.4 GiB |

These are **not measured peaks or upper bounds**. The accounting simplifies
tensor lifetimes and omits some intermediates, temporary optimizer
allocations, CUDA workspace, and allocator overhead. The model's attention
materializes quadratic matrices and performs softmax in FP32, and its loss
uses FP32 logits. Therefore, BF16 does not halve the actual memory usage.
Increasing micro-batch mainly targets throughput; it does not change model
parameters or effective batch.

Start with micro-batch **8**, then try 16 if the measured peak leaves adequate
headroom. If 8 runs out of memory, try 4 with 16 accumulation steps. Compare
throughput as well as memory. The training log's peak is PyTorch-allocated
memory in decimal GB, not reserved or total device memory; inspect device
usage separately. A short pilot exercises several full forward, backward, and
optimizer steps, plus a final evaluation. Pass `--no-checkpoint` if no
checkpoint is needed and disk space is limited.

### Initial 4090 pilot

From `assignment1-basics/`, after preparing the tokenized OpenWebText data:

```sh
uv run python -m cs336_basics.train \
  --train-data data/tokenized/owt_train.npy \
  --valid-data data/tokenized/owt_valid.npy \
  --device cuda --dtype bfloat16 \
  --vocab-size 32000 --context-length 1024 \
  --d-model 768 --num-layers 12 --num-heads 12 --d-ff 2048 \
  --batch-size 8 --gradient-accumulation-steps 8 \
  --max-lr 1e-3 --min-lr 1e-4 --warmup-iters 200 \
  --max-iters 40000 --stop-step 10 \
  --log-interval 1 --eval-batches 2 \
  --checkpoint-dir checkpoints/owt-134m-bf16
```

The 40,000-update schedule is a provisional planning horizon, not an
optimized training budget; it represents 2,621,440,000 sampled tokens if
completed. `--stop-step 10` stops and checkpoints after ten updates without
compressing that schedule. The pilot has not been run on a 4090. To continue,
use the same configuration and schedule, remove `--stop-step 10`, and add
`--resume checkpoints/owt-134m-bf16/latest.pt`. Checkpoints do not store the
training arguments or random generator states; resuming does not reproduce an
uninterrupted run's exact sample sequence.

Initial-pilot next steps (superseded by the chosen candidate above):

1. Measure the candidate on the 4090 at context 1,024, starting with
   micro-batch 8. Include the optimizer step in peak-memory measurements.
2. Compare FP32 and BF16 before selecting a precision and physical batch.
   The training loop supports gradient accumulation; for example, a physical
   batch of 4 and 16 accumulation steps produce an effective batch of 64.
   Whether that physical batch fits the candidate model remains unmeasured.
3. Choose a time or token budget and compare model sizes and learning rates at
   matched budgets. The TinyStories `1e-3` result is a starting candidate,
   not evidence that it is optimal for the larger OpenWebText model.
4. Track validation loss, throughput, memory, and qualitative generations.
   Compare contexts at matched model and token budgets if testing whether
   1,024 improves quality enough to justify its cost.
