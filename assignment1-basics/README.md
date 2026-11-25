# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

### Pre-tokenize data
After training the TinyStories and OpenWebText BPE tokenizers and downloading
the datasets, encode each split into a one-dimensional `uint16` NumPy array.
Chunks are split at `<|endoftext|>` boundaries and processed in parallel:

```sh
uv run python -m cs336_basics.tokenize_dataset --workers 6
```

The arrays are saved under `data/tokenized/` as `tinystories_train.npy`,
`tinystories_valid.npy`, `owt_train.npy`, and `owt_valid.npy`. They can be
loaded without reading the full arrays into memory using
`np.load(path, mmap_mode="r")`.

### Training precision

Training and evaluation default to FP32. Pass `--dtype bfloat16` to use
BF16 autocast for eligible forward operations on CPU or a CUDA device with
native BF16 support, such as an RTX 4090. Parameters, gradients, AdamW state,
and the cross-entropy calculation remain FP32. BF16 does not halve total
training memory, and the feasible batch size must be measured.

For example, add `--dtype bfloat16` to a training command. Use the same option
when resuming a BF16 run: checkpoints store the model and optimizer, but not
the requested training precision.

### Gradient accumulation

`--batch-size` sets the number of sequences in each physical micro-batch.
Use `--gradient-accumulation-steps` to average gradients over multiple
micro-batches before clipping and updating the model. The default is 1,
which preserves the original one-batch-per-update behavior. For example,
`--batch-size 4 --gradient-accumulation-steps 16` gives an effective batch of
64 sequences; with `--context-length 1024`, that is 65,536 tokens per update.

Training loss is the mean across the micro-batches. Learning-rate scheduling,
logging, evaluation, and checkpoint intervals count optimizer updates, not
micro-batches. Evaluation still uses `--batch-size` sequences per evaluation
batch. Checkpoints do not store the accumulation setting; pass the intended
settings explicitly when resuming.

For memory or throughput pilots, pass `--no-checkpoint` to disable both
periodic and final checkpoint writes. No training progress from such a run
can be resumed unless it was saved separately.

### No positional encoding experiment

Pass `--nope` to training to disable RoPE in every transformer layer. The
causal attention mask remains enabled. RMSNorm remains enabled unless
`--no-rmsnorm` is also specified. For example:

```sh
uv run python -m cs336_basics.train \
  --train-data data/tokenized/tinystories_train.npy \
  --valid-data data/tokenized/tinystories_valid.npy \
  --batch-size 64 --max-lr 1e-3 --min-lr 1e-4 \
  --nope --checkpoint-dir checkpoints/nope-lr-1e-3
```

RoPE is enabled by default, and `--rope-theta` has no effect with `--nope`.
RoPE and NoPE checkpoints have the same weight keys, so loading a checkpoint
does not identify which mode was used. Always pass `--nope` when resuming a
NoPE training run and set `NOPE=1` when generating from a NoPE checkpoint:

```sh
NOPE=1 CHECKPOINT=checkpoints/nope-lr-1e-3/latest.pt \
  bash scripts/generate_sample.sh
```
