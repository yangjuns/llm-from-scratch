# LLM Experiment Notes

## Batch size sweep (TinyStories, lr 1e-3)

Setup: powers of two from 1 to 512, token budget matched to 40,000 steps at
batch size 32 (context length 256), peak lr `1e-3`, min lr `1e-4`, FP32, one
24 GB GPU (RTX 4090). Script: `assignment1-basics/scripts/sweep_batch_size.sh`.

### Batch size 256 runs out of memory

Run `batch-256-lr-1e-3`
([W&B](https://wandb.ai/yangjuns-cmu-google/cs336-assignment1/runs/rot2j85j)):

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.44 GiB.
GPU 0 has a total capacity of 23.52 GiB of which 1.88 GiB is free.
Process 2596220 has 21.63 GiB memory in use. Of the allocated memory
20.14 GiB is allocated by PyTorch, and 1.04 GiB is reserved by PyTorch but
unallocated.
```

- PyTorch had about 20.14 GiB allocated and 1.04 GiB reserved-but-unallocated
  when it tried to allocate another 2.44 GiB, with only 1.88 GiB free.
- The `16P + 4A` estimate (see `a1-llm-implementation.md`) predicted
  `363 MB + 93.7 MB * B`, i.e. about 24 GB at `B = 256`, with a ceiling near
  `B = 271` on 24 GiB. In practice 256 does not fit, so the estimate is
  optimistic: it ignores temporaries, such as the extra copies made by the
  float32 softmax and the backward pass, plus allocator overhead and
  fragmentation.
- Practical limit on this GPU is therefore below 256 (128 is the largest
  power of two that fits in the sweep). 512 would not fit either.
- Possible mitigations: set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  to reduce fragmentation, use bf16 autocast, or use gradient accumulation to
  reach a larger effective batch.

## Model used for generation

| Item | Value |
| --- | --- |
| Checkpoint | `checkpoints/batch-64-lr-1e-3/latest.pt` (step 20,000) |
| Parameters | 22,696,448 (vocab 10,000, context 256, d_model 512, 4 layers, 16 heads, d_ff 1344) |
| Training tokens | 327,680,000 (20,000 steps × batch 64 × context 256) |
| Training data | TinyStories |
| Hardware | 1× NVIDIA RTX 4090 (24 GB), FP32 |
| Training time | 3,178 s (about 53 min) |

## Text generation from the batch-64 checkpoint

Setup: `checkpoints/batch-64-lr-1e-3/latest.pt` (step 20000), generated on the
GPU with `scripts/generate_sample.sh` using its defaults: prompt
`<|endoftext|>` (unconditional), temperature 0.8, top-p 0.9, seed 0, up to
256 new tokens.

```text
<|endoftext|>
Once upon a time, there was a little girl named Lucy. She had a pretty skirt that she loved very much. One day, she wanted to wear her skirt to play outside with her friends.
While playing, Lucy fell down and hurt her knee. She started to cry. Her mom heard her and came to help her. "Don't worry, Lucy," her mom said. "We can fix your knee and make it feel better."
Lucy's mom took out some soft cloth and wrapped it around her knee. She kissed it and said, "There, all better now. Let's go to the doctor."
At the doctor, the doctor looked at Lucy's knee and said, "You need to rest and rest. You will feel better soon." Lucy felt a little better. She went back to her friends and they all played together again.
<|endoftext|>
```

The model emitted `<|endoftext|>` on its own, so generation stopped before the
256-token limit.

Model size: **22,696,448 parameters** (vocab 10,000, context 256, d_model 512,
4 layers, 16 heads, d_ff 1344). Embedding and LM head are separate matrices
(`2Vd` = 10,240,000), so they are not weight-tied; the 4 blocks add 12,455,936
and the final RMSNorm adds 512.

### Fluency

- The text is fluent: grammatical sentences, correct dialogue punctuation, a
  consistent character (Lucy) and a clear beginning, middle and end.
- The story has small logic slips typical of a small model: the skirt is
  introduced and never used again, "rest and rest" is repetitive, and the
  doctor visit is not really motivated by a scraped knee.
- The model started the story itself with "Once upon a time", which shows it
  learned how TinyStories begins after an `<|endoftext|>`.

### Factors that affect output quality

- **Sampling settings:** lower temperature and a smaller top-p give safer,
  more repetitive text. Higher values give more variety but more errors.
  0.8 and 0.9 are a reasonable middle.
- **Model and training:** about 22.7M parameters trained on TinyStories for
  20000 steps at batch size 64. More training or a lower validation loss gives
  more coherent, consistent stories.
- **Prompt and data domain:** the model only knows simple children's-story
  English. Prompts outside that domain would produce poor text.
- **Context length:** the model sees at most 256 tokens, so it can lose track
  of details (like the skirt) in longer stories.
