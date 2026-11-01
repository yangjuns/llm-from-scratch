# LLM from Scratch

An end-to-end PyTorch implementation of a Transformer language model and training infrastructure, based on the Stanford CS336 specifications.

## Repository Structure
- `assignment1-basics/` - Custom BPE Tokenizer, Transformer Architecture (RMSNorm, SwiGLU, RoPE, Multi-Head Attention)
- `notes/` - Learning notes across assignments (see below)

## Learning Notes

Write-ups of what I measured and learned along the way. Naming: `aN-<topic>.md` per assignment.

| Assignment | Note | Key takeaway |
|---|---|---|
| A1 | [BPE: TinyStories vs OpenWebText](notes/a1-bpe-tinystories-vs-owt.md) | Vocab reflects corpus: clean data gives word-like tokens, noisy web data wastes slots on junk runs and mojibake |
| A1 | [LLM implementation](notes/a1-llm-implementation.md) | RMSNorm, linear (no bias), SwiGLU, RoPE, and numerical precision: matmuls can use low precision, but positions, norms, softmax, and optimizer state need float32 |
| General | [Tensor operations](notes/tensor-operations.md) | Track shapes, not values: indexing, broadcasting, einsum, stack/cat, rearrange, and masking |
