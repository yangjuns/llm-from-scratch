from cs336_basics import llm

import torch
from cs336_basics import tokenizer


@torch.no_grad()
def decode(
    model: llm.TransformerLM,
    tok: tokenizer.Tokenizer,
    prompt: str,
    max_num_generated_tokens: int,
    temperature: float = 1.0,
    top_p_threshold: float = 0.9,
) -> str:
    # encode prompt into token tensors of shape (1, seq_len)
    tokens: list[int] = tok.encode(prompt)

    device = next(model.parameters()).device
    generated_count = 0
    while generated_count < max_num_generated_tokens:
        # run model on tokens
        # Only the last context_length tokens fit in the model's window.
        context = tokens[-model.context_length :]
        with torch.no_grad():
            logits = model(
                torch.tensor(context, device=device).unsqueeze(0)
            )  # shape (1, seq_len, vocab_size)
        # pick the last token logits
        last_logits = logits[:, -1, :]  # shape (1, vocab_size)
        # temperature scaling with softmax
        probs = torch.softmax(last_logits / temperature, dim=-1)
        # top-p filtering
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cummulative_probs = torch.cumsum(sorted_probs, dim=-1)
        # cummulateive_probs - sorted_probs give cumative without including the
        # current value.
        # If cumulative without current already exceeds threshold, it should be
        # removed
        cutoff = (cummulative_probs - sorted_probs) >= top_p_threshold
        # Identify tokens to REMOVE

        # Zero out removed probabilities
        filtered_probs = sorted_probs.masked_fill(cutoff, 0.0)

        # Sample in sorted-index space
        sampled_index = torch.multinomial(filtered_probs, 1)

        # Map to original vocabulary indices
        token_id: int = sorted_indices.gather(-1, sampled_index).item()
        tokens.append(token_id)
        if tok.decode([token_id]) == "<|endoftext|>":
            break
        generated_count += 1

    return tok.decode(tokens)
