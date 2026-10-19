## Usage
from pathlib import Path

import bpe
import pretokenization as pt

data_path = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "TinyStoriesV2-GPT4-train.txt"
)

with open(data_path, "rb") as f:
    num_processes = 4
    boundaries = pt.find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    print(f"Found {len(boundaries)} chunk boundaries: {boundaries}")

    # Use a small chunk size for testing, but you can use a larger chunk size
    # for production
    boundaries = [
        0,
        1000,
        2000,
        3000,
        4000,
        5000,
        6000,
        7000,
        8000,
        9000,
        10000,
    ]

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        # Split chunks to documents by special token, and remove the special
        # token from the chunk
        documents = chunk.split("<|endoftext|>")
        pre_token_counts = pt.pre_tokenize(
            documents, special_tokens=["<|endoftext|>"]
        )

        # The vocabulary size should be the sum of the lengths of all
        vocab_size = sum(len(pre_token) for pre_token in pre_token_counts)
        print("Vocabulary size:", vocab_size)

        pre_token_counts, merges = bpe.bpe_train(pre_token_counts, merges=None)
        vocab_size = sum(len(pre_token) for pre_token in pre_token_counts)
        print("Vocabulary size after BPE:", vocab_size)

        break
