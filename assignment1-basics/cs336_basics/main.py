## Usage
from pathlib import Path

from cs336_basics import bpe
import pretokenization as pt

import regex as re

data_path = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "TinyStoriesV2-GPT4-train.txt"
)


example_text = """low low low low low
lower lower widest widest widest
newest newest newest newest newest newest<|endoftext|>"""


# Split chunks to documents by special token, and remove the special
# token from the chunk
documents = example_text.split("<|endoftext|>")
pre_token_counts = pt.pre_tokenize(
    documents,
    special_tokens=["<|endoftext|>"],
    regex_pattern=r"\w+|[^\w\s]+|\s+",
)

# The vocabulary size should be the sum of the lengths of all

vocab, merges = bpe.bpe_train(
    pre_token_counts,
    target_vocab_size=256 + 6,
    special_tokens=["<|endoftext|>"],
)
# Print vocab and merges
print("Vocabulary:")
for token, index in vocab.items():
    print(f"{index}: {token}")
print("\nMerges:")
for merge in merges:
    print(f"{merge[0]} + {merge[1]}")
