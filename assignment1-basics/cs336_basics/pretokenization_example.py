import os
from pathlib import Path
from typing import BinaryIO

import bpe
import pretokenization as pt

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than
    # desired_num_chunks
    return sorted(set(chunk_boundaries))


## Usage
data_path = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "TinyStoriesV2-GPT4-train.txt"
)

with open(data_path, "rb") as f:
    num_processes = 4
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
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

        pre_token_counts = pt.pre_tokenize(documents, regex_pattern=PAT)
        # The vocabulary size should be the sum of the lengths of all
        vocab_size = sum(len(pre_token) for pre_token in pre_token_counts)
        print("Vocabulary size:", vocab_size)

        pre_token_counts, merges = bpe.bpe_train(pre_token_counts, merges=None)
        vocab_size = sum(len(pre_token) for pre_token in pre_token_counts)
        print("Vocabulary size after BPE:", vocab_size)
