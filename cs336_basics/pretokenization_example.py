import os
from typing import BinaryIO

import regex as re

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

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


## Usage
with open("data/TinyStoriesV2-GPT4-train.txt", "rb") as f:
    num_processes = 4
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    print(f"Found {len(boundaries)} chunk boundaries: {boundaries}")

    # Use a small chunk size for testing, but you can use a larger chunk size for production
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

        # Split chunks to documents by special token, and remove the special token from the chunk
        documents = chunk.split("<|endoftext|>")
        for i, doc in enumerate(documents):
            documents[i] = doc.strip()
            print(f"Document {i} in chunk {start}-{end}: {documents[i]}")

        # Perform pre-tokenization for each document in the chunk. Create a global pre-token map
        #  dict[tuple[bytes, ...], int], where the tuple contains the pre-tokenized bytes and the int is the count of that pre-token in the chunk.
        pre_token_counts: dict[tuple[bytes, ...], int] = {}
        for i, doc in enumerate(documents):
            pre_tokens = re.finditer(PAT, doc)
            for pre_token in pre_tokens:
                pre_token_bytes = tuple(
                    bytes([byte]) for byte in pre_token.group().encode("utf-8")
                )
                if pre_token_bytes in pre_token_counts:
                    pre_token_counts[pre_token_bytes] += 1
                else:
                    pre_token_counts[pre_token_bytes] = 1

        # Print the counts for this chunk
        print(f"Pre-token counts for chunk {start}-{end}:")
        for pre_token, count in pre_token_counts.items():
            print(f"  {pre_token}: {count}")

        # count each byte pair in the pre-tokenized bytes and print the counts
        # for this chunk
        byte_pair_counts: dict[tuple[bytes, bytes], int] = {}
        for pre_token, count in pre_token_counts.items():
            for i in range(len(pre_token) - 1):
                byte_pair = (pre_token[i], pre_token[i + 1])
                if byte_pair in byte_pair_counts:
                    byte_pair_counts[byte_pair] += count
                else:
                    byte_pair_counts[byte_pair] = count
        print(f"Byte pair counts for chunk {start}-{end}:")
        # Find the pair with the highest count, break ties by lexicographic
        # order of the first byte, then the second byte
        most_frequent_byte_pair = None
        highest_count = 0
        for byte_pair, count in sorted(
            byte_pair_counts.items(), key=lambda x: (-x[1], x[0])
        ):
            if count > highest_count:
                if (
                    most_frequent_byte_pair is None
                    or byte_pair < most_frequent_byte_pair
                ):
                    most_frequent_byte_pair = byte_pair
                    highest_count = count
        print(
            f"Most frequent byte pair for chunk {start}-{end}: "
            f"{most_frequent_byte_pair} with count {highest_count}"
        )
        # In the pre-token counts, replace the most frequent byte pair with a
        # new pre-token that is the concatenation of the two bytes in the pair.

        if most_frequent_byte_pair is not None:
            pass

        break  # Remove this break to process all chunks
