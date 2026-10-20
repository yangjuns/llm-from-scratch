import os
from typing import BinaryIO

import regex as re


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


def pre_tokenize_doc_segment(doc_segment: str, regex_pattern: str):
    """
    Pre-tokenize a document segment using the regex pattern defined in PAT.
    The function creates a global pre-token map dict[tuple[bytes, ...], int]
    """
    pre_token_counts: dict[tuple[bytes, ...], int] = {}
    pre_tokens = re.finditer(regex_pattern, doc_segment)
    for pre_token in pre_tokens:
        pre_token_bytes = tuple(
            bytes([byte]) for byte in pre_token.group().encode("utf-8")
        )
        if pre_token_bytes in pre_token_counts:
            pre_token_counts[pre_token_bytes] += 1
        else:
            pre_token_counts[pre_token_bytes] = 1
    return pre_token_counts


def pre_tokenize_doc(
    document: str,
    special_tokens: list[str],
    regex_pattern: str,
):
    segments = re.split("|".join(special_tokens), document)
    doc_token_counts: dict[tuple[bytes, ...], int] = {}
    for segment in segments:
        segment_token_counts = pre_tokenize_doc_segment(segment, regex_pattern)
        for token, count in segment_token_counts.items():
            if token in doc_token_counts:
                doc_token_counts[token] += count
            else:
                doc_token_counts[token] = count
    return doc_token_counts


def pre_tokenize(
    documents: list[str],
    special_tokens: list[str],
    regex_pattern: str,
):
    """
    Pre-tokenize a list of documents using the regex pattern defined in PAT.
    The function creates a global pre-token map dict[tuple[bytes, ...], int],\
    
    """

    # Perform pre-tokenization for each document in the chunk. Create a
    #  global pre-token map dict[tuple[bytes, ...], int], where the tuple
    #  contains the pre-tokenized bytes and the int is the count of that
    #  pre-token in the chunk.
    token_counts: dict[tuple[bytes, ...], int] = {}
    for doc in documents:
        doc_token_counts = pre_tokenize_doc(doc, special_tokens, regex_pattern)
        for token, count in doc_token_counts.items():
            if token in token_counts:
                token_counts[token] += count
            else:
                token_counts[token] = count
    return token_counts
