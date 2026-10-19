import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pre_tokenize(
    documents: list[str],
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
    pre_token_counts: dict[tuple[bytes, ...], int] = {}
    for doc in documents:
        print("Pre-tokenizing document:", doc)
        pre_tokens = re.finditer(regex_pattern, doc)
        for pre_token in pre_tokens:
            pre_token_bytes = tuple(
                bytes([byte]) for byte in pre_token.group().encode("utf-8")
            )
            if pre_token_bytes in pre_token_counts:
                pre_token_counts[pre_token_bytes] += 1
            else:
                pre_token_counts[pre_token_bytes] = 1
    return pre_token_counts
