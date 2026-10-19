def bpe_train(
    pre_token_counts: dict[tuple[bytes, ...], int],
    merges: list[tuple[bytes, bytes]] | None = None,
):
    """
    Train a BPE model on the pre-tokenized bytes. The function creates a global
    byte pair map dict[tuple[bytes, bytes], int], where the tuple contains the
    byte pair and the int is the count of that byte pair in the chunk.
    """
    # count each byte pair in the pre-tokenized bytes and print the counts
    # for this chunk
    byte_pair_counts: dict[tuple[bytes, bytes], int] = {}
    if merges is None:
        merges = []
    for pre_token, count in pre_token_counts.items():
        for i in range(len(pre_token) - 1):
            byte_pair = (pre_token[i], pre_token[i + 1])
            if byte_pair in byte_pair_counts:
                byte_pair_counts[byte_pair] += count
            else:
                byte_pair_counts[byte_pair] = count
    # Find the pair with the highest count, break ties by lexicographic
    # order of the first byte, then the second byte
    most_frequent_byte_pair: tuple[bytes, bytes] | None = None
    highest_count = 0
    for byte_pair, count in sorted(
        byte_pair_counts.items(), key=lambda x: (-x[1], x[0])
    ):
        if count > highest_count and (
            most_frequent_byte_pair is None
            or byte_pair < most_frequent_byte_pair
        ):
            most_frequent_byte_pair = byte_pair
            highest_count = count
    print(
        f"Most frequent byte pair: "
        f"{most_frequent_byte_pair} with count {highest_count}"
    )

    # Merge
    if most_frequent_byte_pair is not None:
        merges.append(most_frequent_byte_pair)
        for pre_token, count in list(pre_token_counts.items()):
            for i in range(len(pre_token) - 1):
                if (
                    pre_token[i] == most_frequent_byte_pair[0]
                    and pre_token[i + 1] == most_frequent_byte_pair[1]
                ):
                    new_pre_token = (
                        pre_token[:i]
                        + (
                            most_frequent_byte_pair[0]
                            + most_frequent_byte_pair[1],
                        )
                        + pre_token[i + 2 :]
                    )
                    pre_token_counts[new_pre_token] = count
                    del pre_token_counts[pre_token]
    return pre_token_counts, merges
