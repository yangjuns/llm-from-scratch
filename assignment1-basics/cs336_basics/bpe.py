def bpe_train(
    pre_token_counts: dict[tuple[bytes, ...], int],
    target_vocab_size: int,
    special_tokens: list[str] = [],
):
    """
    Train a BPE model on the pre-tokenized bytes with vocab size <=
    target_vocab_size. Returns the vocabulary and the merges.
    """
    merges: list[tuple[bytes, bytes]] = []
    # print pre_token_counts
    for pre_token, count in pre_token_counts.items():
        print(f"{pre_token}  :  {count}")
    # Build the cache of the byte pairs and their counts.
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    for pre_token, count in pre_token_counts.items():
        for i in range(len(pre_token) - 1):
            byte_pair = (pre_token[i], pre_token[i + 1])
            if byte_pair in pair_counts:
                pair_counts[byte_pair] += count
            else:
                pair_counts[byte_pair] = count
    # Print pair counts for debugging
    print("Initial pair counts:")
    for pair, count in pair_counts.items():
        print(f"{pair}: {count}")
    # Build the reverse mapping from count to byte pairs for efficient lookup
    # of the most frequent pair
    count_to_pairs: dict[int, list[tuple[bytes, bytes]]] = {}
    for byte_pair, count in pair_counts.items():
        if count not in count_to_pairs:
            count_to_pairs[count] = list()
        count_to_pairs[count].append(byte_pair)
    # Print count to pairs for debugging

    total_iter = 6
    cur_iter = 0
    while cur_iter < total_iter:
        print("Current count to pairs:")
        for count, pairs in count_to_pairs.items():
            print(f"{count}: {pairs}")
        # Find the pair with the highest count, break ties by lexicographic
        # order of the first byte, then the second byte
        most_frequent_bytes_pair: tuple[bytes, bytes] | None = None
        highest_count = 0

        for count, pairs in count_to_pairs.items():
            if count > highest_count:
                highest_count = count
                most_frequent_bytes_pair = pairs[0]
                for pair in pairs:
                    if pair > most_frequent_bytes_pair:
                        most_frequent_bytes_pair = pair
        # Print most frequent byte pair for debugging
        print(
            f"Most frequent byte pair: {most_frequent_bytes_pair}, count: {highest_count}"
        )
        # Merge and update count index
        if most_frequent_bytes_pair is not None:
            merges.append(most_frequent_bytes_pair)
            count_to_pairs[highest_count].remove(most_frequent_bytes_pair)
            if len(count_to_pairs[highest_count]) == 0:
                del count_to_pairs[highest_count]
            # Merge most frequent bytes pair
            merged_bytes = (
                most_frequent_bytes_pair[0] + most_frequent_bytes_pair[1]
            )
            # Update count_to_pairs
            for count, pairs in count_to_pairs.items():
                for i, pair in enumerate(pairs):
                    if pair[0] == most_frequent_bytes_pair[1]:
                        print(
                            f"first byte of pair {pair} matches second byte of most frequent byte pair {most_frequent_bytes_pair}"
                        )
                        pairs[i] = (merged_bytes, pair[1])
                    if pair[1] == most_frequent_bytes_pair[0]:
                        print(
                            f"second byte of pair {pair} matches first byte of most frequent byte pair {most_frequent_bytes_pair}"
                        )
                        pairs[i] = (pair[0], merged_bytes)
                        print(
                            f"Updated pair: {pair} with new pair: {(pair[0], merged_bytes)}"
                        )
        cur_iter += 1
    # Construct the final vocabulary
    vocab: dict[int, bytes] = {}
    next_token_id = 0
    # Add special tokens to the vocabulary
    for token in special_tokens:
        vocab[next_token_id] = token.encode("utf-8")
        next_token_id += 1
    for pre_token in pre_token_counts:
        for token in pre_token:
            if token not in vocab.values():
                vocab[next_token_id] = token
                next_token_id += 1
    print(f"Final vocabulary size: {len(vocab)}")
    return vocab, merges
