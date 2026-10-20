class BytesPairNode:
    def __init__(self, pair: tuple[bytes, bytes], count: int):
        self.pair = pair
        self.count = count
        self.prev: BytesPairNode | None = None
        self.next: BytesPairNode | None = None


def is_pair_greater(l: tuple[bytes, bytes], r: tuple[bytes, bytes]):
    if l[0] > r[0]:
        return True
    if l[0] == r[0] and l[1] > r[1]:
        return True
    return False


def bpe_train(
    pre_token_counts: dict[tuple[bytes, ...], int],
    target_vocab_size: int,
    special_tokens: list[str] = [],
):
    """
    Train a BPE model on the pre-tokenized bytes with vocab size <=
    target_vocab_size. Returns the vocabulary and the merges.
    """

    # Initialize vocab
    vocab: dict[int, bytes] = {}
    next_token_id = 0
    for token in special_tokens:
        vocab[next_token_id] = token.encode("utf-8")
        next_token_id += 1
    for i in range(256):
        b = bytes([i])
        if b not in vocab.values():
            vocab[next_token_id] = b
            next_token_id += 1

    merges: list[tuple[bytes, bytes]] = []

    # Create a global hash table that maps (bytes, bytes) to a list of
    # BytesPairNode
    pair_nodes: dict[tuple[bytes, bytes], list[BytesPairNode]] = {}
    for pre_token, count in pre_token_counts.items():
        # Create a Double-Linked list for each pre-token, and add the node to
        # the hash table.
        prev_node = None
        for i in range(len(pre_token) - 1):
            node = BytesPairNode((pre_token[i], pre_token[i + 1]), count)
            node.prev = prev_node
            if node.prev is not None:
                node.prev.next = node
            prev_node = node
            if node.pair in pair_nodes:
                pair_nodes[node.pair].append(node)
            else:
                pair_nodes[node.pair] = [node]

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

    while len(vocab) < target_vocab_size:
        # for pair, count in pair_counts.items():
        #     print(f"{pair}: {count}")
        # Find the pair with the highest count, break ties by lexicographic
        # order of the first byte, then the second byte
        highest_count = 0
        highest_pair = None
        for pair, count in pair_counts.items():
            if count > highest_count:
                highest_count = count
                highest_pair = pair
            elif (
                count == highest_count
                and highest_pair is not None
                and is_pair_greater(pair, highest_pair)
            ):
                highest_pair = pair
        # # Print most frequent byte pair for debugging
        # print(
        #     f"Most frequent byte pair: {highest_pair}, count: {highest_count}"
        # )
        # Merge the pair
        if highest_pair is not None:
            # Update vocab
            merged_bytes = highest_pair[0] + highest_pair[1]
            vocab[next_token_id] = merged_bytes
            next_token_id += 1
            merges.append(highest_pair)
            for node in pair_nodes[highest_pair]:
                # X A B Y => X AB Y
                if node.prev is not None:
                    old_left_node = node.prev

                    # Construct new left node
                    new_pair = (old_left_node.pair[0], merged_bytes)
                    new_left_node = BytesPairNode(
                        new_pair,
                        old_left_node.count,
                    )
                    new_left_node.prev = old_left_node.prev
                    new_left_node.next = node.next
                    # Update hash table
                    if new_pair in pair_nodes:
                        pair_nodes[new_pair].append(new_left_node)
                    else:
                        pair_nodes[new_pair] = [new_left_node]
                    # Update the pair_count
                    pair_counts[old_left_node.pair] -= old_left_node.count

                    pair_counts[new_left_node.pair] = (
                        pair_counts.get(new_left_node.pair, 0)
                        + new_left_node.count
                    )
                if node.next is not None:
                    old_right_node = node.next
                    # Construct new right node
                    new_pair = (merged_bytes, old_right_node.pair[1])
                    new_right_node = BytesPairNode(
                        new_pair,
                        old_right_node.count,
                    )
                    new_right_node.next = old_right_node.next
                    new_right_node.prev = node.prev
                    if new_pair in pair_nodes:
                        pair_nodes[new_pair].append(new_right_node)
                    else:
                        pair_nodes[new_pair] = [new_right_node]
                    # Update the pair_count
                    pair_counts[old_right_node.pair] -= old_right_node.count
                    pair_counts[new_right_node.pair] = (
                        pair_counts.get(new_right_node.pair, 0)
                        + new_right_node.count
                    )
            del pair_nodes[highest_pair]
            del pair_counts[highest_pair]

    print(f"Final vocabulary size: {len(vocab)}")
    return vocab, merges
