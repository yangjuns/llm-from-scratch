from pprint import pprint


class TokenNode:
    def __init__(self, token: bytes | None, count: int):
        self.token = token
        self.count = count
        self.prev: TokenNode | None = None
        self.next: TokenNode | None = None

    def __repr__(self) -> str:

        return (
            f"TokenNode(token={self.token}, count={self.count}, "
            f"prev={self.prev.token if self.prev else None}, "
            f"next={self.next.token if self.next else None})"
        )


class TokenNodeList:
    def __init__(self):
        self.head: TokenNode | None = None
        self.tail: TokenNode | None = None

    def append(self, node: TokenNode):
        if self.tail is None:
            self.head = self.tail = node
        else:
            self.tail.next = node
            node.prev = self.tail
            self.tail = node


class PairCounters:
    def __init__(self):
        self.pair_counts: dict[tuple[bytes, bytes], int] = {}

    def increment(self, pair: tuple[bytes, bytes], count: int):
        if pair in self.pair_counts:
            self.pair_counts[pair] += count
        else:
            self.pair_counts[pair] = count

    def decrement(self, pair: tuple[bytes, bytes], count: int):
        if pair in self.pair_counts:
            self.pair_counts[pair] -= count
            if self.pair_counts[pair] == 0:
                del self.pair_counts[pair]

    def __getitem__(self, pair: tuple[bytes, bytes]):
        return self.pair_counts.get(pair, 0)

    def __setitem__(self, pair: tuple[bytes, bytes], count: int):
        self.pair_counts[pair] = count


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

    # Create a global hash table that maps bytes to a list of
    # TokenNodes
    pair_nodes: dict[bytes, list[TokenNode]] = {}
    for pre_token, count in pre_token_counts.items():
        # Create a Double-Linked list for each pre-token, and add the node to
        # the hash table.
        prev_node = None
        for i in range(len(pre_token)):
            node = TokenNode(pre_token[i], count)
            node.prev = prev_node
            if node.prev is not None:
                node.prev.next = node
            prev_node = node
            assert node.token is not None
            if node.token in pair_nodes:
                pair_nodes[node.token].append(node)
            else:
                pair_nodes[node.token] = [node]

    # Build the cache of the byte pairs and their counts.
    pair_counters = PairCounters()
    for pre_token, count in pre_token_counts.items():
        for i in range(len(pre_token) - 1):
            byte_pair = (pre_token[i], pre_token[i + 1])
            pair_counters.increment(byte_pair, count)

    while len(vocab) < target_vocab_size:
        # Find the pair with the highest count, break ties by lexicographic
        # order of the first byte, then the second byte
        highest_count = 0
        highest_pair = None
        for pair, count in pair_counters.pair_counts.items():
            if count > highest_count:
                highest_count = count
                highest_pair = pair
            elif (
                count == highest_count
                and highest_pair is not None
                and is_pair_greater(pair, highest_pair)
            ):
                highest_pair = pair
        # Merge the pair
        if highest_pair is None:
            break

        # Update vocab
        merged_bytes = highest_pair[0] + highest_pair[1]
        vocab[next_token_id] = merged_bytes
        next_token_id += 1
        merges.append(highest_pair)
        for node in pair_nodes[highest_pair[0]]:
            if (
                node.token is None
                or node.next is None
                or node.next.token != highest_pair[1]
            ):
                # The node has been merged and no longer represents a valid
                # occurrence of the selected pair.
                continue
            # X A B Y => X AB Y
            # Construct a new node for the merged bytes
            new_node = TokenNode(
                merged_bytes,
                node.count,
            )
            # Update hash table
            if merged_bytes in pair_nodes:
                pair_nodes[merged_bytes].append(new_node)
            else:
                pair_nodes[merged_bytes] = [new_node]
            # Invalidate the original pair nodes
            node.token = None
            node.next.token = None
            # Connect the new node with the surrounding nodes
            node_before_pair = node.prev
            node_after_pair = node.next.next
            new_node.prev = node_before_pair
            new_node.next = node_after_pair
            # Decrement the count of the highest pair since this node is being merged
            pair_counters.decrement(highest_pair, node.count)
            if node_before_pair is not None:
                # Old left node now needs to connect to the new node
                node_before_pair.next = new_node

                # Update the pair counters for the old and new pairs
                assert node_before_pair.token is not None
                old_pair = (node_before_pair.token, highest_pair[0])
                pair_counters.decrement(old_pair, node.count)
                new_pair = (node_before_pair.token, merged_bytes)
                pair_counters.increment(new_pair, new_node.count)
            if node_after_pair is not None:
                # The old right node is the node after the merged pair. It
                # needs to connect to the new node
                node_after_pair.prev = new_node

                # Update the pair counters for the old and new pairs
                assert node_after_pair.token is not None
                assert new_node.token is not None
                old_pair = (highest_pair[1], node_after_pair.token)
                # Construct new right node
                new_pair = (merged_bytes, node_after_pair.token)
                pair_counters.decrement(old_pair, node_after_pair.count)
                pair_counters.increment(new_pair, node_after_pair.count)

    print(f"Final vocabulary size: {len(vocab)}")
    return vocab, merges
