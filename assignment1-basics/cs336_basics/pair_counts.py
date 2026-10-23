import heapq


class _ReversedPair:
    """Orders pairs descending so the min-heap pops the lexicographically
    largest pair first among equal counts."""

    def __init__(self, pair: tuple[bytes, bytes]):
        self.pair = pair

    def __lt__(self, other: "_ReversedPair") -> bool:
        return self.pair > other.pair


class PairCounters:
    """Pair counts with a max-heap using lazy updates.

    `pair_counts` is the source of truth. The heap may hold stale entries,
    which are discarded when they reach the top in `best()`.
    """

    def __init__(self):
        self.pair_counts: dict[tuple[bytes, bytes], int] = {}
        self._heap: list[tuple[int, _ReversedPair]] = []

    def initialize(self, pair_counts: dict[tuple[bytes, bytes], int]):
        self.pair_counts = pair_counts.copy()
        self._heap = [
            (-count, _ReversedPair(pair))
            for pair, count in self.pair_counts.items()
            if count > 0
        ]
        heapq.heapify(self._heap)

    def _push(self, pair: tuple[bytes, bytes]):
        count = self.pair_counts[pair]
        if count > 0:
            heapq.heappush(self._heap, (-count, _ReversedPair(pair)))

    def increment(self, pair: tuple[bytes, bytes], count: int):
        self.pair_counts[pair] = self.pair_counts.get(pair, 0) + count
        self._push(pair)

    def decrement(self, pair: tuple[bytes, bytes], count: int):
        if pair in self.pair_counts:
            self.pair_counts[pair] -= count
            if self.pair_counts[pair] == 0:
                del self.pair_counts[pair]
            else:
                self._push(pair)

    def best(self) -> tuple[bytes, bytes] | None:
        """Return the pair with the highest count (ties: largest pair)."""
        while self._heap:
            neg_count, rev = self._heap[0]
            if self.pair_counts.get(rev.pair) == -neg_count:
                return rev.pair
            heapq.heappop(self._heap)
        return None

    def __getitem__(self, pair: tuple[bytes, bytes]):
        return self.pair_counts.get(pair, 0)

    def __setitem__(self, pair: tuple[bytes, bytes], count: int):
        self.pair_counts[pair] = count
        self._push(pair)
