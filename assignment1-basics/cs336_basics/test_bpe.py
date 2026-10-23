import unittest

from cs336_basics.bpe import bpe_train, is_pair_greater


class TestBPEPairComparison(unittest.TestCase):
    """Tests for pair lexicographical comparison helper."""

    def test_is_pair_greater_first_byte(self):
        pair_a = (b"b", b"a")
        pair_b = (b"a", b"z")
        self.assertTrue(is_pair_greater(pair_a, pair_b))
        self.assertFalse(is_pair_greater(pair_b, pair_a))

    def test_is_pair_greater_second_byte_tiebreak(self):
        pair_a = (b"a", b"z")
        pair_b = (b"a", b"b")
        self.assertTrue(is_pair_greater(pair_a, pair_b))
        self.assertFalse(is_pair_greater(pair_b, pair_a))

    def test_is_pair_greater_equal_pairs(self):
        pair_a = (b"a", b"b")
        pair_b = (b"a", b"b")
        self.assertFalse(is_pair_greater(pair_a, pair_b))


class TestBPETrain(unittest.TestCase):
    """Tests for the main BPE training algorithm."""

    def test_base_vocabulary_initialization(self):
        """Verify that all 256 single-byte values are included in the initial vocabulary."""
        vocab, merges = bpe_train(pre_token_counts={}, target_vocab_size=256)

        # All 256 byte values should be present
        for i in range(256):
            self.assertIn(bytes([i]), vocab.values())
        self.assertEqual(len(merges), 0)

    def test_special_tokens_indexing(self):
        """Special tokens must appear first in vocabulary starting at token ID 0."""
        special_tokens = ["<|endoftext|>", "<|padding|>"]
        pre_token_counts = {(b"a", b"b"): 5}

        vocab, _ = bpe_train(
            pre_token_counts=pre_token_counts,
            target_vocab_size=260,
            special_tokens=special_tokens,
        )

        self.assertEqual(vocab[0], b"<|endoftext|>")
        self.assertEqual(vocab[1], b"<|padding|>")

    def test_basic_merge_selection(self):
        """Ensure the most frequent byte pair is selected and recorded in merges."""
        # Pair (b'a', b'b') occurs 10 times, (b'b', b'c') occurs 5 times
        pre_token_counts = {
            (b"a", b"b", b"c"): 5,
            (b"a", b"b", b"d"): 5,
        }

        # Request 1 merge step above base 256 bytes
        vocab, merges = bpe_train(
            pre_token_counts=pre_token_counts,
            target_vocab_size=257,
        )

        self.assertEqual(len(merges), 1)
        self.assertEqual(merges[0], (b"a", b"b"))
        self.assertIn(b"ab", vocab.values())

    def test_merge_only_matching_pair_occurrences(self):
        """Do not merge other tokens that share only the selected pair's first byte."""
        pre_token_counts = {
            (b"a", b"b"): 5,
            (b"a", b"c"): 4,
        }

        vocab, merges = bpe_train(
            pre_token_counts=pre_token_counts,
            target_vocab_size=258,
        )

        self.assertEqual(merges, [(b"a", b"b"), (b"a", b"c")])
        self.assertIn(b"ab", vocab.values())
        self.assertIn(b"ac", vocab.values())

    def test_lexicographical_tie_breaking(self):
        """When counts are equal, higher lexicographical pair should be chosen."""
        # Pair (b'x', b'y') and (b'a', b'b') both occur 10 times
        pre_token_counts = {
            (b"a", b"b"): 10,
            (b"x", b"y"): 10,
        }

        vocab, merges = bpe_train(
            pre_token_counts=pre_token_counts,
            target_vocab_size=257,
        )

        self.assertEqual(len(merges), 1)
        # b'x' > b'a', so (b'x', b'y') should be selected first
        self.assertEqual(merges[0], (b"x", b"y"))
        self.assertIn(b"xy", vocab.values())

    def test_target_vocab_size_limit(self):
        """Training should stop once target_vocab_size is reached."""
        pre_token_counts = {
            (b"a", b"b", b"c", b"d", b"e"): 50,
        }

        target_size = 258  # 256 base bytes + 2 merges max
        vocab, merges = bpe_train(
            pre_token_counts=pre_token_counts,
            target_vocab_size=target_size,
        )

        self.assertLessEqual(len(vocab), target_size)
        self.assertLessEqual(len(merges), 2)

    def test_empty_and_single_byte_inputs(self):
        """Handled gracefully when sequence has fewer than 2 bytes to pair."""
        pre_token_counts = {
            (b"a",): 10,
            (): 5,
        }

        vocab, merges = bpe_train(
            pre_token_counts=pre_token_counts,
            target_vocab_size=260,
        )

        self.assertEqual(len(merges), 0)
        self.assertEqual(len(vocab), 256)


class TestBPEOverlappingPairs(unittest.TestCase):
    """Tests for overlapping and repeating byte sequences (e.g., 'a a a')."""

    def test_overlapping_triple_repeating_bytes(self):
        """Verify that sequence (b'a', b'a', b'a') handles middle node consumption correctly.

        In (b'a', b'a', b'a'), there are two adjacent (b'a', b'a') pairs.
        Merging the first pair consumes the middle 'a', invalidating the second pair's
        node reference so it is safely skipped rather than double-merged.
        """
        pre_token_counts = {
            (b"a", b"a", b"a"): 10,
        }

        vocab, merges = bpe_train(
            pre_token_counts=pre_token_counts,
            target_vocab_size=257,
        )

        self.assertEqual(len(merges), 1)
        self.assertEqual(merges[0], (b"a", b"a"))
        self.assertIn(b"aa", vocab.values())

    def test_multi_step_repeating_sequence_merges(self):
        """Verify that multi-step merges on repeating tokens (b'a', b'a', b'a', b'a')

        do not raise KeyError during dictionary count updates and properly form
        hierarchical merges: (a, a) -> aa, then (aa, aa) -> aaaa.
        """
        pre_token_counts = {
            (b"a", b"a", b"a", b"a"): 5,
        }

        vocab, merges = bpe_train(
            pre_token_counts=pre_token_counts,
            target_vocab_size=258,
        )

        self.assertEqual(len(merges), 2)
        self.assertEqual(merges[0], (b"a", b"a"))
        self.assertEqual(merges[1], (b"aa", b"aa"))
        self.assertIn(b"aaaa", vocab.values())


def test_interleaved_common_substrings():
    """Verify safety when old_right_node.pair or old_left_node.pair

    coincides with highest_pair across multiple word types.
    """
    pre_token_counts = {
        (b"t", b"h", b"e", b"t", b"h", b"e"): 100,
        (b"t", b"h", b"e"): 50,
    }

    vocab, merges = bpe_train(
        pre_token_counts=pre_token_counts,
        target_vocab_size=258,
    )

    assert merges[0] == (b"t", b"h")
    assert b"th" in vocab.values()


if __name__ == "__main__":
    unittest.main()
