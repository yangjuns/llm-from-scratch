from collections.abc import Iterable, Iterator

import json
import regex as re


class Tokenizer:
    regex_pattern = (
        r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|"
        r"\s+(?!\S)|\s+"
    )

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        """
        Construct a tokenizer from a given vocabulary, list of merges, and
        (optionally) a list of special tokens. This function should accept
        the following parameters:
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None = None
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = (
            special_tokens if special_tokens is not None else []
        )
        # Construct a reverse mapping from token bytes to their IDs for quick lookup
        self.vocab_reverse = {v: k for k, v in self.vocab.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.special_token_set = set(self.special_tokens)
        # Longest first so overlapping special tokens match the longer one
        self.special_pattern = (
            re.compile(
                "("
                + "|".join(
                    re.escape(t)
                    for t in sorted(self.special_tokens, key=len, reverse=True)
                )
                + ")"
            )
            if self.special_tokens
            else None
        )

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        """
        Construct a tokenizer from files containing the vocabulary and merges.
        vocab_filepath: str, a JSON file containing the vocabulary
        merges_filepath: str, a JSON file containing the merges
        special_tokens: list[str] | None = None
        """
        vocab: dict[int, bytes] = {}
        merges: list[tuple[bytes, bytes]] = []
        with open(vocab_filepath, encoding="utf-8") as f:
            # Parse the JSON file containing the vocabulary, convert to
            # dictionary with int keys. Json contains a list of vocabulary
            # items.
            vocab_mapping = json.load(f)

            for item in vocab_mapping:
                vocab[item["id"]] = bytes.fromhex(item["hex"])

        with open(merges_filepath, encoding="utf-8") as f:
            # Parse the JSON file containing the merges
            merge_mapping = json.load(f)
            merges = [
                (
                    bytes.fromhex(item["left"]["hex"]),
                    bytes.fromhex(item["right"]["hex"]),
                )
                for item in merge_mapping
            ]

        return cls(vocab, merges, special_tokens)

    def _merge_token(self, token_bytes: tuple[bytes, ...]) -> list[bytes]:
        # Repeatedly apply the earliest-created merge present in the token
        parts = list(token_bytes)
        while len(parts) > 1:
            best_rank = None
            for i in range(len(parts) - 1):
                rank = self.merge_ranks.get((parts[i], parts[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
            if best_rank is None:
                break
            merged: list[bytes] = []
            i = 0
            while i < len(parts):
                if (
                    i < len(parts) - 1
                    and self.merge_ranks.get((parts[i], parts[i + 1]))
                    == best_rank
                ):
                    merged.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1
            parts = merged
        return parts

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        token_ids: list[int] = []
        if self.special_pattern is not None:
            # The capture group keeps special tokens in the split result
            segments = self.special_pattern.split(text)
        else:
            segments = [text]
        for segment in segments:
            if not segment:
                continue
            if segment in self.special_token_set:
                token_ids.append(self.vocab_reverse[segment.encode("utf-8")])
                continue
            for match in re.finditer(self.regex_pattern, segment):
                pre_token = tuple(
                    bytes([b]) for b in match.group().encode("utf-8")
                )
                for part in self._merge_token(pre_token):
                    token_ids.append(self.vocab_reverse[part])
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a
        generator that lazily yields token IDs. This is required for
        memory-efficient tokenization of large files that we cannot directly
        load into memory.
        """
        buffered_text: str = ""
        for text in iterable:
            # Keep reading the iterable until we hit a special token or the end of
            # the iterable
            buffered_text += text
            if self.special_pattern is None:
                yield from self.encode(buffered_text)
                buffered_text = ""
                continue
            segments = self.special_pattern.split(buffered_text)
            for segment in segments[:-1]:
                yield from self.encode(segment)
            buffered_text = segments[-1]
        if buffered_text:
            yield from self.encode(buffered_text)

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        return b"".join(self.vocab[i] for i in ids).decode(
            "utf-8", errors="replace"
        )
