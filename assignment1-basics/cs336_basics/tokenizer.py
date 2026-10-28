from collections.abc import Iterable, Iterator

import json
import re

import cs336_basics.pretokenization as pt


class Tokenizer:
    regex_pattern = (
        r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+[\r\n]*|"
        r"\s*[\r\n]|\s+(?!\S)|\s+"
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

    def _merge_token(self, token_bytes: tuple[bytes, ...]) -> bytes:
        merge_applicable = True
        while merge_applicable:
            # Iterate over the tuple of bytes to each if we have any applicable
            # merges
            merge_applied = False
            for i in range(len(token_bytes) - 1):
                # Check if the current pair of bytes matches any merge rule
                for left, right in self.merges:
                    if token_bytes[i] == left and token_bytes[i + 1] == right:
                        token_bytes = (
                            token_bytes[:i]
                            + tuple([left + right])
                            + token_bytes[i + 2 :]
                        )
                        merge_applied = True
                        break
                if merge_applied:
                    break
            if not merge_applied:
                merge_applicable = False
        return b"".join(token_bytes)

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        # Pre-tokenization
        pre_tokens = pt.pre_tokenize_doc(
            text,
            self.special_tokens,
            regex_pattern=self.regex_pattern,
        )
        # Represent
        token_bytes_list: list[bytes] = [b"" for _ in pre_tokens.keys()]
        # Apply BPE merge for each token in token_bytes_list
        for i, token_bytes in enumerate(pre_tokens.keys()):
            token_bytes_list[i] = self._merge_token(token_bytes)
        token_ids = [
            self.vocab_reverse[token_bytes] for token_bytes in token_bytes_list
        ]
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
            # Split on the special tokens
            segments = re.split("|".join(self.special_tokens), buffered_text)
            for segment in segments[:-1]:
                yield from self.encode(segment)
            buffered_text = segments[-1]
        if buffered_text:
            yield from self.encode(buffered_text)

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        return b"".join(self.vocab[i] for i in ids).decode()
