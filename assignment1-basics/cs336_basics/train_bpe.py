import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import regex as re

from cs336_basics import bpe
from cs336_basics import pretokenization

SPECIAL_TOKEN = "<|endoftext|>"
NUM_PROCESSES = 6
PRE_TOKEN_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+|"""
    r""" ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _pretokenize_chunk(
    args: tuple[Path, int, int, str],
) -> dict[tuple[bytes, ...], int]:
    input_path, start, end, pattern = args
    with input_path.open("rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    documents = chunk.split(SPECIAL_TOKEN)
    # pre_tokenize builds a regex from special_tokens, so pass it escaped.
    return pretokenization.pre_tokenize(
        documents, [re.escape(SPECIAL_TOKEN)], pattern
    )


def _token_record(token: bytes) -> dict[str, str]:
    return {
        "hex": token.hex(),
        "text": token.decode("utf-8", errors="backslashreplace"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a byte-level BPE tokenizer on a text corpus."
    )
    parser.add_argument(
        "input_path", type=Path, help="Path to the training text file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/bpe"),
        help="Directory for the serialized vocabulary and merges",
    )
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument(
        "--num-processes",
        type=int,
        default=NUM_PROCESSES,
        help=f"Number of processes for pre-tokenization (default: {NUM_PROCESSES})",
    )
    args = parser.parse_args()
    if args.num_processes < 1:
        parser.error("--num-processes must be at least 1")

    with args.input_path.open("rb") as f:
        boundaries = pretokenization.find_chunk_boundaries(
            f, args.num_processes, SPECIAL_TOKEN.encode("utf-8")
        )

    tasks = [
        (args.input_path, start, end, PRE_TOKEN_PATTERN)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    pre_token_counts: dict[tuple[bytes, ...], int] = {}
    with ProcessPoolExecutor(max_workers=args.num_processes) as executor:
        for chunk_counts in executor.map(_pretokenize_chunk, tasks):
            for token, count in chunk_counts.items():
                pre_token_counts[token] = pre_token_counts.get(token, 0) + count

    vocab, merges = bpe.bpe_train(
        pre_token_counts=pre_token_counts,
        target_vocab_size=args.vocab_size,
        special_tokens=[SPECIAL_TOKEN],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (args.output_dir / "vocab.json").open("w", encoding="utf-8") as f:
        json.dump(
            [
                {"id": token_id, **_token_record(token)}
                for token_id, token in sorted(vocab.items())
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )

    with (args.output_dir / "merges.json").open("w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "rank": rank,
                    "left": _token_record(left),
                    "right": _token_record(right),
                }
                for rank, (left, right) in enumerate(merges)
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Saved {len(vocab)} vocabulary entries and {len(merges)} merges "
        f"to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
