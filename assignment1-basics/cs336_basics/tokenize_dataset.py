import argparse
import codecs
import os
import shutil
import tempfile
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path

import numpy as np

from cs336_basics.pretokenization import find_chunk_boundaries
from cs336_basics.tokenizer import Tokenizer

SPECIAL_TOKEN = "<|endoftext|>"
SPECIAL_TOKEN_BYTES = SPECIAL_TOKEN.encode("ascii")
CHUNK_SIZE_BYTES = 1 << 20
CHUNK_SIZE_IDS = 1_000_000
TOKEN_DTYPE = np.dtype("<u2")
_WORKER_TOKENIZER: Tokenizer | None = None


def _initialize_worker(model_dir: str) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = Tokenizer.from_files(
        str(Path(model_dir) / "vocab.json"),
        str(Path(model_dir) / "merges.json"),
        [SPECIAL_TOKEN],
    )
    max_token_id = max(_WORKER_TOKENIZER.vocab)
    if max_token_id > np.iinfo(TOKEN_DTYPE).max:
        raise ValueError(f"token ID {max_token_id} cannot fit in {TOKEN_DTYPE}")


def _text_chunks(
    source: Path,
    start: int,
    end: int,
) -> Iterator[str]:
    decoder = codecs.getincrementaldecoder("utf-8")()
    remaining = end - start
    with source.open("rb") as f:
        f.seek(start)
        while remaining:
            data = f.read(min(CHUNK_SIZE_BYTES, remaining))
            if not data:
                raise OSError(f"Unexpected EOF while reading {source}")
            remaining -= len(data)
            text = decoder.decode(data, final=remaining == 0)
            if text:
                yield text


def _encode_partition(
    task: tuple[str, int, int, str, int],
) -> tuple[str, int]:
    source_name, start, end, output_name, part_number = task
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("Tokenizer worker was not initialized")
    source = Path(source_name)
    output = Path(output_name)
    print(
        f"Encoding {source.name} partition {part_number}: "
        f"{start:,}-{end:,} bytes",
        flush=True,
    )

    count = 0
    ids = _WORKER_TOKENIZER.encode_iterable(_text_chunks(source, start, end))
    with output.open("wb") as f:
        while True:
            chunk = np.fromiter(islice(ids, CHUNK_SIZE_IDS), dtype=TOKEN_DTYPE)
            if not chunk.size:
                break
            chunk.tofile(f)
            count += chunk.size

    print(
        f"Finished {source.name} partition {part_number}: {count:,} IDs",
        flush=True,
    )
    return str(output), count


def _merge_partitions(
    output: Path,
    parts: list[tuple[str, int]],
) -> int:
    total_ids = sum(count for _, count in parts)
    temp_output = output.with_name(output.name + ".tmp")
    try:
        with temp_output.open("wb") as f:
            np.lib.format.write_array_header_1_0(
                f,
                {
                    "descr": TOKEN_DTYPE.str,
                    "fortran_order": False,
                    "shape": (total_ids,),
                },
            )
            for part_path, expected_count in parts:
                part_path_obj = Path(part_path)
                expected_bytes = expected_count * TOKEN_DTYPE.itemsize
                if part_path_obj.stat().st_size != expected_bytes:
                    raise OSError(
                        f"Unexpected size for {part_path_obj}: "
                        f"{part_path_obj.stat().st_size} != {expected_bytes}"
                    )
                with part_path_obj.open("rb") as part:
                    shutil.copyfileobj(part, f, length=8 << 20)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_output, output)
    finally:
        temp_output.unlink(missing_ok=True)

    encoded = np.load(output, mmap_mode="r")
    if encoded.dtype != np.uint16 or encoded.shape != (total_ids,):
        raise RuntimeError(
            f"Invalid saved array {output}: {encoded.dtype} {encoded.shape}"
        )
    return total_ids


def _encode_file(
    executor: ProcessPoolExecutor,
    source: Path,
    output: Path,
    workers: int,
) -> None:
    with source.open("rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            desired_num_chunks=workers * 2,
            split_special_token=SPECIAL_TOKEN_BYTES,
        )
    tasks: list[tuple[str, int, int, str, int]] = []
    with tempfile.TemporaryDirectory(
        prefix=f"{output.stem}-parts-", dir=output.parent
    ) as temp_dir:
        for part_number, (start, end) in enumerate(
            zip(boundaries[:-1], boundaries[1:])
        ):
            part_path = Path(temp_dir) / f"part-{part_number:03}.u16"
            tasks.append((str(source), start, end, str(part_path), part_number))
        if not tasks:
            raise ValueError(f"No input chunks found for {source}")
        parts = list(executor.map(_encode_partition, tasks, chunksize=1))
        total_ids = _merge_partitions(output, parts)
    print(
        f"Saved {total_ids:,} uint16 IDs to {output} "
        f"({output.stat().st_size:,} bytes)",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode TinyStories and OpenWebText datasets in parallel."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(os.cpu_count() or 1, 6),
        help="Number of worker processes (default: up to 6).",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    data_dir = Path("data")
    artifacts_dir = Path("artifacts")
    output_dir = data_dir / "tokenized"
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        (
            "tinystories_bpe",
            [
                (
                    data_dir / "TinyStoriesV2-GPT4-train.txt",
                    output_dir / "tinystories_train.npy",
                ),
                (
                    data_dir / "TinyStoriesV2-GPT4-valid.txt",
                    output_dir / "tinystories_valid.npy",
                ),
            ],
        ),
        (
            "owt_train_bpe",
            [
                (data_dir / "owt_train.txt", output_dir / "owt_train.npy"),
                (data_dir / "owt_valid.txt", output_dir / "owt_valid.npy"),
            ],
        ),
    ]

    started = time.perf_counter()
    for model_name, files in datasets:
        model_dir = artifacts_dir / model_name
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_initialize_worker,
            initargs=(str(model_dir),),
        ) as executor:
            for source, output in files:
                if not source.is_file():
                    raise FileNotFoundError(source)
                _encode_file(executor, source, output, args.workers)
    print(f"All datasets encoded in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
