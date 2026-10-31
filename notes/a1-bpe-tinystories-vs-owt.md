# BPE tokenizer Learning Notes

## Part 1 - Implementation

### BPE data structures

- Use a doubly linked list per pre-token for constant-time neighbor lookup.
  A merge only touches k places, where k is the number of occurrences of the
  pair, instead of rescanning the whole sequence.
- Profile with py-spy. It was very useful for finding the real bottleneck.
- Finding the most frequent pair is the hot spot; optimize it with a max-heap.
  - A BST can also work, but each merge also changes the counts of
    neighboring pairs. A BST ordered by count cannot also be keyed by byte
    pair, so locating and updating an entry is expensive.
  - Common pattern: max-heap + hash table (pair -> current count) with lazy
    updates. Push a new entry when a count changes, and when popping, skip
    entries whose count no longer matches the table.
  - Amortized O(log k) per update.
### Saving vocab and merges to JSON: hex vs text

- Storing the hex of the token bytes is lossless: `bytes.fromhex(hex)`
  recovers the exact bytes.
- Storing the decoded text is lossy. Tokens are arbitrary byte sequences, and
  many (e.g. a lone `0xE2` or continuation bytes `0x80`-`0xBF`) are not valid
  UTF-8 on their own. Decoding them needs `errors="replace"`/`"ignore"`, so
  different byte sequences can collapse to the same text.
- Keep `hex` as the source of truth; `text` is only for human inspection.
- `json.load` on these files returns a list of dicts, so convert to
  `dict[int, bytes]` and `list[tuple[bytes, bytes]]` after loading.

### Special tokens

- Nearly every modern tokenizer defines them:
  - Document boundaries: `<|endoftext|>` (GPT-2/3/4), `<s>`/`</s>` (Llama 2, T5),
    `<|begin_of_text|>`/`<|end_of_text|>` (Llama 3).
  - Padding/unknown: `<pad>`, `<unk>` (byte-level BPE rarely needs `<unk>`).
  - Chat formatting: `<|im_start|>`, `<|im_end|>`, role headers.
  - Tool use / reasoning (`<think>`), multimodal placeholders, FIM markers.
- Training: they are reserved vocab entries, not text that feeds merge counting.
  - Declared up front with fixed IDs and count toward vocab size.
  - Corpus is split on them before pre-tokenization, so merges never cross
    them and never learn their characters (e.g. `<|endof` + `text|>`).
  - Parallel chunk boundaries are aligned to the special token.
- Encoding: split on special tokens, map each directly to its ID, run BPE on
  the text between. Match longest first when they overlap.
- Data pipeline: insert an EOS/separator between documents when tokenizing;
  many chat/tool tokens are added later in fine-tuning and their embeddings
  are trained then.

### Why so many tokens start with a space?

- 75% of the TinyStories vocab and 69% of OWT start with a space. This is expected, not a bug.
- The GPT-2 pre-tokenizer regex attaches a single leading space to each word (` ?\p{L}+`), and BPE merges never cross pre-token boundaries. Mid-sentence words almost always follow a space, so ` the` is the common form and `the` is rare.
- Why it's a good design:
  - Decoding is lossless with no separate whitespace tokens.
  - It avoids spending a token on every space, which would roughly double sequence length.
  - Word-initial and word-continuation pieces stay distinct (` un` vs `un`).
- Real downsides:
  - Duplicated vocab: `the`, ` the`, `The` and ` The` are separate.
  - Spacing sensitivity: a trailing space in a prompt tokenizes separately from the next word, which can behave oddly.
  - No benefit for languages without spaces (Chinese, Japanese).
- OWT has a lower share because of more punctuation, symbols, non-ASCII text and subword pieces.
- Things that would actually be a concern: whitespace-only tokens and junk runs (`----`, mojibake), not space-prefixed words.


## Part 2 - Datasets

TinyStories and OpenWebText tokenizer artifacts are gitignored:
`assignment1-basics/artifacts/tinystories_bpe/` (10K vocabulary) and
`assignment1-basics/artifacts/owt_train_bpe/` (32K vocabulary).

### Vocabulary statistics

| | TinyStories (10k) | OpenWebText (32k) |
|---|---|---|
| Merges | 9,743 | 31,743 |
| Mean token length (bytes, excl. specials/single bytes) | 5.92 | 6.38 |
| Longest token | 15 bytes | 64 bytes |
| Tokens starting with a space | 74.7% | 69.2% |
| Tokens with non-ASCII bytes | 20 | 431 |

Mean token length by merge rank:

| Merge ranks | TinyStories | OpenWebText |
|---|---|---|
| 257-1000 | 3.87 | 3.42 |
| 1000-5000 | 5.71 | 5.41 |
| 5000-10000 | 6.38 | 6.19 |

### Corpus observations

- Both start with the same kind of merges: ` t`, ` a`, `he`, `in`, `re` (space attached to the word), and token length grows with merge rank.
- TinyStories tokens are almost all plain English words (` accomplishment`, ` disappointment`, ` responsibility`) from a small child-story vocabulary.
- OWT also learns web/wiki fragments (`Wik`, `Hak`) and many non-ASCII tokens.
- OWT's longest tokens are junk/formatting: runs of `----`, `____`, `====`, `....`, em-dashes, and the mojibake sequence `ÃÂÃÂ...` (the corpus has encoding corruption).
- At equal ranks OWT tokens are slightly shorter; its higher overall mean comes from the bigger vocab and the long junk tokens.
- OWT has fewer space-prefixed tokens because of more punctuation, symbols and non-ASCII tokens.

## Part 3 - Training

The successful OpenWebText BPE training run used `owt_train.txt`, a 32K
vocabulary, and eight worker processes. The recorded command was:

```text
uv run python -u -m cs336_basics.train_bpe data/owt_train.txt --vocab-size 32000 --num-processes 8 --output-dir artifacts/owe_train_bpe
```

| Metric | Recorded result |
|---|---:|
| Wall-clock time | 1:10:25 |
| User CPU time | 8,504.70 s |
| System CPU time | 85.84 s |
| Average CPU utilization | 203% |
| Maximum resident set size | 27,033,932 KiB (~25.8 GiB) |
| Swap | 0 |
| Result | 32,000 vocabulary entries; 31,743 merges |

I rented a GCP E2 high-memory VM with 16 vCPUs and
128 GB RAM; treat those machine specifications as recalled rather than
independently verified by the log.

The pitfall: I first tried training on my local WSL:Ubuntu VM, which has 24 GB
RAM, and the attempt failed. I then moved the run to a rented GCP VM. .

## Part 3 - Encoding

### Sampled compression ratio

Sampled 10 documents from each validation split using `random.Random(42)`.
Documents were split on `<|endoftext|>`, stripped, and sampled independently
from their own corpus. Ratios are total UTF-8 document bytes divided by total
token IDs; separators are excluded. Each row uses the same documents for both
tokenizers, so the cross-tokenizer comparison is directly comparable.

| Sample documents | Bytes | Tokenizer | Tokens | Bytes/token |
|---|---:|---|---:|---:|
| TinyStories (10 docs) | 8,457 | TinyStories 10K | 2,084 | 4.058 |
| TinyStories (10 docs) | 8,457 | OpenWebText 32K | 2,145 | 3.943 |
| OpenWebText (10 docs) | 39,375 | TinyStories 10K | 12,439 | 3.165 |
| OpenWebText (10 docs) | 39,375 | OpenWebText 32K | 9,275 | 4.245 |

| Sample documents | TinyStories tokenizer | OpenWebText tokenizer |
|---|---:|---:|
| TinyStories | 4.058 bytes/token | 3.943 bytes/token |
| OpenWebText | 3.165 bytes/token | 4.245 bytes/token |

On these samples, the OpenWebText tokenizer's ratio on TinyStories is about
3% lower than the TinyStories tokenizer's. On OpenWebText, the TinyStories
tokenizer's ratio is about 25% lower than the OpenWebText tokenizer's. Lower
bytes/token means more token IDs are needed for the same document bytes.
Qualitatively, the TinyStories tokenizer splits some OpenWebText words into
more pieces (for example, `demographic` into `dem`, `og`, `ra`, `ph`), while
the OpenWebText tokenizer handles that word as one token.

### Full dataset token IDs

Each split was encoded with its corresponding tokenizer and serialized as a
one-dimensional `uint16` NumPy array. Token IDs fit in `uint16` (0-65,535):
the TinyStories IDs range from 0 to 9,999, and the OpenWebText IDs from 0 to
31,999. The arrays live under the gitignored `assignment1-basics/data/`
directory and can be memory-mapped with `np.load(path, mmap_mode="r")`.

| Dataset | Token IDs |
|---|---:|
| TinyStories train | 541,229,343 |
| TinyStories validation | 5,465,883 |
| OpenWebText train | 2,727,120,453 |
| OpenWebText validation | 66,401,098 |

The full encoding run processed all four splits with six worker processes.
Across the datasets, it produced 3,340,216,777 token IDs in 5,042.2 seconds
(about 1 hour 24 minutes), an aggregate rate of approximately 662,500
tokens/second. This is the elapsed time for the complete multi-split run, not
the per-split timing.

| Full-run metric | Result |
|---|---:|
| Worker processes | 6 |
| Splits encoded | 4 |
| Token IDs produced | 3,340,216,777 |
| Elapsed time | 5,042.2 s (1 h 24 min 2 s) |
| Aggregate throughput | ~662,500 tokens/s |

### Tokenization throughput on OpenWebText

Benchmarked all 59,059 documents in `owt_valid.txt`. The file is 289,998,753
bytes; after splitting on `<|endoftext|>` and stripping document boundaries,
the encoded text is 289,230,986 UTF-8 bytes. Data and tokenizer were loaded
before timing, the first document was used as a warm-up, and each document was
encoded separately. The separator is excluded. Each implementation was run
through the full dataset five times; rates below use aggregate work divided
by aggregate elapsed time for these full-dataset runs. The six-worker result
uses one run on a 30.0 MB slice and is shown separately for context.

| Implementation / encoding | Workload | Tokens | Time | Tokens/s | Decimal MB/s |
|---|---|---:|---:|---:|---:|
| OWT BPE 32K (`cs336_basics.Tokenizer`, Python), sequential | Full validation set, mean of 5 passes | 66,342,039/pass | 491.697 s/pass | 134,925 | 0.588 |
| `tiktoken` 0.12.0 (`gpt2`), sequential | Full validation set, mean of 5 passes | 65,794,501/pass | 31.211 s/pass | 2,108,028 | 9.267 |
| OWT BPE 32K, 6 worker processes | 30.0 MB validation slice, 1 run | 6,894,479 | 10.55 s | 653,500 | 2.84 |

On this machine and workload, `tiktoken` processed the input bytes about
15.8x faster. This is an implementation-throughput comparison, not a
tokenizer-quality comparison: the encodings and vocabularies differ, and the
custom tokenizer is a Python reference implementation while `tiktoken` is
optimized native code.

Afterward, my encoder was parallelized across six worker processes.
On a 30.0 MB OWT sample, it encoded 6,894,479 IDs in 10.55 seconds:
about 653,500 tokens/s or 2.84 MB/s. This is just over 5x the earlier
single-process rate of about 129,000 tokens/s measured on a separate 2.0 MB
slice. The comparison is encouraging, but not a controlled scaling benchmark:
the inputs differed, and throughput can vary with sample composition,
partitioning, CPU availability, and disk activity. It does show how separate
processes can use additional CPU cores for this Python implementation.

The `tiktoken` result above, about 2.1 million tokens/s, came from sequential
`encode()` calls over documents in one process, without explicit batching,
threading, or multiprocessing. It was not pinned to a single CPU core, so
describe it as a single-process sequential result rather than a strict
single-core measurement. It is an impressive reference point for an
optimized production tokenizer; direct speedup claims still need matching
workloads and machine conditions.

Machine/runtime: Intel Core i5-8400 @ 2.80 GHz, six CPUs exposed under WSL2
(Microsoft hypervisor), Linux 6.18, Python 3.13.16. The benchmark loop was
sequential and single-process. CPU model, available CPU time, virtualization,
load, and threading can all affect throughput, so report the hardware and
benchmark setup when comparing results.

For context, the [OpenAI `tiktoken` README performance section](https://github.com/openai/tiktoken#performance)
reports `tiktoken` as 3-6x faster than a comparable open-source tokenizer,
based on 1 GB of text and `GPT2TokenizerFast` (`tokenizers` 0.13.2,
`transformers` 4.24.0, `tiktoken` 0.2.0). That is not directly comparable to
this run because the machine, workload, versions, and reference implementation
differ.
