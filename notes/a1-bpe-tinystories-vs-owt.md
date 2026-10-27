# BPE tokenizer: TinyStories vs OpenWebText

Artifacts (gitignored): `assignment1-basics/artifacts/tinystories_bpe/` (10k vocab) and `assignment1-basics/artifacts/owt_train_bpe/` (32k vocab).
Comparison is from vocab/merge statistics only; no compression-ratio (bytes/token) measurement yet.

## Stats

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

## Observations

- Both start with the same kind of merges: ` t`, ` a`, `he`, `in`, `re` (space attached to the word), and token length grows with merge rank.
- TinyStories tokens are almost all plain English words (` accomplishment`, ` disappointment`, ` responsibility`) from a small child-story vocabulary.
- OWT also learns web/wiki fragments (`Wik`, `Hak`) and many non-ASCII tokens.
- OWT's longest tokens are junk/formatting: runs of `----`, `____`, `====`, `....`, em-dashes, and the mojibake sequence `ÃÂÃÂ...` (the corpus has encoding corruption).
- At equal ranks OWT tokens are slightly shorter; its higher overall mean comes from the bigger vocab and the long junk tokens.
- OWT has fewer space-prefixed tokens because of more punctuation, symbols and non-ASCII tokens.

## Learnings

- Vocabulary reflects corpus domain and cleanliness: clean narrow data gives word-like tokens; noisy web data spends vocab slots on junk runs.
- Pre-tokenization lets long repeated-character runs become single tokens, so they can waste vocab entries.
- OWT training cost (32k vocab, 8 processes): 1h10m wall-clock, ~27 GB peak RSS, on a remote GCP CPU. Commit used: `5632caa`.
- The `owe_train_bpe` output dir name in the remote log was a typo; renamed locally to `owt_train_bpe`.

## Next steps

- Measure bytes/token on the TinyStories and OWT validation sets with each tokenizer.
- Cross-evaluate: TinyStories tokenizer on OWT and vice versa.
