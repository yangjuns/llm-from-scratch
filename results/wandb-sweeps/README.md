# W&B sweep export

Exported from `yangjuns-cmu-google/cs336-assignment1`. Each CSV contains
the unsampled history for one run, named by run name and W&B run ID.
`runs.json` records run states, configurations, summaries, history row
counts, and links to the original runs.

The history has `_step`, `_timestamp`, `_runtime`, `train_loss`, `val_loss`,
`lr`, `elapsed_s`, and `peak_gpu_mem_gb` columns. Empty cells indicate that
the metric was not logged for that step. `_timestamp` is Unix time in seconds.
The trainer uses `val_loss`, not `valid_loss`.

The export uses W&B's history API with unsampled step windows. Returned row
counts were checked against W&B's recorded `_step` counts. Other run files
are not included. The batch-256 run failed before
logging a step, so its CSV contains only the header. Some other batch runs
stopped before the planned token budget; consult their state and final step
in `runs.json`. For resumed runs, `elapsed_s` may reset and should not be
treated as total elapsed time without inspecting the history.

The `lr-*--gpu-utilization.csv` and `batch-*--gpu-utilization.csv` files are
separate GPU telemetry exports from W&B's system events, requested with a
10,000-event sample limit. Batch 256 returned no GPU samples.
They contain `_timestamp` (Unix seconds), `_runtime` (which can reset on
resume), and `gpu_utilization_percent` (`system.gpu.0.gpu`). W&B's recorded
metric counts did not match the returned GPU sample counts, so completeness
is unverified. Other system metrics are not included.
