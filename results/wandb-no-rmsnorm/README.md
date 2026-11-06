# No-RMSNorm W&B export

Snapshot exported at **2026-11-06 09:00:53 UTC** from
`yangjuns-cmu-google/cs336-assignment1`. `runs.json` records each run's W&B
state, configuration, summary, final recorded step, row counts, and run link.
The `1e-4` run was marked `running` at export time, so this is not necessarily
its final history.

Each history CSV contains `_step`, `_timestamp`, `_runtime`, `train_loss`,
`val_loss`, `lr`, `elapsed_s`, and `peak_gpu_mem_gb`; blank cells indicate a
metric was not logged at that step. `_timestamp` is Unix time in seconds.
The unsampled W&B history was retrieved in step windows; the number of rows
matched W&B's numeric `_step` count at retrieval.

The `--gpu-utilization.csv` files contain separate W&B system-event samples
(`_timestamp`, `_runtime`, and `gpu_utilization_percent`, corresponding to
`system.gpu.0.gpu`). Each request asked for up to 10,000 events. W&B's
recorded counts differed from returned sample counts, so GPU sample
completeness is unverified. Other system metrics and run files are not
included. The existing RMSNorm comparison run is in
[the sweep export](../wandb-sweeps/README.md).
