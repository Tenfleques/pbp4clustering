# Clustering (Clean, Modular, Real-Data)

This directory contains a minimal, modular pipeline that uses only the PBP core and aggregation functions from `src/pbp` on real-world datasets.

Current dataset:
- UCI HAR (Human Activity Recognition): per-sample 6×128 matrices from accelerometer + gyroscope; true activity labels only.

## Run HAR

```bash
.venv/bin/python clustering/run_har.py --data-dir ./data/har --results-dir ./results --agg sum
```

Outputs:
- PCA scatter colored by true targets: `clustering/results/har_targets_pbp_sum.png`

Design:
- `clustering/datasets/har_loader.py`: downloads/loads HAR, builds (N, 6, 128) matrices
- `clustering/pbp_transform.py`: converts matrices to PBP vectors via `src.pbp.core.pbp_vector`
- `clustering/visualize.py`: simple PCA scatter using true targets
- `clustering/run_har.py`: glue script

Notes:
- No synthetic/fake data is used here.
- Only `src/pbp/core.py` and `src/pbp/aggregation_functions.py` are imported. 