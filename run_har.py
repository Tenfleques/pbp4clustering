#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np

from datasets.har_loader import (
    load_har_six_axis,
    load_har_axis_feature_format,
    load_har_axis_feature_columns,
)
from pbp_transform import matrices_to_pbp_vectors
from visualize import scatter_features


def main():
    parser = argparse.ArgumentParser(description="Run PBP on UCI HAR (6-axis) and plot targets.")
    parser.add_argument("--data-dir", default="./data/har", help="Download/cache directory for HAR")
    parser.add_argument("--results-dir", default="./results", help="Directory to write outputs")
    parser.add_argument("--agg", default="sum", help="Aggregation function name (e.g., sum, mean, median, adaptive)")
    parser.add_argument("--rows", type=int, default=None, help="Optionally reduce rows (m) to this value via PCA before PBP (e.g., 2 or 3)")
    parser.add_argument("--walking-idle", action="store_true", help="Map labels to binary: Walking (1) vs Idle (0)")
    parser.add_argument("--axis-feature-format", action="store_true", help="Use 3-row axis format with concatenated feature columns (x,y,z)")
    parser.add_argument("--include-body-acc", action="store_true", help="Include body_acc features when using axis-feature-format")
    parser.add_argument("--axis-feature-columns", action="store_true", help="Use 3-row axis format with standalone feature columns aggregated over time")
    parser.add_argument(
        "--time-agg",
        default="mean",
        choices=[
            "mean",
            "median",
            "sum",
            "min",
            "max",
            "std",
            "var",
            "rms",
            "absmean",
            "mad",
            "iqr",
            "energy",
            "entropy",
        ],
        help="Aggregation over time when using --axis-feature-columns",
    )
    args = parser.parse_args()

    if args.axis_feature_columns:
        X_mats, y, meta = load_har_axis_feature_columns(
            args.data_dir,
            include_body_acc=args.include_body_acc,
            time_agg=args.time_agg,
        )
    elif args.axis_feature_format:
        X_mats, y, meta = load_har_axis_feature_format(args.data_dir, include_body_acc=args.include_body_acc)
    else:
        X_mats, y, meta = load_har_six_axis(args.data_dir)
    print(f"Loaded HAR: matrices={X_mats.shape}, labels={y.shape}, m×n={meta['matrix_shape']}")

    if args.walking_idle:
        walking_set = {1, 2, 3}
        y = np.array([1 if int(lbl) in walking_set else 0 for lbl in y], dtype=int)
        meta["labels_map"] = {0: "Idle", 1: "Walking"}

    X_pbp = matrices_to_pbp_vectors(X_mats, agg=args.agg, rows_target=args.rows)
    print(f"PBP vectors: {X_pbp.shape} (agg={args.agg})")

    X_pbp = np.array(X_pbp)
    non_zero_cols = ~(np.all(X_pbp == 0, axis=0))
    X_pbp = X_pbp[:, non_zero_cols]

    # print(X_pbp[:2])
    # print(f"PBP vectors: {X_pbp.shape} (agg={args.agg})")

    out_png = str(Path(args.results_dir) / f"har_targets_pbp_{args.agg}.png")
    scatter_features(X_pbp, y, out_png, title=f"HAR PBP (agg={args.agg}) - True Targets", label_names=meta.get("labels_map"))
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main() 