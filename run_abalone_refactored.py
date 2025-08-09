#!/usr/bin/env python3
"""
Refactored Abalone dataset runner with custom options.
"""

from src.base_runner import BaseRunner
from src.cli_args import get_base_parser, add_data_dir_arg
from datasets.abalone_loader import load_abalone_matrices
from pathlib import Path
import numpy as np


class AbaloneRunner(BaseRunner):
    """Custom runner for Abalone dataset with matrix format options."""
    
    def __init__(self):
        super().__init__("abalone")
    
    def load_data(self, args):
        # Load Abalone dataset with specified option
        X_mats, y, meta = load_abalone_matrices(args.data_dir, option=args.option)
        # Store option in metadata for custom plot filename
        meta['option'] = args.option
        return X_mats, y, meta
    
    def run(self, args):
        # Override run to customize plot filename with option
        X_matrices, y, metadata = self.load_data(args)
        print(f"Loaded {self.dataset_name}: matrices={X_matrices.shape}, labels={y.shape}")
        if "matrix_shape" in metadata:
            print(f"  Matrix shape: {metadata['matrix_shape']}")
        
        # Check if we have enough classes
        n_clusters = len(set(int(v) for v in y))
        if n_clusters < 2:
            print("Not enough classes for clustering metrics.")
            return
        
        # Continue with standard pipeline
        from pbp_transform import matrices_to_pbp_vectors
        from src.pipeline import filter_zero_columns, cluster_and_predict
        from src.metrics import calculate_all_metrics
        from src.utils import print_metrics_summary
        from visualize import scatter_features
        
        X_pbp = matrices_to_pbp_vectors(X_matrices, agg=args.agg)
        X_pbp = np.asarray(X_pbp)
        print(f"PBP vectors: {X_pbp.shape} (agg={args.agg})")
        
        X_pbp = filter_zero_columns(X_pbp)
        
        if args.plot:
            # Custom filename with option
            out_png = str(Path(args.results_dir) / f"abalone_targets_pbp_{metadata['option']}_{args.agg}.png")
            title = f"Abalone PBP (opt={metadata['option']}, agg={args.agg}) - Rings"
            scatter_features(X_pbp, y, out_png, title=title)
            print(f"Saved: {out_png}")
        
        pred, km = cluster_and_predict(X_pbp, n_clusters=n_clusters)
        
        cv_splits = getattr(args, "cv_splits", 5)
        all_metrics = calculate_all_metrics(X_pbp, y, pred, km, cv_splits)
        
        print_metrics_summary(
            all_metrics["cluster"],
            all_metrics["supervised"],
            {
                "dataset_name": "Abalone",
                "agg_func": args.agg,
                "n_samples": X_pbp.shape[0],
                "n_features": X_pbp.shape[1],
                "n_clusters": n_clusters
            }
        )


def main():
    # Setup argument parser
    parser = get_base_parser(
        description="Run PBP on Abalone (Option A 3x1, Option B 2x4) and report metrics/plot."
    )
    parser = add_data_dir_arg(parser, default="./data/abalone")
    parser.add_argument(
        "--option", choices=["A", "B"], default="B",
        help="Matrix format option: A=(3x1), B=(2x4)"
    )
    parser.set_defaults(cv_splits=5)  # Abalone uses 5 CV splits
    args = parser.parse_args()
    
    # Create and run the pipeline
    runner = AbaloneRunner()
    runner.run(args)


if __name__ == "__main__":
    main()