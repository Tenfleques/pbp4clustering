#!/usr/bin/env python3
"""
Refactored Wine dataset runner with custom preprocessing.
"""

from src.base_runner import BaseRunner
from src.cli_args import get_base_parser, add_data_dir_arg
from src.utils import aggregate_rows_in_blocks
from datasets.wine_loader import load_wine_quality
import numpy as np


class WineRunner(BaseRunner):
    """Custom runner for Wine dataset with row block aggregation."""
    
    def __init__(self):
        super().__init__("wine")
    
    def load_data(self, args):
        # Load Wine Quality dataset
        X_mats, y, meta = load_wine_quality(args.data_dir, merge_types=True)
        return X_mats, y, meta
    
    def preprocess_matrices(self, X_matrices, args):
        # Aggregate features into blocks
        X_agg = aggregate_rows_in_blocks(X_matrices, args.row_blocks)
        print(f"Row-block aggregation to {args.row_blocks}: shape={X_agg.shape}")
        return X_agg


def main():
    # Setup argument parser
    parser = get_base_parser(
        description="Run PBP on UCI Wine Quality with 2/3-row inputs and report metrics."
    )
    parser = add_data_dir_arg(parser, default="./data/wine")
    parser.add_argument(
        "--row-blocks", type=int, choices=[2, 3], default=3,
        help="Aggregate feature rows into this many blocks before PBP"
    )
    parser.set_defaults(cv_splits=3)  # Wine uses 3 CV splits
    args = parser.parse_args()
    
    # Create and run the pipeline
    runner = WineRunner()
    runner.run(args)


if __name__ == "__main__":
    main()