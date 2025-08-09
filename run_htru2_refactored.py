#!/usr/bin/env python3
"""
Refactored HTRU2 dataset runner using the new modular architecture.
"""

from src.base_runner import SimpleRunner
from src.cli_args import get_base_parser, add_data_dir_arg
from datasets.htru2_loader import load_htru2_matrices


def main():
    # Setup argument parser
    parser = get_base_parser(
        description="Run PBP on HTRU2 dataset (pulsar data) and report metrics."
    )
    parser = add_data_dir_arg(parser, default="./data/htru2")
    parser.set_defaults(cv_splits=5)  # HTRU2 uses 5 CV splits
    args = parser.parse_args()
    
    # Create and run the pipeline
    runner = SimpleRunner("htru2", load_htru2_matrices)
    runner.run(args)


if __name__ == "__main__":
    main()