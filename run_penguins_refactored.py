#!/usr/bin/env python3
"""
Refactored Penguins dataset runner using the new modular architecture.
"""

from src.base_runner import SimpleRunner
from src.cli_args import get_base_parser, add_data_dir_arg
from datasets.penguins_loader import load_penguins_matrices


def main():
    # Setup argument parser
    parser = get_base_parser(
        description="Run PBP on Penguins dataset and report metrics."
    )
    parser = add_data_dir_arg(parser, default="./data/penguins")
    parser.set_defaults(cv_splits=5)  # Penguins uses 5 CV splits
    args = parser.parse_args()
    
    # Create and run the pipeline
    runner = SimpleRunner("penguins", load_penguins_matrices)
    runner.run(args)


if __name__ == "__main__":
    main()