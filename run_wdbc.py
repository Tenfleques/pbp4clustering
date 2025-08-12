#!/usr/bin/env python3
"""
Refactored WDBC dataset runner using the new modular architecture.
"""

from src.base_runner import SimpleRunner
from src.cli_args import get_base_parser
from datasets.wdbc_loader import load_wdbc_matrices


def main():
    # Setup argument parser (WDBC doesn't use data-dir)
    parser = get_base_parser(
        description="Run PBP on WDBC dataset and report metrics."
    )
    parser.set_defaults(cv_splits=5)  # WDBC uses 5 CV splits
    args = parser.parse_args()
    
    # Create and run the pipeline
    runner = SimpleRunner("wdbc", load_wdbc_matrices)
    runner.run(args)


if __name__ == "__main__":
    main()