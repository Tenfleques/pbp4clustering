
#!/usr/bin/env python3
"""
Refactored Iris dataset runner using the new modular architecture.
This demonstrates the simplified structure with shared modules.
"""

from src.base_runner import SimpleRunner
from src.cli_args import get_base_parser
from datasets.iris_loader import load_iris_matrices


def main():
    # Setup argument parser
    parser = get_base_parser(
        description="Run PBP on Iris with 2x2 matrices and report metrics."
    )
    parser.set_defaults(cv_splits=5)  # Iris uses 5 CV splits
    args = parser.parse_args()
    
    # Create and run the pipeline
    runner = SimpleRunner("iris", load_iris_matrices)
    runner.run(args)


if __name__ == "__main__":
    main()