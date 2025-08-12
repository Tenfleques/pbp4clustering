"""
Command-line argument parsing utilities.
Provides standardized argument parsers for the clustering project.
"""

import argparse
from typing import Optional


def get_base_parser(description: Optional[str] = None) -> argparse.ArgumentParser:
    """
    Create base argument parser with common arguments.
    
    Args:
        description: Optional description for the parser
        
    Returns:
        ArgumentParser with common arguments
    """
    if description is None:
        description = "Run PBP clustering pipeline with metrics reporting"
    
    parser = argparse.ArgumentParser(description=description)
    
    # Common arguments across all runners
    parser.add_argument(
        "--agg", 
        default="sum", 
        help="Aggregation function name for PBP transformation"
    )
    
    parser.add_argument(
        "--results-dir", 
        default="./results", 
        help="Directory to write plot outputs"
    )
    
    parser.add_argument(
        "--plot", 
        dest="plot", 
        action="store_true", 
        default=True,
        help="Generate scatter plot of PBP features (default)"
    )
    
    parser.add_argument(
        "--no-plot", 
        dest="plot", 
        action="store_false",
        help="Disable plotting"
    )
    
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=3,
        help="Number of cross-validation splits for supervised metrics (default: 3)"
    )
    
    parser.add_argument(
        "--show-fig",
        action="store_true",
        help="Display plots interactively (in addition to saving them)"
    )
    
    return parser


def add_data_dir_arg(
    parser: argparse.ArgumentParser, 
    default: Optional[str] = None,
    required: bool = False
) -> argparse.ArgumentParser:
    """
    Add data directory argument to parser.
    
    Args:
        parser: Existing ArgumentParser
        default: Default data directory path
        required: Whether the argument is required
        
    Returns:
        Modified ArgumentParser
    """
    parser.add_argument(
        "--data-dir",
        default=default,
        required=required,
        help="Directory containing the dataset files"
    )
    return parser


def add_dataset_specific_args(
    parser: argparse.ArgumentParser, 
    dataset_name: str
) -> argparse.ArgumentParser:
    """
    Add dataset-specific arguments based on dataset name.
    
    Args:
        parser: Existing ArgumentParser
        dataset_name: Name of the dataset
        
    Returns:
        Modified ArgumentParser with dataset-specific arguments
    """
    # HAR-specific arguments
    if dataset_name.lower() == "har":
        parser.add_argument(
            "--include-total-acc",
            action="store_true",
            help="Include total acceleration in HAR data"
        )
        parser.add_argument(
            "--include-body-acc",
            action="store_true",
            help="Include body acceleration in HAR data"
        )
        parser.add_argument(
            "--include-gyro",
            action="store_true",
            help="Include gyroscope data in HAR data"
        )
        parser.add_argument(
            "--axis-feature-format",
            action="store_true",
            help="Use axis-feature format instead of feature-axis"
        )
        parser.add_argument(
            "--row-major",
            action="store_true",
            help="Use row-major ordering for HAR data"
        )
    
    # Wine-specific arguments
    elif dataset_name.lower() == "wine":
        parser.add_argument(
            "--aggregate-to-2x2",
            action="store_true",
            help="Aggregate wine features to 2x2 matrices"
        )
    
    # Spectroscopy-specific arguments
    elif dataset_name.lower() == "spectroscopy":
        parser.add_argument(
            "--aggregate-to-6x6",
            action="store_true",
            help="Aggregate spectroscopy features to 6x6 matrices"
        )
    
    return parser


def parse_args_with_defaults(
    dataset_name: str,
    description: Optional[str] = None,
    data_dir_default: Optional[str] = None,
    data_dir_required: bool = False
) -> argparse.Namespace:
    """
    Convenience function to create parser and parse arguments in one call.
    
    Args:
        dataset_name: Name of the dataset for specific arguments
        description: Optional parser description
        data_dir_default: Default data directory
        data_dir_required: Whether data directory is required
        
    Returns:
        Parsed arguments namespace
    """
    parser = get_base_parser(description)
    
    if data_dir_default is not None or data_dir_required:
        parser = add_data_dir_arg(parser, data_dir_default, data_dir_required)
    
    parser = add_dataset_specific_args(parser, dataset_name)
    
    return parser.parse_args()