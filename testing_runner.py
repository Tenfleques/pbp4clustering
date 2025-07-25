#!/usr/bin/env python3
"""
Testing Runner - Entry point for dataset testing functionality

This script provides access to the dataset testing functionality from the src structure.
Now includes aggregation function optimization and comprehensive dataset testing.
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.analysis.testing import DatasetTester

def main():
    """Main function to run dataset testing."""
    parser = argparse.ArgumentParser(description="Dataset testing with aggregation optimization")
    parser.add_argument("-d", "--dataset", default='all', help="Specific dataset to test")
    parser.add_argument("--no-optimization", action="store_true", help="Disable aggregation function optimization")
    parser.add_argument("--data-dir", default='./data', help="Data directory")
    parser.add_argument("--results-dir", default='./results', help="Results directory")
    args = parser.parse_args()
    
    # Create directories
    data_dir = args.data_dir
    results_dir = args.results_dir
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f"{results_dir}/figures", exist_ok=True)
    os.makedirs(f"{results_dir}/tables", exist_ok=True)
    
    # Initialize tester with optimization
    use_optimization = not args.no_optimization
    tester = DatasetTester(data_dir, results_dir, use_optimized_aggregation=use_optimization)
    
    if args.dataset == 'all':
        print("=" * 80)
        print("COMPREHENSIVE DATASET TESTING WITH AGGREGATION OPTIMIZATION")
        print("=" * 80)
        print(f"Data directory: {data_dir}")
        print(f"Results directory: {results_dir}")
        print(f"Aggregation optimization: {'Enabled' if use_optimization else 'Disabled'}")
        print()
        
        results = tester.test_all_datasets()
        
        if results:
            print(f"\n✅ Comprehensive dataset testing completed successfully!")
            print(f"Results for {len(results)} datasets generated.")
            print(f"Check {results_dir} for saved visualizations and summary.")
        else:
            print(f"\n❌ Comprehensive dataset testing failed or no results generated.")
    else:
        print(f"Testing specific dataset: {args.dataset}")
        result = tester.test_dataset(args.dataset)
        
        if result:
            print(f"\n✅ Dataset testing completed successfully!")
            print(f"Results for {args.dataset} generated.")
        else:
            print(f"\n❌ Dataset testing failed for {args.dataset}.")


if __name__ == "__main__":
    main() 