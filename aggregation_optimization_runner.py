#!/usr/bin/env python3
"""
Aggregation Function Optimization Runner

This script tests different aggregation functions on datasets and finds
the optimal one for each dataset based on clustering performance.
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append('src')

from src.data.consolidated_loader import ConsolidatedDatasetLoader
from src.analysis.aggregation_optimization import AggregationOptimizer, create_aggregation_comparison_report


def main():
    """Main function to run aggregation optimization."""
    print("=" * 80)
    print("AGGREGATION FUNCTION OPTIMIZATION RUNNER")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize loader
    print("Loading datasets...")
    loader = ConsolidatedDatasetLoader()
    
    # Get dataset configuration
    dataset_config = loader.get_available_datasets()
    print(f"Found {sum(len(datasets) for datasets in dataset_config.values())} datasets across {len(dataset_config)} categories")
    
    # Select datasets to test (limit to a reasonable number for testing)
    test_datasets = [
        'iris', 'breast_cancer', 'wine', 'digits', 'diabetes',
        'sonar', 'glass', 'seeds', 'thyroid', 'pima'
    ]
    
    print(f"Testing aggregation functions on {len(test_datasets)} datasets:")
    for dataset_name in test_datasets:
        print(f"  - {dataset_name}")
    print()
    
    # Load datasets
    datasets = {}
    for dataset_name in test_datasets:
        try:
            print(f"Loading {dataset_name}...")
            dataset = loader.load_dataset(dataset_name)
            if dataset is not None:
                datasets[dataset_name] = dataset
                print(f"  ✓ Loaded: {dataset['X'].shape}")
            else:
                print(f"  ✗ Failed to load {dataset_name}")
        except Exception as e:
            print(f"  ✗ Error loading {dataset_name}: {e}")
    
    print(f"\nSuccessfully loaded {len(datasets)} datasets")
    
    if not datasets:
        print("No datasets loaded. Exiting.")
        return
    
    # Initialize optimizer
    print("\nInitializing aggregation optimizer...")
    optimizer = AggregationOptimizer(random_state=42)
    
    # Run optimization
    print("\nRunning aggregation function optimization...")
    results = optimizer.optimize_multiple_datasets(datasets)
    
    # Create results directory
    results_dir = Path('aggregation_optimization_results')
    results_dir.mkdir(exist_ok=True)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save raw results
    results_file = results_dir / f'aggregation_optimization_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json_results = json.loads(json.dumps(results, default=lambda x: float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else x))
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Create and save report
    report_file = results_dir / f'aggregation_optimization_report_{timestamp}.md'
    report_content = create_aggregation_comparison_report(results, str(report_file))
    
    print(f"Report saved to: {report_file}")
    
    # Print summary
    summary = results['summary']
    print(f"\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Total datasets tested: {summary['total_datasets']}")
    print(f"Successful optimizations: {summary['successful_datasets']}")
    print(f"Success rate: {summary['successful_datasets']/summary['total_datasets']*100:.1f}%")
    
    if summary['best_functions']:
        print(f"\nBest aggregation functions by dataset:")
        for dataset_name, best_func in summary['best_functions'].items():
            print(f"  {dataset_name:20s}: {best_func}")
        
        # Count function usage
        function_counts = {}
        for func_name in summary['best_functions'].values():
            function_counts[func_name] = function_counts.get(func_name, 0) + 1
        
        print(f"\nAggregation function usage:")
        for func_name, count in sorted(function_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {func_name:20s}: {count} datasets")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main() 