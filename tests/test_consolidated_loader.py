#!/usr/bin/env python3
"""
Test script for the consolidated dataset loader.

This script demonstrates the functionality of the consolidated loader
and tests loading datasets from different categories.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data.consolidated_loader import ConsolidatedDatasetLoader

def test_consolidated_loader():
    """Test the consolidated loader functionality."""
    print("=== Testing Consolidated Dataset Loader ===\n")
    
    # Initialize the consolidated loader
    loader = ConsolidatedDatasetLoader()
    
    # Test 1: Validate configuration
    print("1. Validating dataset configuration...")
    issues = loader.validate_dataset_config()
    
    if any(issues.values()):
        print("Configuration issues found:")
        for issue_type, items in issues.items():
            if items:
                print(f"  {issue_type}: {items}")
        return False
    else:
        print("✓ Configuration validation passed")
    
    # Test 2: Get available datasets
    print("\n2. Available datasets by category:")
    available_datasets = loader.get_available_datasets()
    for category, datasets in available_datasets.items():
        print(f"  {category}: {len(datasets)} datasets")
        for dataset in datasets:
            print(f"    - {dataset}")
    
    # Test 3: Load a few datasets from different categories
    print("\n3. Testing dataset loading...")
    
    # Test standard datasets
    print("\n   Testing standard datasets...")
    standard_datasets = ['iris', 'wine']
    for dataset_name in standard_datasets:
        if dataset_name in available_datasets.get('standard', []):
            print(f"    Loading {dataset_name}...")
            dataset = loader.load_dataset(dataset_name)
            if dataset is not None:
                print(f"    ✓ {dataset_name}: {dataset['X'].shape}")
            else:
                print(f"    ✗ Failed to load {dataset_name}")
    
    # Test UCI datasets
    print("\n   Testing UCI datasets...")
    uci_datasets = ['glass', 'seeds']
    for dataset_name in uci_datasets:
        if dataset_name in available_datasets.get('uci', []):
            print(f"    Loading {dataset_name}...")
            dataset = loader.load_dataset(dataset_name)
            if dataset is not None:
                print(f"    ✓ {dataset_name}: {dataset['X'].shape}")
            else:
                print(f"    ✗ Failed to load {dataset_name}")
    
    # Test 4: Load datasets by category
    print("\n4. Testing category-based loading...")
    
    # Test standard category
    print("\n   Loading all standard datasets...")
    standard_results = loader.load_datasets_by_category('standard')
    successful_standard = sum(1 for result in standard_results.values() if result is not None)
    print(f"    ✓ Standard: {successful_standard}/{len(standard_results)} datasets loaded")
    
    # Test 5: Get dataset information
    print("\n5. Testing dataset information retrieval...")
    
    test_datasets = ['iris', 'glass']
    for dataset_name in test_datasets:
        info = loader.get_dataset_info(dataset_name)
        if info is not None:
            print(f"    {dataset_name}: {info['shape']}, {info['n_classes']} classes, {info['data_type']}")
        else:
            print(f"    {dataset_name}: No information available")
    
    # Test 6: Save configuration
    print("\n6. Testing configuration management...")
    
    # Add a test dataset
    loader.add_dataset('standard', 'test_dataset')
    print("    ✓ Added test_dataset to standard category")
    
    # Remove the test dataset
    loader.remove_dataset('standard', 'test_dataset')
    print("    ✓ Removed test_dataset from standard category")
    
    # Test 7: Save summary
    print("\n7. Testing summary generation...")
    summary = loader.save_dataset_summary('test_dataset_summary.json')
    print(f"    ✓ Summary saved with {summary['total_datasets']} total datasets")
    
    print("\n=== All tests completed successfully! ===")
    return True

def main():
    """Main function to run the test."""
    try:
        success = test_consolidated_loader()
        if success:
            print("\n✓ All tests passed!")
        else:
            print("\n✗ Some tests failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 