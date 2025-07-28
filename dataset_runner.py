#!/usr/bin/env python3
"""
Dataset Runner - Entry point for dataset loading functionality

This script provides access to the dataset loading functionality from the src structure.
"""

import sys
import os
import warnings
import logging

# Suppress all warnings and verbose output
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

# Suppress UMAP verbose output
os.environ['UMAP_VERBOSE'] = '0'

# Suppress other verbose libraries
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from src.data.consolidated_loader import ConsolidatedDatasetLoader

def main():
    """Main function to demonstrate dataset loading."""
    loader = ConsolidatedDatasetLoader()
    
    # Load all datasets
    datasets = loader.load_all_datasets()
    
    # Print information about each dataset
    for name, dataset in datasets.items():
        print(f"\n{name.upper()} DATASET:")
        print(f"  Shape: {dataset['X'].shape}")
        print(f"  Description: {dataset['description']}")
        print(f"  Number of classes: {len(np.unique(dataset['y']))}")


if __name__ == "__main__":
    main() 