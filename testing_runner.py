#!/usr/bin/env python3
"""
Testing Runner - Entry point for dataset testing functionality

This script provides access to the dataset testing functionality from the src structure.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.analysis.testing import DatasetTester

def main():
    """Main function to run dataset testing."""
    tester = DatasetTester('./data')
    results = tester.test_all_datasets()
    
    if results:
        print(f"\n✅ Dataset testing completed successfully!")
        print(f"Results for {len(results)} datasets generated.")
    else:
        print(f"\n❌ Dataset testing failed or no results generated.")


if __name__ == "__main__":
    main() 