#!/usr/bin/env python3
"""
Comparison Runner - Entry point for comprehensive comparison functionality

This script provides access to the comprehensive comparison functionality from the src structure.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.analysis.comparison import ComprehensiveComparison

def main():
    """Main function to run comprehensive comparison."""
    comparison = ComprehensiveComparison('./data')
    results = comparison.run_comprehensive_comparison()
    
    if results:
        print(f"\n✅ Comprehensive comparison completed successfully!")
        print(f"Results for {len(results)} datasets generated.")
    else:
        print(f"\n❌ Comprehensive comparison failed or no results generated.")


if __name__ == "__main__":
    main() 