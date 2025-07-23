#!/usr/bin/env python3
"""
Example script demonstrating how to use the feature analysis functionality.

This script shows how to analyze PBP features for different datasets and
interpret the results.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from feature_analysis import analyze_features, print_analysis_results, extract_top_features, visualize_features, load_dataset_data

def main():
    """Example usage of feature analysis."""
    
    # List of valid datasets for PBP analysis
    datasets = ['iris', 'breast_cancer']
    
    print("PBP Feature Analysis Examples")
    print("=" * 50)
    
    for dataset in datasets:
        try:
            print(f"\nAnalyzing dataset: {dataset}")
            print("-" * 30)
            
            # Perform analysis
            results = analyze_features(dataset, top_k=5)
            
            # Print results
            print_analysis_results(results, top_k=5)
            
            # Show key insights
            print(f"\nKey Insights for {dataset}:")
            if 'combined' in results['top_features']:
                top_feature_idx = results['top_features']['combined']['indices'][0]
                print(f"  - Most important feature: {top_feature_idx}")
                print(f"  - Combined score: {results['top_features']['combined']['scores'][0]:.4f}")
            
            # Extract and visualize top features
            try:
                print(f"\nExtracting top features for {dataset}...")
                features, targets = load_dataset_data(dataset)
                extracted_features, selected_indices = extract_top_features(features, results, k=3)
                
                print(f"  - Selected features: {selected_indices}")
                print(f"  - Extracted shape: {extracted_features.shape}")
                
                # Create visualization
                visualize_features(extracted_features, targets, selected_indices, dataset, save_plot=True)
                
            except Exception as e:
                print(f"  - Visualization error: {e}")
            
            print("\n" + "="*50)
            
        except Exception as e:
            print(f"Error analyzing {dataset}: {e}")
            continue

if __name__ == "__main__":
    main() 