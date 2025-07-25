#!/usr/bin/env python3
"""
Test Script for Optimized PBP Performance

This script demonstrates the improved performance of PBP dimensionality reduction
with aggregation function optimization compared to the default approach.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add src to path
sys.path.append('src')

from src.data.consolidated_loader import ConsolidatedDatasetLoader
from src.pbp.core import pbp_vector
from src.pbp.aggregation_functions import get_aggregation_function
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler


class OptimizedPBPComparison:
    """Compare optimized vs default PBP performance."""
    
    def __init__(self, data_dir='./data', results_dir='./results'):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.loader = ConsolidatedDatasetLoader()
        self.results = {}
        
    def test_dataset_comparison(self, dataset_name):
        """Compare default vs optimized PBP for a dataset."""
        print(f"\n{'='*60}")
        print(f"Testing {dataset_name}: Default vs Optimized PBP")
        print(f"{'='*60}")
        
        # Load dataset
        dataset = self.loader.load_dataset(dataset_name)
        if dataset is None:
            print(f"Failed to load {dataset_name}")
            return None
        
        X = dataset['X']
        y = dataset['y']
        
        print(f"Dataset shape: {X.shape}")
        print(f"Number of classes: {len(np.unique(y))}")
        
        results = {}
        
        # Test 1: Default sum aggregation
        print(f"\n1. Testing default sum aggregation...")
        default_results = self.test_pbp_with_aggregation(X, y, lambda x: x.sum(), "Default (sum)")
        results['default'] = default_results
        
        # Test 2: Optimized aggregation (if available)
        print(f"\n2. Testing optimized aggregation...")
        try:
            # Try to load cached optimal function
            cache_file = os.path.join(self.data_dir, f'{dataset_name}_optimal_aggregation.json')
            if os.path.exists(cache_file):
                import json
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    if 'best_function' in cache_data:
                        agg_func_name = cache_data['best_function']
                        agg_func = get_aggregation_function(agg_func_name)
                        optimized_results = self.test_pbp_with_aggregation(X, y, agg_func, f"Optimized ({agg_func_name})")
                        results['optimized'] = optimized_results
                        results['optimized']['agg_func_name'] = agg_func_name
                    else:
                        print("  No optimal function found in cache")
                        results['optimized'] = default_results
                        results['optimized']['agg_func_name'] = 'sum'
            else:
                print("  No optimization cache found, using default")
                results['optimized'] = default_results
                results['optimized']['agg_func_name'] = 'sum'
        except Exception as e:
            print(f"  Error in optimized test: {e}")
            results['optimized'] = default_results
            results['optimized']['agg_func_name'] = 'sum'
        
        # Test 3: Best aggregation functions from our optimization results
        print(f"\n3. Testing best aggregation functions...")
        best_functions = {
            'iris': 'entropy',
            'breast_cancer': 'robust_adaptive',
            'wine': 'iqr',
            'digits': 'adaptive',
            'diabetes': 'entropy',
            'sonar': 'sum',
            'glass': 'iqr',
            'seeds': 'sum',
            'thyroid': 'sum',
            'pima': 'iqr'
        }
        
        if dataset_name in best_functions:
            best_func_name = best_functions[dataset_name]
            best_func = get_aggregation_function(best_func_name)
            best_results = self.test_pbp_with_aggregation(X, y, best_func, f"Best ({best_func_name})")
            results['best'] = best_results
            results['best']['agg_func_name'] = best_func_name
        else:
            print("  No best function known for this dataset")
            results['best'] = default_results
            results['best']['agg_func_name'] = 'sum'
        
        # Print comparison
        self.print_comparison(dataset_name, results)
        
        return results
    
    def test_pbp_with_aggregation(self, X, y, agg_func, method_name):
        """Test PBP with a specific aggregation function."""
        print(f"  Computing PBP features with {method_name}...")
        
        reduced_samples = []
        for i in range(X.shape[0]):
            try:
                pbp_result = pbp_vector(X[i], agg_func)
                reduced_samples.append(pbp_result)
            except Exception as e:
                print(f"    Error processing sample {i}: {e}")
                reduced_samples.append(X[i].flatten())
        
        X_reduced = np.array(reduced_samples)
        
        # Remove zero columns
        zero_columns = np.all(X_reduced == 0, axis=0)
        X_reduced = X_reduced[:, ~zero_columns]
        
        print(f"    Reduced shape: {X_reduced.shape}")
        
        # Evaluate clustering
        if len(np.unique(y)) > 1:
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_reduced)
            
            # Apply K-means
            kmeans = KMeans(n_clusters=len(np.unique(y)), random_state=42, n_init=10)
            y_pred = kmeans.fit_predict(X_scaled)
            
            # Calculate metrics
            silhouette = silhouette_score(X_scaled, y_pred)
            davies_bouldin = davies_bouldin_score(X_scaled, y_pred)
            
            return {
                'method': method_name,
                'silhouette_score': silhouette,
                'davies_bouldin_score': davies_bouldin,
                'reduced_shape': X_reduced.shape,
                'success': True
            }
        else:
            return {
                'method': method_name,
                'silhouette_score': 0.0,
                'davies_bouldin_score': float('inf'),
                'reduced_shape': X_reduced.shape,
                'success': True
            }
    
    def print_comparison(self, dataset_name, results):
        """Print comparison results."""
        print(f"\n{'='*50}")
        print(f"COMPARISON RESULTS: {dataset_name}")
        print(f"{'='*50}")
        
        print(f"{'Method':<25} {'Silhouette':<12} {'Davies-Bouldin':<15} {'Shape':<15}")
        print("-" * 70)
        
        for method_key, result in results.items():
            if method_key == 'optimized':
                agg_func = result.get('agg_func_name', 'sum')
                method_display = f"Optimized ({agg_func})"
            elif method_key == 'best':
                agg_func = result.get('agg_func_name', 'sum')
                method_display = f"Best ({agg_func})"
            else:
                method_display = "Default (sum)"
            
            silhouette = result['silhouette_score']
            davies_bouldin = result['davies_bouldin_score']
            shape = f"{result['reduced_shape'][1]}"
            
            print(f"{method_display:<25} {silhouette:<12.4f} {davies_bouldin:<15.4f} {shape:<15}")
        
        # Calculate improvements
        default_silhouette = results['default']['silhouette_score']
        optimized_silhouette = results['optimized']['silhouette_score']
        best_silhouette = results['best']['silhouette_score']
        
        if default_silhouette > 0:
            optimized_improvement = ((optimized_silhouette - default_silhouette) / default_silhouette) * 100
            best_improvement = ((best_silhouette - default_silhouette) / default_silhouette) * 100
            
            print(f"\nImprovements over default:")
            print(f"  Optimized: {optimized_improvement:+.2f}%")
            print(f"  Best: {best_improvement:+.2f}%")
        
        return results
    
    def run_comprehensive_test(self):
        """Run comprehensive test on multiple datasets."""
        print("=" * 80)
        print("COMPREHENSIVE PBP OPTIMIZATION COMPARISON")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Test datasets
        test_datasets = [
            'iris', 'breast_cancer', 'wine', 'digits', 'diabetes',
            'sonar', 'glass', 'seeds', 'thyroid', 'pima'
        ]
        
        all_results = {}
        
        for dataset_name in test_datasets:
            try:
                results = self.test_dataset_comparison(dataset_name)
                if results:
                    all_results[dataset_name] = results
            except Exception as e:
                print(f"Error testing {dataset_name}: {e}")
        
        # Generate summary
        self.generate_summary_report(all_results)
        
        return all_results
    
    def generate_summary_report(self, all_results):
        """Generate summary report of all results."""
        print(f"\n{'='*80}")
        print("SUMMARY REPORT - PBP OPTIMIZATION COMPARISON")
        print(f"{'='*80}")
        
        if not all_results:
            print("No results available.")
            return
        
        # Create summary table
        summary_data = []
        for dataset_name, results in all_results.items():
            default_silhouette = results['default']['silhouette_score']
            optimized_silhouette = results['optimized']['silhouette_score']
            best_silhouette = results['best']['silhouette_score']
            
            optimized_agg = results['optimized'].get('agg_func_name', 'sum')
            best_agg = results['best'].get('agg_func_name', 'sum')
            
            summary_data.append({
                'Dataset': dataset_name,
                'Default_Silhouette': default_silhouette,
                'Optimized_Silhouette': optimized_silhouette,
                'Best_Silhouette': best_silhouette,
                'Optimized_Agg': optimized_agg,
                'Best_Agg': best_agg,
                'Optimized_Improvement': ((optimized_silhouette - default_silhouette) / default_silhouette * 100) if default_silhouette > 0 else 0,
                'Best_Improvement': ((best_silhouette - default_silhouette) / default_silhouette * 100) if default_silhouette > 0 else 0
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Print summary table
        print("\nSummary Table:")
        print(summary_df.to_string(index=False, float_format='%.4f'))
        
        # Save to file
        output_file = f"{self.results_dir}/tables/pbp_optimization_comparison.csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        summary_df.to_csv(output_file, index=False)
        print(f"\nSummary saved to: {output_file}")
        
        # Statistics
        print(f"\nOverall Statistics:")
        print(f"Average Default Silhouette: {summary_df['Default_Silhouette'].mean():.4f} ± {summary_df['Default_Silhouette'].std():.4f}")
        print(f"Average Optimized Silhouette: {summary_df['Optimized_Silhouette'].mean():.4f} ± {summary_df['Optimized_Silhouette'].std():.4f}")
        print(f"Average Best Silhouette: {summary_df['Best_Silhouette'].mean():.4f} ± {summary_df['Best_Silhouette'].std():.4f}")
        
        print(f"\nAverage Improvements:")
        print(f"Optimized over Default: {summary_df['Optimized_Improvement'].mean():+.2f}%")
        print(f"Best over Default: {summary_df['Best_Improvement'].mean():+.2f}%")
        
        # Wins analysis
        optimized_wins = len(summary_df[summary_df['Optimized_Silhouette'] > summary_df['Default_Silhouette']])
        best_wins = len(summary_df[summary_df['Best_Silhouette'] > summary_df['Default_Silhouette']])
        total_datasets = len(summary_df)
        
        print(f"\nWin Analysis:")
        print(f"Optimized wins: {optimized_wins}/{total_datasets} ({optimized_wins/total_datasets*100:.1f}%)")
        print(f"Best wins: {best_wins}/{total_datasets} ({best_wins/total_datasets*100:.1f}%)")
        
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test optimized PBP performance")
    parser.add_argument("-d", "--dataset", default='all', help="Specific dataset to test")
    args = parser.parse_args()
    
    # Create directories
    data_dir = './data'
    results_dir = './results'
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize comparison
    comparison = OptimizedPBPComparison(data_dir, results_dir)
    
    if args.dataset == 'all':
        results = comparison.run_comprehensive_test()
    else:
        results = comparison.test_dataset_comparison(args.dataset)
    
    print(f"\n✅ Testing completed!")


if __name__ == "__main__":
    main() 