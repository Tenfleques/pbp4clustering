#!/usr/bin/env python3
"""
Aggregation Function Optimization for PBP Clustering

This module provides functionality to test different aggregation functions
and find the optimal one for each dataset based on clustering performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Callable
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from ..pbp.core import pbp_vector
from ..pbp.aggregation_functions import (
    get_aggregation_function, 
    get_recommended_aggregation_functions,
    get_all_aggregation_functions
)


class AggregationOptimizer:
    """
    Optimizes aggregation functions for PBP clustering.
    
    This class tests different aggregation functions on datasets and
    evaluates their impact on clustering performance.
    """
    
    def __init__(self, n_clusters: int = None, random_state: int = 42):
        """
        Initialize the aggregation optimizer.
        
        Args:
            n_clusters: Number of clusters for evaluation (auto-determined if None)
            random_state: Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.results = {}
        self.best_functions = {}
        
    def evaluate_clustering(self, X: np.ndarray, y_true: np.ndarray = None) -> Dict[str, float]:
        """
        Evaluate clustering performance using multiple metrics.
        
        Args:
            X: Feature matrix
            y_true: True labels (optional, for supervised metrics)
            
        Returns:
            Dict: Dictionary of clustering metrics
        """
        if X.shape[0] < 2:
            return {
                'silhouette': 0.0,
                'calinski_harabasz': 0.0,
                'davies_bouldin': float('inf'),
                'inertia': 0.0
            }
        
        # Determine number of clusters
        n_clusters = self.n_clusters
        if n_clusters is None:
            n_clusters = len(np.unique(y_true)) if y_true is not None else min(3, X.shape[0] // 10)
            n_clusters = max(2, min(n_clusters, X.shape[0] // 2))
        
        try:
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            y_pred = kmeans.fit_predict(X)
            
            # Calculate metrics
            metrics = {}
            
            # Silhouette score (higher is better)
            if len(np.unique(y_pred)) > 1:
                metrics['silhouette'] = silhouette_score(X, y_pred)
            else:
                metrics['silhouette'] = 0.0
            
            # Calinski-Harabasz score (higher is better)
            if len(np.unique(y_pred)) > 1:
                metrics['calinski_harabasz'] = calinski_harabasz_score(X, y_pred)
            else:
                metrics['calinski_harabasz'] = 0.0
            
            # Davies-Bouldin score (lower is better)
            if len(np.unique(y_pred)) > 1:
                metrics['davies_bouldin'] = davies_bouldin_score(X, y_pred)
            else:
                metrics['davies_bouldin'] = float('inf')
            
            # Inertia (lower is better)
            metrics['inertia'] = kmeans.inertia_
            
            return metrics
            
        except Exception as e:
            print(f"Error in clustering evaluation: {e}")
            return {
                'silhouette': 0.0,
                'calinski_harabasz': 0.0,
                'davies_bouldin': float('inf'),
                'inertia': float('inf')
            }
    
    def compute_pbp_features(self, dataset: Dict[str, Any], agg_func: Callable) -> np.ndarray:
        """
        Compute PBP features using a specific aggregation function.
        
        Args:
            dataset: Dataset dictionary with 'X' key containing matrix data
            agg_func: Aggregation function to use
            
        Returns:
            np.ndarray: PBP feature vectors
        """
        X_matrices = dataset['X']
        pbp_features = []
        
        for i in range(X_matrices.shape[0]):
            try:
                # Compute PBP vector for each sample
                pbp_vec = pbp_vector(X_matrices[i], agg_func)
                pbp_features.append(pbp_vec)
            except Exception as e:
                print(f"Error computing PBP for sample {i}: {e}")
                # Use zeros as fallback
                pbp_features.append(np.zeros(2**X_matrices.shape[1] - 1))
        
        return np.array(pbp_features)
    
    def test_aggregation_function(self, dataset: Dict[str, Any], 
                                agg_func_name: str, 
                                agg_func: Callable) -> Dict[str, Any]:
        """
        Test a single aggregation function on a dataset.
        
        Args:
            dataset: Dataset dictionary
            agg_func_name: Name of the aggregation function
            agg_func: Aggregation function to test
            
        Returns:
            Dict: Results for this aggregation function
        """
        print(f"  Testing {agg_func_name}...")
        
        try:
            # Compute PBP features
            pbp_features = self.compute_pbp_features(dataset, agg_func)
            
            # Standardize features
            scaler = StandardScaler()
            pbp_features_scaled = scaler.fit_transform(pbp_features)
            
            # Evaluate clustering
            y_true = dataset.get('y', None)
            metrics = self.evaluate_clustering(pbp_features_scaled, y_true)
            
            # Add metadata
            result = {
                'agg_func_name': agg_func_name,
                'metrics': metrics,
                'feature_shape': pbp_features.shape,
                'success': True,
                'error': None
            }
            
            print(f"    ✓ Silhouette: {metrics['silhouette']:.4f}, "
                  f"Calinski-Harabasz: {metrics['calinski_harabasz']:.2f}, "
                  f"Davies-Bouldin: {metrics['davies_bouldin']:.4f}")
            
            return result
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            return {
                'agg_func_name': agg_func_name,
                'metrics': {
                    'silhouette': 0.0,
                    'calinski_harabasz': 0.0,
                    'davies_bouldin': float('inf'),
                    'inertia': float('inf')
                },
                'feature_shape': None,
                'success': False,
                'error': str(e)
            }
    
    def optimize_dataset(self, dataset: Dict[str, Any], 
                        dataset_name: str,
                        agg_functions: List[str] = None) -> Dict[str, Any]:
        """
        Optimize aggregation functions for a specific dataset.
        
        Args:
            dataset: Dataset dictionary
            dataset_name: Name of the dataset
            agg_functions: List of aggregation function names to test
            
        Returns:
            Dict: Optimization results
        """
        if agg_functions is None:
            agg_functions = get_recommended_aggregation_functions()
        
        print(f"\nOptimizing aggregation functions for {dataset_name}")
        print(f"Dataset shape: {dataset['X'].shape}")
        print(f"Testing {len(agg_functions)} aggregation functions...")
        
        results = []
        
        for agg_func_name in agg_functions:
            try:
                agg_func = get_aggregation_function(agg_func_name)
                result = self.test_aggregation_function(dataset, agg_func_name, agg_func)
                results.append(result)
            except Exception as e:
                print(f"  ✗ Failed to get aggregation function {agg_func_name}: {e}")
                continue
        
        # Find best aggregation function
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            print(f"  ✗ No successful aggregation functions for {dataset_name}")
            return {
                'dataset_name': dataset_name,
                'best_function': None,
                'best_metrics': None,
                'all_results': results,
                'success': False
            }
        
        # Score aggregation functions (higher is better)
        for result in successful_results:
            metrics = result['metrics']
            # Combined score: silhouette + calinski_harabasz/1000 - davies_bouldin/10
            result['combined_score'] = (
                metrics['silhouette'] + 
                metrics['calinski_harabasz'] / 1000 - 
                metrics['davies_bouldin'] / 10
            )
        
        # Find best function
        best_result = max(successful_results, key=lambda x: x['combined_score'])
        
        print(f"\n  Best aggregation function: {best_result['agg_func_name']}")
        print(f"  Best metrics:")
        print(f"    Silhouette: {best_result['metrics']['silhouette']:.4f}")
        print(f"    Calinski-Harabasz: {best_result['metrics']['calinski_harabasz']:.2f}")
        print(f"    Davies-Bouldin: {best_result['metrics']['davies_bouldin']:.4f}")
        print(f"    Combined Score: {best_result['combined_score']:.4f}")
        
        return {
            'dataset_name': dataset_name,
            'best_function': best_result['agg_func_name'],
            'best_metrics': best_result['metrics'],
            'best_combined_score': best_result['combined_score'],
            'all_results': results,
            'success': True
        }
    
    def optimize_multiple_datasets(self, datasets: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Optimize aggregation functions for multiple datasets.
        
        Args:
            datasets: Dictionary mapping dataset names to dataset dictionaries
            
        Returns:
            Dict: Overall optimization results
        """
        print("=" * 80)
        print("AGGREGATION FUNCTION OPTIMIZATION")
        print("=" * 80)
        
        all_results = {}
        summary = {
            'total_datasets': len(datasets),
            'successful_datasets': 0,
            'best_functions': {},
            'performance_summary': {}
        }
        
        for dataset_name, dataset in datasets.items():
            try:
                result = self.optimize_dataset(dataset, dataset_name)
                all_results[dataset_name] = result
                
                if result['success']:
                    summary['successful_datasets'] += 1
                    summary['best_functions'][dataset_name] = result['best_function']
                    
                    # Track performance
                    metrics = result['best_metrics']
                    for metric_name, value in metrics.items():
                        if metric_name not in summary['performance_summary']:
                            summary['performance_summary'][metric_name] = []
                        summary['performance_summary'][metric_name].append(value)
                
            except Exception as e:
                print(f"✗ Error optimizing {dataset_name}: {e}")
                all_results[dataset_name] = {
                    'dataset_name': dataset_name,
                    'success': False,
                    'error': str(e)
                }
        
        # Calculate summary statistics
        for metric_name, values in summary['performance_summary'].items():
            if values:
                summary['performance_summary'][metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # Print summary
        print("\n" + "=" * 80)
        print("OPTIMIZATION SUMMARY")
        print("=" * 80)
        print(f"Total datasets: {summary['total_datasets']}")
        print(f"Successful optimizations: {summary['successful_datasets']}")
        print(f"Success rate: {summary['successful_datasets']/summary['total_datasets']*100:.1f}%")
        
        print("\nBest aggregation functions by dataset:")
        for dataset_name, best_func in summary['best_functions'].items():
            print(f"  {dataset_name:20s}: {best_func}")
        
        print("\nPerformance summary:")
        for metric_name, stats in summary['performance_summary'].items():
            print(f"  {metric_name:20s}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        
        return {
            'all_results': all_results,
            'summary': summary
        }
    
    def get_optimal_aggregation_functions(self) -> Dict[str, str]:
        """
        Get the optimal aggregation function for each dataset.
        
        Returns:
            Dict: Mapping from dataset names to optimal aggregation function names
        """
        return self.best_functions.copy()


def create_aggregation_comparison_report(results: Dict[str, Any], 
                                       output_file: str = None) -> str:
    """
    Create a detailed comparison report of aggregation function performance.
    
    Args:
        results: Results from aggregation optimization
        output_file: Optional file to save the report
        
    Returns:
        str: Report content
    """
    report_lines = []
    report_lines.append("# Aggregation Function Optimization Report")
    report_lines.append("")
    
    # Summary statistics
    summary = results['summary']
    report_lines.append("## Summary")
    report_lines.append(f"- Total datasets: {summary['total_datasets']}")
    report_lines.append(f"- Successful optimizations: {summary['successful_datasets']}")
    report_lines.append(f"- Success rate: {summary['successful_datasets']/summary['total_datasets']*100:.1f}%")
    report_lines.append("")
    
    # Best functions by dataset
    report_lines.append("## Optimal Aggregation Functions by Dataset")
    report_lines.append("")
    report_lines.append("| Dataset | Best Function | Silhouette | Calinski-Harabasz | Davies-Bouldin |")
    report_lines.append("|---------|---------------|------------|-------------------|----------------|")
    
    for dataset_name, dataset_result in results['all_results'].items():
        if dataset_result['success']:
            metrics = dataset_result['best_metrics']
            report_lines.append(
                f"| {dataset_name} | {dataset_result['best_function']} | "
                f"{metrics['silhouette']:.4f} | {metrics['calinski_harabasz']:.2f} | "
                f"{metrics['davies_bouldin']:.4f} |"
            )
        else:
            report_lines.append(f"| {dataset_name} | FAILED | - | - | - |")
    
    report_lines.append("")
    
    # Performance summary
    report_lines.append("## Performance Summary")
    report_lines.append("")
    for metric_name, stats in summary['performance_summary'].items():
        report_lines.append(f"### {metric_name.replace('_', ' ').title()}")
        report_lines.append(f"- Mean: {stats['mean']:.4f}")
        report_lines.append(f"- Std: {stats['std']:.4f}")
        report_lines.append(f"- Min: {stats['min']:.4f}")
        report_lines.append(f"- Max: {stats['max']:.4f}")
        report_lines.append("")
    
    # Detailed results
    report_lines.append("## Detailed Results by Dataset")
    report_lines.append("")
    
    for dataset_name, dataset_result in results['all_results'].items():
        report_lines.append(f"### {dataset_name}")
        if dataset_result['success']:
            report_lines.append(f"**Best function:** {dataset_result['best_function']}")
            report_lines.append(f"**Best combined score:** {dataset_result['best_combined_score']:.4f}")
            report_lines.append("")
            report_lines.append("All function results:")
            report_lines.append("")
            
            # Sort by combined score
            all_results = dataset_result['all_results']
            successful_results = [r for r in all_results if r['success']]
            successful_results.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
            
            for result in successful_results:
                metrics = result['metrics']
                report_lines.append(
                    f"- {result['agg_func_name']}: "
                    f"Silhouette={metrics['silhouette']:.4f}, "
                    f"Calinski-Harabasz={metrics['calinski_harabasz']:.2f}, "
                    f"Davies-Bouldin={metrics['davies_bouldin']:.4f}"
                )
        else:
            report_lines.append("**Status:** Failed")
            if 'error' in dataset_result:
                report_lines.append(f"**Error:** {dataset_result['error']}")
        report_lines.append("")
    
    report_content = "\n".join(report_lines)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_content)
        print(f"Report saved to {output_file}")
    
    return report_content


if __name__ == "__main__":
    # Test the optimizer
    print("Testing AggregationOptimizer...")
    
    # Create test dataset
    np.random.seed(42)
    test_data = {
        'X': np.random.randn(100, 2, 3),  # 100 samples, 2x3 matrices
        'y': np.random.choice([0, 1], size=100)
    }
    
    optimizer = AggregationOptimizer(n_clusters=2)
    result = optimizer.optimize_dataset(test_data, "test_dataset")
    
    print(f"\nTest completed. Best function: {result['best_function']}") 