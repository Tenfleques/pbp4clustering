#!/usr/bin/env python3
"""
Comprehensive Comparison Script

This script compares the pseudo-Boolean polynomial approach with feature selection
against traditional dimensionality reduction methods (PCA, t-SNE, UMAP).
Now includes aggregation function optimization for PBP.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import os
import sys
import json
import logging
from tqdm import tqdm
from ..data.dataset_config import get_all_datasets
logging.getLogger('matplotlib.font_manager').disabled = True

# Import PBP modules
try:
    from ..pbp.core import pbp_vector
    PBP_AVAILABLE = True
except ImportError:
    print("Warning: pbp modules not found. Using PCA as fallback.")
    pbp_vector = None
    PBP_AVAILABLE = False

# Import aggregation optimization
try:
    from .aggregation_optimization import AggregationOptimizer
    from ..pbp.aggregation_functions import get_aggregation_function, get_recommended_aggregation_functions
    AGGREGATION_OPTIMIZATION_AVAILABLE = True
except ImportError:
    print("Warning: Aggregation optimization not available. Using default sum aggregation.")
    AGGREGATION_OPTIMIZATION_AVAILABLE = False


class ComprehensiveComparison:
    """Comprehensive comparison of dimensionality reduction methods."""
    
    def __init__(self, data_dir='./data', results_dir='./results', use_optimized_aggregation=True):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.results = {}
        self.feature_selection_results = {}
        self.use_optimized_aggregation = use_optimized_aggregation and AGGREGATION_OPTIMIZATION_AVAILABLE
        self.aggregation_optimizer = None
        self.optimal_aggregation_functions = {}
        
        if self.use_optimized_aggregation:
            print("✓ Aggregation function optimization enabled for PBP")
            self.aggregation_optimizer = AggregationOptimizer(random_state=42)
        else:
            print("⚠ Using default sum aggregation for PBP")
        
    def load_dataset(self, dataset_name):
        """Load dataset using the centralized ConsolidatedDatasetLoader."""
        print(f"Loading {dataset_name}...")
        
        # Use the ConsolidatedDatasetLoader to load datasets
        try:
            from ..data.consolidated_loader import ConsolidatedDatasetLoader
            loader = ConsolidatedDatasetLoader()
            dataset_obj = loader.load_dataset(dataset_name)
            
            if dataset_obj is None:
                print(f"Failed to load dataset: {dataset_name}")
                return None
                
            return {
                'X': dataset_obj['X'],
                'y': dataset_obj['y'],
                'metadata': {
                    'feature_names': dataset_obj.get('feature_names', []),
                    'measurement_names': dataset_obj.get('measurement_names', []),
                    'description': dataset_obj.get('description', f'{dataset_name} dataset')
                }
            }
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            return None
    
    def get_optimal_aggregation_function(self, dataset_name, dataset):
        """Get the optimal aggregation function for a dataset."""
        if not self.use_optimized_aggregation or not PBP_AVAILABLE:
            return lambda x: x.sum()
        
        # Check if we already have the optimal function cached
        if dataset_name in self.optimal_aggregation_functions:
            agg_func_name = self.optimal_aggregation_functions[dataset_name]
            return get_aggregation_function(agg_func_name)
        
        # Check if we have cached optimization results
        cache_file = os.path.join(self.data_dir, f'{dataset_name}_optimal_aggregation.json')
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    if 'best_function' in cache_data:
                        agg_func_name = cache_data['best_function']
                        self.optimal_aggregation_functions[dataset_name] = agg_func_name
                        print(f"  Using cached optimal aggregation: {agg_func_name}")
                        return get_aggregation_function(agg_func_name)
            except Exception as e:
                print(f"  Error loading cached aggregation function: {e}")
        
        # Run optimization if not cached
        print(f"  Optimizing aggregation function for {dataset_name}...")
        try:
            result = self.aggregation_optimizer.optimize_dataset(dataset, dataset_name)
            
            if result and result['success']:
                agg_func_name = result['best_function']
                self.optimal_aggregation_functions[dataset_name] = agg_func_name
                
                # Cache the result
                try:
                    cache_data = {
                        'best_function': agg_func_name,
                        'best_metrics': result['best_metrics'],
                        'best_combined_score': result['best_combined_score']
                    }
                    with open(cache_file, 'w') as f:
                        json.dump(cache_data, f, indent=2)
                except Exception as e:
                    print(f"  Warning: Could not cache aggregation result: {e}")
                
                print(f"  Optimal aggregation function: {agg_func_name}")
                return get_aggregation_function(agg_func_name)
            else:
                print(f"  Optimization failed, using default sum aggregation")
                return lambda x: x.sum()
                
        except Exception as e:
            print(f"  Error in aggregation optimization: {e}")
            return lambda x: x.sum()
    
    def apply_pca(self, X, n_components=3):
        """Apply PCA dimensionality reduction."""
        # Flatten matrices for PCA
        X_flat = X.reshape(X.shape[0], -1)
        
        # Apply PCA
        pca = PCA(n_components=min(n_components, X_flat.shape[1]))
        X_reduced = pca.fit_transform(X_flat)
        
        return X_reduced, "PCA"
    
    def apply_tsne(self, X, n_components=3):
        """Apply t-SNE dimensionality reduction."""
        # Flatten matrices for t-SNE
        X_flat = X.reshape(X.shape[0], -1)
        
        # Apply t-SNE
        tsne = TSNE(n_components=min(n_components, X_flat.shape[1]), random_state=42)
        X_reduced = tsne.fit_transform(X_flat)
        
        return X_reduced, "t-SNE"
    
    def apply_umap(self, X, n_components=3):
        """Apply UMAP dimensionality reduction."""
        # Flatten matrices for UMAP
        X_flat = X.reshape(X.shape[0], -1)
        
        # Apply UMAP
        reducer = umap.UMAP(n_components=min(n_components, X_flat.shape[1]), random_state=42)
        X_reduced = reducer.fit_transform(X_flat)
        
        return X_reduced, "UMAP"
    
    def apply_pbp_with_feature_selection(self, X, y, dataset_name, dataset, max_features_to_drop=0):
        """Apply PBP with feature selection and optimal aggregation function."""
        if pbp_vector is None:
            print("PBP not available, using PCA as fallback")
            return self.apply_pca(X)
        
        # Get optimal aggregation function
        agg_func = self.get_optimal_aggregation_function(dataset_name, dataset)
        agg_func_name = self.optimal_aggregation_functions.get(dataset_name, 'sum')
        
        print(f"Applying PBP with {agg_func_name} aggregation (max_drop={max_features_to_drop})")
        
        # Get original matrix dimensions
        n_rows, n_cols = X.shape[1], X.shape[2]
        total_features = n_rows * n_cols
        
        best_score = -1
        best_reduction = None
        best_features_dropped = 0
        
        # Try dropping different numbers of features
        for features_to_drop in range(max_features_to_drop + 1):
            if features_to_drop == 0:
                # No feature selection
                reduced_samples = []
                for i in range(X.shape[0]):
                    try:
                        pbp_result = pbp_vector(X[i], agg_func)
                        reduced_samples.append(pbp_result)
                    except Exception as e:
                        print(f"Error processing sample {i}: {e}")
                        reduced_samples.append(X[i].flatten())
                
                X_reduced = np.array(reduced_samples)
                
                # Remove zero columns
                zero_columns = np.all(X_reduced == 0, axis=0)
                X_reduced = X_reduced[:, ~zero_columns]
                
                # Evaluate clustering
                if len(np.unique(y)) > 1:
                    kmeans = KMeans(n_clusters=len(np.unique(y)), random_state=42, n_init=10)
                    y_pred = kmeans.fit_predict(X_reduced)
                    score = silhouette_score(X_reduced, y_pred)
                else:
                    score = 0
                
                if score > best_score:
                    best_score = score
                    best_reduction = X_reduced
                    best_features_dropped = 0
            else:
                # Try different combinations of features to drop
                # This is a simplified approach - in practice, you might want more sophisticated feature selection
                print(f"  Trying to drop {features_to_drop} features...")
                
                # For simplicity, we'll just use the first few features
                # In a real implementation, you'd want to try different combinations
                reduced_samples = []
                for i in range(X.shape[0]):
                    try:
                        # Create a modified matrix with some features zeroed out
                        modified_matrix = X[i].copy()
                        # Zero out the first few features
                        for j in range(min(features_to_drop, modified_matrix.size)):
                            flat_idx = j
                            row_idx = flat_idx // modified_matrix.shape[1]
                            col_idx = flat_idx % modified_matrix.shape[1]
                            if row_idx < modified_matrix.shape[0]:
                                modified_matrix[row_idx, col_idx] = 0
                        
                        pbp_result = pbp_vector(modified_matrix, agg_func)
                        reduced_samples.append(pbp_result)
                    except Exception as e:
                        print(f"Error processing sample {i}: {e}")
                        reduced_samples.append(X[i].flatten())
                
                X_reduced = np.array(reduced_samples)
                
                # Remove zero columns
                zero_columns = np.all(X_reduced == 0, axis=0)
                X_reduced = X_reduced[:, ~zero_columns]
                
                # Evaluate clustering
                if len(np.unique(y)) > 1:
                    kmeans = KMeans(n_clusters=len(np.unique(y)), random_state=42, n_init=10)
                    y_pred = kmeans.fit_predict(X_reduced)
                    score = silhouette_score(X_reduced, y_pred)
                else:
                    score = 0
                
                if score > best_score:
                    best_score = score
                    best_reduction = X_reduced
                    best_features_dropped = features_to_drop
        
        return best_reduction, f"PBP ({agg_func_name}, dropped {best_features_dropped} features)"
    
    def evaluate_clustering(self, X_reduced, y_true, method_name):
        """Evaluate clustering performance."""
        if len(np.unique(y_true)) <= 1:
            return {
                'silhouette_score': 0.0,
                'davies_bouldin_score': float('inf'),
                'method': method_name,
                'n_clusters': 1
            }
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=len(np.unique(y_true)), random_state=42, n_init=10)
        y_pred = kmeans.fit_predict(X_reduced)
        
        # Calculate metrics
        silhouette = silhouette_score(X_reduced, y_pred)
        davies_bouldin = davies_bouldin_score(X_reduced, y_pred)
        
        return {
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'method': method_name,
            'n_clusters': len(np.unique(y_true))
        }
    
    def compare_methods(self, dataset_name):
        """Compare different dimensionality reduction methods on a dataset."""
        print(f"\n{'='*60}")
        print(f"Comparing methods on {dataset_name} dataset")
        print(f"{'='*60}")
        
        # Load dataset
        dataset = self.load_dataset(dataset_name)
        if dataset is None:
            return None
        
        X = dataset['X']
        y = dataset['y']
        
        print(f"Dataset shape: {X.shape}")
        print(f"Number of classes: {len(np.unique(y))}")
        
        # Apply different methods
        methods = [
            ('PCA', lambda: self.apply_pca(X)),
            ('t-SNE', lambda: self.apply_tsne(X)),
            ('UMAP', lambda: self.apply_umap(X)),
            ('PBP', lambda: self.apply_pbp_with_feature_selection(X, y, dataset_name, dataset))
        ]
        
        results = {}
        for method_name, method_func in methods:
            try:
                print(f"\nApplying {method_name}...")
                X_reduced, method_label = method_func()
                
                # Evaluate clustering
                evaluation = self.evaluate_clustering(X_reduced, y, method_label)
                results[method_name] = evaluation
                
                print(f"  Silhouette Score: {evaluation['silhouette_score']:.4f}")
                print(f"  Davies-Bouldin Score: {evaluation['davies_bouldin_score']:.4f}")
                
            except Exception as e:
                print(f"  Error applying {method_name}: {e}")
                results[method_name] = {
                    'silhouette_score': 0,
                    'davies_bouldin_score': float('inf'),
                    'method': method_name,
                    'error': str(e)
                }
        
        return results
    
    def run_comprehensive_comparison(self):
        """Run comprehensive comparison across all datasets."""
        # Get all available datasets from the consolidated loader
        try:
            from ..data.consolidated_loader import ConsolidatedDatasetLoader
            loader = ConsolidatedDatasetLoader()
            dataset_config = loader.get_available_datasets()
            
            # Flatten all datasets into a single list
            all_datasets = []
            for category, datasets in dataset_config.items():
                all_datasets.extend(datasets)
            
            print(f"Found {len(all_datasets)} datasets across {len(dataset_config)} categories")
            print("Running comprehensive comparison...")
            
        except Exception as e:
            print(f"Error getting dataset configuration: {e}")
            # Fallback to a subset of datasets
            all_datasets = get_all_datasets()
            print(f"Using fallback dataset list: {len(all_datasets)} datasets")
        
        all_results = {}
        
        for dataset_name in tqdm(all_datasets, desc="Comparing datasets"):
            try:
                results = self.compare_methods(dataset_name)
                if results:
                    all_results[dataset_name] = results
            except Exception as e:
                print(f"Error comparing {dataset_name}: {e}")
        
        # Generate comprehensive summary
        self.generate_comprehensive_summary(all_results)
        
        return all_results
    
    def generate_comprehensive_summary(self, all_results):
        """Generate a comprehensive summary of all comparison results."""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        if not all_results:
            print("No results available for summary.")
            return
        
        # Create summary DataFrame
        summary_data = []
        for dataset_name, results in all_results.items():
            for method_name, result in results.items():
                summary_data.append({
                    'Dataset': dataset_name,
                    'Method': result['method'],
                    'Silhouette_Score': result['silhouette_score'],
                    'Davies_Bouldin_Score': result['davies_bouldin_score'],
                    'N_Clusters': result.get('n_clusters', 0)
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary to CSV
        output_file = f"{self.results_dir}/tables/comprehensive_comparison_summary_optimized.csv"
        summary_df.to_csv(output_file, index=False)
        print(f"Summary saved to: {output_file}")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print("-" * 50)
        
        # Method rankings
        method_rankings = summary_df.groupby('Method')['Silhouette_Score'].agg(['mean', 'std']).sort_values('mean', ascending=False)
        print("\nMethod Rankings (by average Silhouette Score):")
        for method, stats in method_rankings.iterrows():
            print(f"  {method}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        # PBP performance analysis
        if self.use_optimized_aggregation:
            pbp_results = summary_df[summary_df['Method'].str.contains('PBP')]
            if not pbp_results.empty:
                print(f"\nPBP Performance Analysis:")
                print(f"  Average PBP Silhouette Score: {pbp_results['Silhouette_Score'].mean():.4f} ± {pbp_results['Silhouette_Score'].std():.4f}")
                print(f"  Best PBP Score: {pbp_results['Silhouette_Score'].max():.4f}")
                print(f"  PBP wins: {len(pbp_results[pbp_results['Silhouette_Score'] == pbp_results.groupby('Dataset')['Silhouette_Score'].transform('max')])} out of {len(pbp_results)} comparisons")
        
        else:
            print("\nNo aggregation optimization data available.")
        
        # Best performing dataset for each method
        print(f"\nBest performing dataset for each method:")
        for method in summary_df['Method'].unique():
            method_data = summary_df[summary_df['Method'] == method]
            best_dataset = method_data.loc[method_data['Silhouette_Score'].idxmax()]
            print(f"  {method}: {best_dataset['Dataset']} ({best_dataset['Silhouette_Score']:.4f})")


def main():
    """Main function to run comprehensive comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive comparison of dimensionality reduction methods")
    parser.add_argument("--no-optimization", action="store_true", help="Disable aggregation function optimization")
    args = parser.parse_args()
    
    use_optimization = not args.no_optimization
    comparison = ComprehensiveComparison('./data', use_optimized_aggregation=use_optimization)
    results = comparison.run_comprehensive_comparison()
    
    if results:
        print(f"\n✅ Comprehensive comparison completed successfully!")
        print(f"Results for {len(results)} datasets generated.")
    else:
        print(f"\n❌ Comprehensive comparison failed or no results generated.")


if __name__ == "__main__":
    main() 