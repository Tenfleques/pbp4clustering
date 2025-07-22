#!/usr/bin/env python3
"""
Comprehensive Comparison: PBP vs PCA, t-SNE, UMAP

This script compares the pseudo-Boolean polynomial approach with feature selection
against traditional dimensionality reduction methods (PCA, t-SNE, UMAP).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import os
import sys
import json
import warnings
import logging
from tqdm import tqdm

# Suppress warnings and verbose output
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

# Suppress UMAP verbose output
os.environ['UMAP_VERBOSE'] = '0'

# Import UMAP after setting environment variables
import umap

# Add colorama for colored terminal output
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False
    # Define dummy color classes if colorama is not available
    class Fore:
        GREEN = ""
        RED = ""
        YELLOW = ""
        BLUE = ""
        MAGENTA = ""
        CYAN = ""
        WHITE = ""
        RESET = ""
    
    class Back:
        GREEN = ""
        RED = ""
        YELLOW = ""
        BLUE = ""
        MAGENTA = ""
        CYAN = ""
        WHITE = ""
        RESET = ""
    
    class Style:
        BRIGHT = ""
        DIM = ""
        NORMAL = ""
        RESET_ALL = ""

# Import PBP functions
try:
    from ..pbp.core import pbp_vector
    from ..data.loader import DatasetTransformer
except ImportError:
    print("Warning: pbp modules not found. Using PCA as fallback.")
    pbp_vector = None


class ComprehensiveComparison:
    """Comprehensive comparison of dimensionality reduction methods."""
    
    def __init__(self, data_dir='./data', results_dir='./results'):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.results = {}
        self.feature_selection_results = {}
        self.transformer = DatasetTransformer()
        
    def load_dataset(self, dataset_name):
        """Load dataset using the centralized DatasetTransformer."""
        print(f"Loading {dataset_name}...")
        
        # Use the DatasetTransformer to load datasets
        dataset_obj = self.transformer.load_dataset(dataset_name)
        
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
    
    def apply_pca(self, X, n_components=3):
        """Apply PCA dimensionality reduction."""
        # Flatten matrices for PCA
        X_flat = X.reshape(X.shape[0], -1)
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_flat)
        
        # Apply PCA
        pca = PCA(n_components=min(n_components, X_scaled.shape[1]))
        X_reduced = pca.fit_transform(X_scaled)
        
        return X_reduced, "PCA"
    
    def apply_tsne(self, X, n_components=3):
        """Apply t-SNE dimensionality reduction."""
        # Flatten matrices for t-SNE
        X_flat = X.reshape(X.shape[0], -1)
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_flat)
        
        # Apply t-SNE
        tsne = TSNE(n_components=n_components, random_state=42, perplexity=min(30, X_scaled.shape[0]-1))
        X_reduced = tsne.fit_transform(X_scaled)
        
        return X_reduced, "t-SNE"
    
    def apply_umap(self, X, n_components=3):
        """Apply UMAP dimensionality reduction."""
        # Flatten matrices for UMAP
        X_flat = X.reshape(X.shape[0], -1)
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_flat)
        
        # Apply UMAP
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        X_reduced = reducer.fit_transform(X_scaled)
        
        return X_reduced, "UMAP"
    
    def apply_pbp_with_feature_selection(self, X, y, max_features_to_drop=0):
        """Apply PBP with feature selection."""
        if pbp_vector is None:
            print("PBP not available, using PCA as fallback")
            return self.apply_pca(X)
        
        print(f"Applying PBP with feature selection (max_drop={max_features_to_drop})")
        
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
                        pbp_result = pbp_vector(X[i])
                        reduced_samples.append(pbp_result)
                    except Exception as e:
                        print(f"Error processing sample {i}: {e}")
                        reduced_samples.append(X[i].flatten())
                
                X_reduced = np.array(reduced_samples)
                
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
                        
                        pbp_result = pbp_vector(modified_matrix)
                        reduced_samples.append(pbp_result)
                    except Exception as e:
                        print(f"Error processing sample {i}: {e}")
                        reduced_samples.append(X[i].flatten())
                
                X_reduced = np.array(reduced_samples)
                
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
        
        return best_reduction, f"PBP (dropped {best_features_dropped} features)"
    
    def evaluate_clustering(self, X_reduced, y_true, method_name):
        """Evaluate clustering performance."""
        if len(np.unique(y_true)) <= 1:
            return {
                'silhouette_score': 0,
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
        metadata = dataset['metadata']
        
        print(f"Dataset shape: {X.shape}")
        print(f"Number of classes: {len(np.unique(y))}")
        
        results = {}
        
        # Apply different methods
        methods = [
            ('PCA', lambda: self.apply_pca(X)),
            ('t-SNE', lambda: self.apply_tsne(X)),
            ('UMAP', lambda: self.apply_umap(X)),
            ('PBP', lambda: self.apply_pbp_with_feature_selection(X, y))
        ]
        
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
        
        # Store results
        self.results[dataset_name] = results
        
        return results
    
    def run_comprehensive_comparison(self):
        """Run comprehensive comparison on all available datasets."""
        print("Running comprehensive comparison on all datasets...")
        
        # List of datasets to test
        datasets = [
            'iris', 'breast_cancer', 'wine', 'digits', 'diabetes',
            'sonar', 'glass', 'vehicle', 'ecoli', 'yeast'
        ]
        
        all_results = {}
        
        for dataset_name in tqdm(datasets, desc="Processing datasets"):
            try:
                results = self.compare_methods(dataset_name)
                if results:
                    all_results[dataset_name] = results
            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")
        
        # Generate summary
        self.generate_comprehensive_summary(all_results)
        
        return all_results
    
    def generate_comprehensive_summary(self, all_results):
        """Generate a comprehensive summary of all results."""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        # Create summary DataFrame
        summary_data = []
        
        for dataset_name, results in all_results.items():
            for method_name, evaluation in results.items():
                if 'error' not in evaluation:
                    summary_data.append({
                        'Dataset': dataset_name,
                        'Method': method_name,
                        'Silhouette_Score': evaluation['silhouette_score'],
                        'Davies_Bouldin_Score': evaluation['davies_bouldin_score'],
                        'N_Clusters': evaluation['n_clusters']
                    })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            
            # Save summary to CSV
            output_file = f"{self.results_dir}/tables/comprehensive_comparison_summary.csv"
            summary_df.to_csv(output_file, index=False)
            print(f"Summary saved to: {output_file}")
            
            # Print summary statistics
            print("\nSummary Statistics:")
            print("-" * 50)
            
            # Best method per dataset
            print("\nBest method per dataset (by Silhouette Score):")
            best_per_dataset = summary_df.loc[summary_df.groupby('Dataset')['Silhouette_Score'].idxmax()]
            for _, row in best_per_dataset.iterrows():
                print(f"  {row['Dataset']}: {row['Method']} (Score: {row['Silhouette_Score']:.4f})")
            
            # Overall best method
            print(f"\nOverall best method: {summary_df.loc[summary_df['Silhouette_Score'].idxmax(), 'Method']}")
            print(f"Best score: {summary_df['Silhouette_Score'].max():.4f}")
            
            # Method rankings
            print("\nMethod rankings (average Silhouette Score):")
            method_rankings = summary_df.groupby('Method')['Silhouette_Score'].agg(['mean', 'std']).sort_values('mean', ascending=False)
            for method, stats in method_rankings.iterrows():
                print(f"  {method}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        else:
            print("No valid results to summarize.")


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