#!/usr/bin/env python3
"""
Real Data Evaluation Pipeline for Pseudo-Boolean Polynomial Analysis

This module evaluates PBP performance on real datasets and compares against
traditional dimensionality reduction methods (PCA, t-SNE, UMAP).
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Import existing PBP implementation
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.pbp.core import pbp_vector

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with: pip install umap-learn")

class RealDataEvaluator:
    """Evaluates PBP and traditional methods on real datasets."""
    
    def __init__(self, data_dir='./data/real', results_dir='./results/real_data_evaluation'):
        self.data_dirs = [Path(data_dir), Path('./data/real_medical')]
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.datasets = {}
        self.results = {}
        
    def load_real_datasets(self):
        """Load all real datasets from the data directories."""
        print("Loading real datasets...")
        
        loaded_datasets = {}
        
        # Search in all data directories
        for data_dir in self.data_dirs:
            if not data_dir.exists():
                continue
                
            # Find all metadata files
            metadata_files = list(data_dir.glob("*_metadata.json"))
            
            for metadata_file in metadata_files:
                dataset_name = metadata_file.stem.replace('_metadata', '')
                
                try:
                    # Load metadata
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Load data arrays
                    X_file = data_dir / f"{dataset_name}_X.npy"
                    y_file = data_dir / f"{dataset_name}_y.npy"
                    
                    if X_file.exists() and y_file.exists():
                        # Load X data (should be numeric)
                        X = np.load(X_file)
                        
                        # Load y data - handle both numeric and object arrays
                        try:
                            y = np.load(y_file)
                        except:
                            # If regular load fails, try with allow_pickle
                            y = np.load(y_file, allow_pickle=True)
                        
                        # Convert string labels to numeric codes if needed
                        if y.dtype == 'object':
                            from sklearn.preprocessing import LabelEncoder
                            le = LabelEncoder()
                            y = le.fit_transform(y)
                        
                        dataset = {
                            'X': X,
                            'y': y,
                            'metadata': metadata
                        }
                        
                        loaded_datasets[dataset_name] = dataset
                        print(f"  ✓ Loaded {dataset_name}: {X.shape}")
                    
                except Exception as e:
                    print(f"  ✗ Failed to load {dataset_name}: {e}")
                    continue
        
        self.datasets = loaded_datasets
        print(f"Successfully loaded {len(loaded_datasets)} real datasets\n")
        return loaded_datasets
    
    def evaluate_pbp_method(self, X, y, dataset_name):
        """Evaluate PBP method on a dataset."""
        try:
            print(f"    Evaluating PBP...")
            
            # Process each matrix sample
            if os.path.exists(os.path.join(self.data_dir, f'{dataset_name}_pbp_features.npy')):
                X_pbp = np.load(os.path.join(self.data_dir, f'{dataset_name}_pbp_features.npy'))
            else:
                pbp_features = []
                for i in range(X.shape[0]):
                    matrix = X[i]
                    # Convert matrix to PBP vector representation
                    pbp_repr = pbp_vector(matrix)
                    pbp_features.append(pbp_repr)
                
                # Convert to numpy array
                X_pbp = np.array(pbp_features)
            
            # Evaluate clustering performance
            if len(np.unique(y)) < X_pbp.shape[0]:  # Only if we have fewer classes than samples
                # Use KMeans with known number of clusters
                n_clusters = len(np.unique(y))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_pbp)
                
                # Calculate metrics
                silhouette = silhouette_score(X_pbp, cluster_labels)
                davies_bouldin = davies_bouldin_score(X_pbp, cluster_labels)
                
                np.save(os.path.join(self.data_dir, f'{dataset_name}_pbp_features.npy'), X_pbp)
                results = {
                    'method': 'PBP',
                    'n_features': X_pbp.shape[1],
                    'silhouette_score': silhouette,
                    'davies_bouldin_score': davies_bouldin,
                    'n_clusters': n_clusters,
                    'reduced_data': X_pbp,
                    'cluster_labels': cluster_labels,
                    'success': True
                }
            else:
                results = {
                    'method': 'PBP',
                    'success': False,
                    'error': 'Too few samples for clustering evaluation'
                }
            
            return results
            
        except Exception as e:
            return {
                'method': 'PBP',
                'success': False,
                'error': str(e)
            }
    
    def evaluate_traditional_method(self, X, y, method_name, n_components=2):
        """Evaluate traditional dimensionality reduction method."""
        try:
            print(f"    Evaluating {method_name}...")
            
            # Flatten matrices to vectors for traditional methods
            X_flat = X.reshape(X.shape[0], -1)
            
            # Apply dimensionality reduction
            if method_name == 'PCA':
                reducer = PCA(n_components=n_components, random_state=42)
                X_reduced = reducer.fit_transform(X_flat)
            elif method_name == 't-SNE':
                reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, X_flat.shape[0]-1))
                X_reduced = reducer.fit_transform(X_flat)
            elif method_name == 'UMAP' and UMAP_AVAILABLE:
                reducer = umap.UMAP(n_components=n_components, random_state=42)
                X_reduced = reducer.fit_transform(X_flat)
            else:
                return {
                    'method': method_name,
                    'success': False,
                    'error': f'{method_name} not available'
                }
            
            # Evaluate clustering performance
            if len(np.unique(y)) < X_reduced.shape[0]:
                n_clusters = len(np.unique(y))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_reduced)
                
                # Calculate metrics
                silhouette = silhouette_score(X_reduced, cluster_labels)
                davies_bouldin = davies_bouldin_score(X_reduced, cluster_labels)
                
                results = {
                    'method': method_name,
                    'n_features': X_reduced.shape[1],
                    'silhouette_score': silhouette,
                    'davies_bouldin_score': davies_bouldin,
                    'n_clusters': n_clusters,
                    'reduced_data': X_reduced,
                    'cluster_labels': cluster_labels,
                    'success': True
                }
            else:
                results = {
                    'method': method_name,
                    'success': False,
                    'error': 'Too few samples for clustering evaluation'
                }
            
            return results
            
        except Exception as e:
            return {
                'method': method_name,
                'success': False,
                'error': str(e)
            }
    
    def evaluate_raw_data(self, X, y):
        """Evaluate clustering performance on raw data without dimensionality reduction."""
        try:
            print(f"    Evaluating Raw Data...")
            
            # Flatten matrices to vectors for raw data evaluation
            X_flat = X.reshape(X.shape[0], -1)
            
            # Evaluate clustering performance on raw data
            if len(np.unique(y)) < X_flat.shape[0]:
                n_clusters = len(np.unique(y))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_flat)
                
                # Calculate metrics
                silhouette = silhouette_score(X_flat, cluster_labels)
                davies_bouldin = davies_bouldin_score(X_flat, cluster_labels)
                
                results = {
                    'method': 'Raw',
                    'n_features': X_flat.shape[1],
                    'silhouette_score': silhouette,
                    'davies_bouldin_score': davies_bouldin,
                    'n_clusters': n_clusters,
                    'reduced_data': X_flat,
                    'cluster_labels': cluster_labels,
                    'success': True
                }
            else:
                results = {
                    'method': 'Raw',
                    'success': False,
                    'error': 'Too few samples for clustering evaluation'
                }
            
            return results
            
        except Exception as e:
            return {
                'method': 'Raw',
                'success': False,
                'error': str(e)
            }
    
    def evaluate_dataset(self, dataset_name, dataset):
        """Evaluate all methods on a single dataset."""
        print(f"\nEvaluating dataset: {dataset_name}")
        print(f"  Shape: {dataset['X'].shape}")
        print(f"  Classes: {len(np.unique(dataset['y']))}")
        print(f"  Data type: {dataset['metadata'].get('data_type', 'unknown')}")
        
        X = dataset['X']
        y = dataset['y']
        
        dataset_results = {}
        
        # Evaluate Raw Data (baseline)
        raw_results = self.evaluate_raw_data(X, y)
        dataset_results['Raw'] = raw_results
        
        # Evaluate PBP
        pbp_results = self.evaluate_pbp_method(X, y, dataset_name)
        dataset_results['PBP'] = pbp_results
        
        # Evaluate traditional methods
        methods = ['PCA', 't-SNE']
        if UMAP_AVAILABLE:
            methods.append('UMAP')
        
        for method in methods:
            method_results = self.evaluate_traditional_method(X, y, method)
            dataset_results[method] = method_results
        
        return dataset_results
    
    def evaluate_all_datasets(self):
        """Evaluate all loaded datasets with all methods."""
        print("="*80)
        print("COMPREHENSIVE REAL DATA EVALUATION")
        print("="*80)
        
        if not self.datasets:
            self.load_real_datasets()
        
        all_results = {}
        
        for dataset_name, dataset in self.datasets.items():
            dataset_results = self.evaluate_dataset(dataset_name, dataset)
            all_results[dataset_name] = dataset_results
        
        self.results = all_results
        return all_results
    
    def generate_performance_summary(self):
        """Generate a comprehensive performance summary."""
        if not self.results:
            print("No results available. Run evaluate_all_datasets() first.")
            return
        
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        
        # Collect successful results
        summary_data = []
        
        for dataset_name, dataset_results in self.results.items():
            dataset_info = self.datasets[dataset_name]['metadata']
            
            for method_name, method_results in dataset_results.items():
                if method_results.get('success', False):
                    summary_data.append({
                        'Dataset': dataset_name,
                        'Data_Type': dataset_info.get('data_type', 'unknown'),
                        'Sample_Count': dataset_info.get('sample_count', 'unknown'),
                        'Method': method_name,
                        'Silhouette_Score': method_results['silhouette_score'],
                        'Davies_Bouldin_Score': method_results['davies_bouldin_score'],
                        'N_Features': method_results['n_features'],
                        'N_Clusters': method_results['n_clusters']
                    })
        
        if not summary_data:
            print("No successful evaluations found.")
            return
        
        # Create DataFrame for analysis
        df = pd.DataFrame(summary_data)
        
        print("\nDETAILED RESULTS:")
        print("-" * 80)
        
        # Import dataset info
        try:
            from src.data.dataset_info import get_dataset_info
        except ImportError:
            get_dataset_info = lambda x: None
        
        for dataset_name in df['Dataset'].unique():
            dataset_df = df[df['Dataset'] == dataset_name]
            dataset_info = self.datasets[dataset_name]['metadata']
            dataset_desc = get_dataset_info(dataset_name)
            
            print(f"\n{dataset_name.upper()}:")
            print(f"  Data Type: {dataset_df['Data_Type'].iloc[0]}")
            print(f"  Sample Count: {dataset_info.get('sample_count', 'unknown')}")
            
            if dataset_desc:
                print(f"  Description: {dataset_desc['description'][:100]}...")
                print(f"  Domain: {dataset_desc.get('domain', 'unknown')}")
            
            for _, row in dataset_df.iterrows():
                print(f"  {row['Method']:>8}: Silhouette={row['Silhouette_Score']:.3f}, "
                      f"Davies-Bouldin={row['Davies_Bouldin_Score']:.3f}")
        
        print("\nAGGREGATE PERFORMANCE:")
        print("-" * 80)
        
        # Method rankings
        method_summary = df.groupby('Method').agg({
            'Silhouette_Score': ['mean', 'std'],
            'Davies_Bouldin_Score': ['mean', 'std']
        }).round(3)
        
        print("\nSILHOUETTE SCORE (higher is better):")
        sil_ranking = df.groupby('Method')['Silhouette_Score'].mean().sort_values(ascending=False)
        for i, (method, score) in enumerate(sil_ranking.items(), 1):
            print(f"  {i}. {method}: {score:.3f}")
        
        print("\nDAVIES-BOULDIN SCORE (lower is better):")
        db_ranking = df.groupby('Method')['Davies_Bouldin_Score'].mean().sort_values(ascending=True)
        for i, (method, score) in enumerate(db_ranking.items(), 1):
            print(f"  {i}. {method}: {score:.3f}")
        
        # PBP performance analysis
        print("\nPBP PERFORMANCE ANALYSIS:")
        print("-" * 80)
        
        pbp_results = df[df['Method'] == 'PBP']
        if not pbp_results.empty:
            # Best performing datasets for PBP
            best_pbp = pbp_results.loc[pbp_results['Silhouette_Score'].idxmax()]
            print(f"Best PBP Performance: {best_pbp['Dataset']} (Silhouette: {best_pbp['Silhouette_Score']:.3f})")
            
            # Data type analysis
            pbp_by_type = pbp_results.groupby('Data_Type')['Silhouette_Score'].mean().sort_values(ascending=False)
            print(f"\nPBP Performance by Data Type:")
            for data_type, avg_score in pbp_by_type.items():
                print(f"  {data_type}: {avg_score:.3f}")
            
            # Win analysis
            wins_count = 0
            total_datasets = len(df['Dataset'].unique())
            
            for dataset_name in df['Dataset'].unique():
                dataset_scores = df[df['Dataset'] == dataset_name]
                if not dataset_scores.empty:
                    pbp_score = dataset_scores[dataset_scores['Method'] == 'PBP']['Silhouette_Score'].values
                    if len(pbp_score) > 0:
                        max_score = dataset_scores['Silhouette_Score'].max()
                        if pbp_score[0] == max_score:
                            wins_count += 1
            
            print(f"\nPBP wins on {wins_count}/{total_datasets} datasets ({wins_count/total_datasets*100:.1f}%)")
        
        # Save summary to file
        summary_file = self.results_dir / 'performance_summary.csv'
        df.to_csv(summary_file, index=False)
        print(f"\nDetailed results saved to: {summary_file}")
        
        return df
    
    def save_results(self):
        """Save all results to files."""
        if not self.results:
            print("No results to save.")
            return
        
        # Save raw results
        results_file = self.results_dir / 'raw_results.json'
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for dataset_name, dataset_results in self.results.items():
            json_results[dataset_name] = {}
            for method_name, method_results in dataset_results.items():
                json_method_results = method_results.copy()
                # Remove numpy arrays that can't be serialized
                json_method_results.pop('reduced_data', None)
                json_method_results.pop('cluster_labels', None)
                json_results[dataset_name][method_name] = json_method_results
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Raw results saved to: {results_file}")


def main():
    """Main function to run comprehensive evaluation."""
    print("=== Real Data Evaluation Pipeline ===\n")
    
    # Initialize evaluator
    evaluator = RealDataEvaluator()
    
    # Load datasets
    datasets = evaluator.load_real_datasets()
    
    if not datasets:
        print("No datasets found. Please run the real data pipeline first.")
        return
    
    # Run comprehensive evaluation
    results = evaluator.evaluate_all_datasets()
    
    # Generate performance summary
    summary_df = evaluator.generate_performance_summary()
    
    # Save results
    evaluator.save_results()
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Evaluated {len(datasets)} real datasets")
    print(f"Results saved in: {evaluator.results_dir}")


if __name__ == "__main__":
    main() 