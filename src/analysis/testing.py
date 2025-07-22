#!/usr/bin/env python3
"""
Test Script for Pseudo-Boolean Polynomial Dimensionality Reduction

This script tests all datasets with the pbp_vector approach, including visualization 
and clustering analysis for all 10 datasets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import os
import sys
import json
import logging
logging.getLogger('matplotlib.font_manager').disabled = True

# Import PBP modules
try:
    from ..pbp.core import pbp_vector
    PBP_AVAILABLE = True
except ImportError:
    print("Warning: pbp modules not found. Install required dependencies.")
    pbp_vector = None
    PBP_AVAILABLE = False

# Import the centralized DatasetTransformer
from ..data.loader import DatasetTransformer

class DatasetTester:
    """Test datasets with pbp_vector approach."""
    
    def __init__(self, data_dir='./data', results_dir='./results'):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.results = {}
        self.transformer = DatasetTransformer()
        
    def load_dataset(self, dataset_name):
        """Load dataset using the centralized DatasetTransformer."""
        print(f"Loading {dataset_name} using centralized loader...")
        
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
    
    def apply_pbp_reduction(self, X):
        """Apply pbp_vector reduction to dataset."""
        if not PBP_AVAILABLE:
            print("PBP module not available. Using PCA as fallback.")
            X_reduced, method_label = self.apply_pca_reduction(X)
            return X_reduced, method_label
        
        print("Applying actual PBP vector reduction...")
        reduced_samples = []
        
        for i in range(X.shape[0]):
            try:
                pbp_result = pbp_vector(X[i])
                reduced_samples.append(pbp_result)
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                # Use original sample if reduction fails
                reduced_samples.append(X[i].flatten())
        
        return np.array(reduced_samples), "PBP"
    
    def apply_pca_reduction(self, X, n_components=3):
        """Apply PCA reduction as fallback."""
        # Flatten matrices for PCA
        X_flat = X.reshape(X.shape[0], -1)
        
        # Apply PCA
        pca = PCA(n_components=min(n_components, X_flat.shape[1]))
        X_reduced = pca.fit_transform(X_flat)
        
        return X_reduced, "PCA"
    
    def evaluate_clustering(self, X_reduced, y_true, n_clusters=None):
        """Evaluate clustering performance."""
        if n_clusters is None:
            n_clusters = len(np.unique(y_true))
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        y_pred = kmeans.fit_predict(X_reduced)
        
        # Calculate metrics
        silhouette = silhouette_score(X_reduced, y_pred)
        davies_bouldin = davies_bouldin_score(X_reduced, y_pred)
        
        return {
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'y_pred': y_pred,
            'n_clusters': n_clusters
        }
    
    def visualize_results(self, X_reduced, y_true, y_pred, dataset_name, metadata, method_label="PBP"):
        """Visualize clustering results."""
        # Ensure y_true and y_pred are numeric and the same length as X_reduced
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Convert string labels to numeric
        if y_true.dtype == object or y_true.dtype.type is np.str_:
            y_true = pd.Categorical(y_true).codes
        if y_pred.dtype == object or y_pred.dtype.type is np.str_:
            y_pred = pd.Categorical(y_pred).codes
            
        # Ensure arrays match the reduced data size
        if len(y_true) != X_reduced.shape[0]:
            print(f"Warning: y_true length ({len(y_true)}) doesn't match X_reduced ({X_reduced.shape[0]}). Skipping visualization.")
            return
        if len(y_pred) != X_reduced.shape[0]:
            print(f"Warning: y_pred length ({len(y_pred)}) doesn't match X_reduced ({X_reduced.shape[0]}). Skipping visualization.")
            return
        
        # Create results directory if it doesn't exist
        os.makedirs(f"{self.results_dir}/figures", exist_ok=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: True labels
        scatter1 = axes[0].scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_true, cmap='viridis', alpha=0.7)
        axes[0].set_title(f'{dataset_name} - True Labels ({method_label})')
        axes[0].set_xlabel('Component 1')
        axes[0].set_ylabel('Component 2')
        plt.colorbar(scatter1, ax=axes[0])
        
        # Plot 2: Predicted labels
        scatter2 = axes[1].scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
        axes[1].set_title(f'{dataset_name} - Predicted Labels ({method_label})')
        axes[1].set_xlabel('Component 1')
        axes[1].set_ylabel('Component 2')
        plt.colorbar(scatter2, ax=axes[1])
        
        # Add 3D plot if we have 3 components
        if X_reduced.shape[1] >= 3:
            fig_3d = plt.figure(figsize=(10, 8))
            ax_3d = fig_3d.add_subplot(111, projection='3d')
            scatter_3d = ax_3d.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], 
                                      c=y_pred, cmap='viridis', alpha=0.7)
            ax_3d.set_title(f'{dataset_name} - 3D Clustering ({method_label})')
            ax_3d.set_xlabel('Component 1')
            ax_3d.set_ylabel('Component 2')
            ax_3d.set_zlabel('Component 3')
            plt.colorbar(scatter_3d, ax=ax_3d)
            
            # Save 3D plot
            plt.savefig(f"{self.results_dir}/figures/{dataset_name}_clustering_3d.png", dpi=300, bbox_inches='tight')
            plt.close(fig_3d)
        
        # Save 2D plots
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/figures/{dataset_name}_clustering_{method_label}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Visualization saved: {dataset_name}_clustering_{method_label}.png")
    
    def test_dataset(self, dataset_name):
        """Test a single dataset with PBP approach."""
        print(f"\n{'='*60}")
        print(f"Testing {dataset_name} dataset")
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
        
        # Apply PBP reduction
        X_reduced, method_label = self.apply_pbp_reduction(X)
        print(f"Reduced shape: {X_reduced.shape}")
        
        # Evaluate clustering
        evaluation = self.evaluate_clustering(X_reduced, y)
        
        print(f"Clustering Results:")
        print(f"  Silhouette Score: {evaluation['silhouette_score']:.4f}")
        print(f"  Davies-Bouldin Score: {evaluation['davies_bouldin_score']:.4f}")
        print(f"  Number of clusters: {evaluation['n_clusters']}")
        
        # Visualize results
        self.visualize_results(X_reduced, y, evaluation['y_pred'], dataset_name, metadata, method_label)
        
        # Store results
        self.results[dataset_name] = {
            'X_reduced': X_reduced,
            'y_true': y,
            'y_pred': evaluation['y_pred'],
            'evaluation': evaluation,
            'method': method_label,
            'metadata': metadata
        }
        
        return self.results[dataset_name]
    
    def test_all_datasets(self):
        """Test all available datasets."""
        print("Testing all datasets with PBP approach...")
        
        # List of datasets to test
        datasets = [
            'iris', 'breast_cancer', 'wine', 'digits', 'diabetes',
            'sonar', 'glass', 'vehicle', 'ecoli', 'yeast'
        ]
        
        
        successful_tests = 0
        
        for dataset_name in datasets:
            try:
                result = self.test_dataset(dataset_name)
                if result:
                    successful_tests += 1
            except Exception as e:
                print(f"Error testing {dataset_name}: {e}")
        
        print(f"\n{'='*60}")
        print(f"Testing completed: {successful_tests}/{len(datasets)} datasets successful")
        print(f"{'='*60}")
        
        # Generate summary report
        self.generate_summary_report(self.results)
        
        return self.results
    
    def generate_summary_report(self, results):
        """Generate a summary report of all test results."""
        print(f"\n{'='*80}")
        print("PBP TESTING SUMMARY REPORT")
        print(f"{'='*80}")
        
        if not results:
            print("No results to summarize.")
            return
        
        # Create summary DataFrame
        summary_data = []
        
        for dataset_name, result in results.items():
            evaluation = result['evaluation']
            summary_data.append({
                'Dataset': dataset_name,
                'Method': result['method'],
                'Silhouette_Score': evaluation['silhouette_score'],
                'Davies_Bouldin_Score': evaluation['davies_bouldin_score'],
                'N_Clusters': evaluation['n_clusters'],
                'Original_Shape': result['X_reduced'].shape,
                'N_Samples': result['X_reduced'].shape[0]
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary to CSV
        output_file = f"{self.results_dir}/tables/test_summary_pbp_vector.csv"
        summary_df.to_csv(output_file, index=False)
        print(f"Summary saved to: {output_file}")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print("-" * 50)
        
        print(f"Total datasets tested: {len(results)}")
        print(f"Average Silhouette Score: {summary_df['Silhouette_Score'].mean():.4f}")
        print(f"Average Davies-Bouldin Score: {summary_df['Davies_Bouldin_Score'].mean():.4f}")
        
        # Best performing dataset
        best_dataset = summary_df.loc[summary_df['Silhouette_Score'].idxmax()]
        print(f"\nBest performing dataset: {best_dataset['Dataset']}")
        print(f"  Silhouette Score: {best_dataset['Silhouette_Score']:.4f}")
        print(f"  Davies-Bouldin Score: {best_dataset['Davies_Bouldin_Score']:.4f}")
        
        # Dataset rankings
        print("\nDataset rankings (by Silhouette Score):")
        rankings = summary_df.sort_values('Silhouette_Score', ascending=False)
        for _, row in rankings.iterrows():
            print(f"  {row['Dataset']}: {row['Silhouette_Score']:.4f}")


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