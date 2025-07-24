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
import argparse
import sys
import logging
import tqdm
logging.getLogger('matplotlib.font_manager').disabled = True


# Import PBP modules
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from pbp import pbp_vector
    PBP_AVAILABLE = True
except ImportError:
    try:
        from src.pbp.core import pbp_vector
        PBP_AVAILABLE = True
    except ImportError:
        print("Warning: pbp modules not found. Install required dependencies.")
        pbp_vector = None
        PBP_AVAILABLE = False

# Import the improved DatasetTransformer from loader.py
from src.data.loader import DatasetTransformer

class DatasetTester:
    """Test datasets with pbp_vector approach."""
    
    def __init__(self, data_dir='./data', results_dir='./results'):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.results = {}
        self.transformer = DatasetTransformer()
        
    def load_dataset(self, dataset_name):
        """Load dataset using the improved DatasetTransformer."""
        print(f"Loading {dataset_name} with improved preprocessing...")
        
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
    
    def apply_pbp_reduction(self, X, dataset_name):
        """Apply pbp_vector reduction to dataset."""
        if not PBP_AVAILABLE:
            print("PBP module not available. Using PCA as fallback.")
            X_reduced, method_label = self.apply_pca_reduction(X)
            return X_reduced, method_label
        
        print("Applying actual PBP vector reduction...")
        reduced_samples = []
        
        if os.path.exists(os.path.join(self.data_dir, f'{dataset_name}_pbp_features.npy')):
            reduced_samples = np.load(os.path.join(self.data_dir, f'{dataset_name}_pbp_features.npy'))
        else:
            for i in tqdm.tqdm(range(X.shape[0])):
                try:
                    pbp_result = pbp_vector(X[i])
                    reduced_samples.append(pbp_result)
                    
                except Exception as e:
                    print(f"Error processing sample {i}: {e}")
                    # Use original sample if reduction fails
                    reduced_samples.append(X[i].flatten())
        
        reduced_samples = np.array(reduced_samples)
        zero_columns = np.all(reduced_samples == 0, axis=0)
        print(f"Has zero columns: {np.sum(zero_columns)} / {reduced_samples.shape[1]}")
        reduced_samples = reduced_samples[:, ~zero_columns]

        np.save(os.path.join(self.data_dir, f'{dataset_name}_pbp_features.npy'), reduced_samples)

        return reduced_samples, "PBP"
    
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
            'cluster_centers': kmeans.cluster_centers_
        }
    
    def visualize_results(self, X_reduced, y_true, y_pred, dataset_name, metadata, method_label="PBP"):
        """Visualize clustering results."""
        # Ensure y_true and y_pred are numeric and the same length as X_reduced
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.dtype == object or y_true.dtype.type is np.str_:
            y_true = pd.Categorical(y_true).codes
        if y_pred.dtype == object or y_pred.dtype.type is np.str_:
            y_pred = pd.Categorical(y_pred).codes
        if len(y_true) != X_reduced.shape[0]:
            print(f"Skipping visualization for {dataset_name}: y_true length mismatch.")
            return
        if len(y_pred) != X_reduced.shape[0]:
            print(f"Skipping visualization for {dataset_name}: y_pred length mismatch.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        scatter1 = axes[0].scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_true, cmap='viridis', alpha=0.7)
        axes[0].set_title(f'{dataset_name} - True Labels\n({method_label})')
        axes[0].set_xlabel('Component 1')
        axes[0].set_ylabel('Component 2')
        plt.colorbar(scatter1, ax=axes[0])

        scatter2 = axes[1].scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
        axes[1].set_title(f'{dataset_name} - Predicted Clusters\n({method_label})')
        axes[1].set_xlabel('Component 1')
        axes[1].set_ylabel('Component 2')
        plt.colorbar(scatter2, ax=axes[1])

        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/figures/{dataset_name}_clustering_{method_label}.png", dpi=300, bbox_inches='tight')
        # plt.show()
        
        # 3D visualization if we have 3+ components
        if X_reduced.shape[1] >= 3:
            fig = plt.figure(figsize=(15, 6))
            
            ax1 = fig.add_subplot(121, projection='3d')
            scatter1 = ax1.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], 
                                  c=y_true, cmap='viridis', alpha=0.7)
            ax1.set_title(f'{dataset_name} - True Labels (3D)')
            ax1.set_xlabel('Component 1')
            ax1.set_ylabel('Component 2')
            ax1.set_zlabel('Component 3')
            
            ax2 = fig.add_subplot(122, projection='3d')
            scatter2 = ax2.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], 
                                  c=y_pred, cmap='viridis', alpha=0.7)
            ax2.set_title(f'{dataset_name} - Predicted Clusters (3D)')
            ax2.set_xlabel('Component 1')
            ax2.set_ylabel('Component 2')
            ax2.set_zlabel('Component 3')
            
            plt.tight_layout()
            plt.savefig(f"{self.results_dir}/figures/{dataset_name}_clustering_3d.png", dpi=300, bbox_inches='tight')
            # plt.show()
    
    def test_dataset(self, dataset_name):
        """Test a specific dataset with the pbp_vector approach."""
        print(f"\n{'='*50}")
        print(f"Testing dataset: {dataset_name}")
        print(f"{'='*50}")
        
        # Load dataset using the improved DatasetTransformer
        dataset = self.load_dataset(dataset_name)
        if dataset is None:
            return None
        
        X = dataset['X']
        y = dataset['y']
        metadata = dataset['metadata']
        
        print(f"Original shape: {X.shape}")
        print(f"Number of classes: {len(np.unique(y))}")
        print(f"Feature names: {metadata['feature_names']}")
        print(f"Measurement names: {metadata['measurement_names']}")
        
        # Apply pbp_vector reduction
        print("\nApplying pbp_vector reduction...")
        X_reduced, method_label = self.apply_pbp_reduction(X, dataset_name)
        
        print(f"Reduced shape: {X_reduced.shape}")
        print(f"Dimensionality reduction: {X.shape[1] * X.shape[2]} -> {X_reduced.shape[1]}")
        
        # Evaluate clustering
        print("\nEvaluating clustering performance...")
        clustering_results = self.evaluate_clustering(X_reduced, y)
        
        print(f"Silhouette Score: {clustering_results['silhouette_score']:.4f}")
        print(f"Davies-Bouldin Score: {clustering_results['davies_bouldin_score']:.4f}")
        
        # Visualize results
        print("\nGenerating visualizations...")
        # self.visualize_results(X_reduced, y, clustering_results['y_pred'], 
                            #  dataset_name, metadata, method_label=method_label)
        
        # Store results
        self.results[dataset_name] = {
            'original_shape': X.shape,
            'reduced_shape': X_reduced.shape,
            'clustering_results': clustering_results,
            'metadata': metadata
        }
        
        return self.results[dataset_name]
    
    def test_all_datasets(self, include_transformed_digits=True):
        """Test all available datasets."""
        # List of all datasets including conforming datasets
        datasets = [
            'iris',
            'breast_cancer', 
            'wine',
            'diabetes',
            'glass',
            'digits',
            'sonar',
            'vehicle',
            'ecoli',
            'yeast',
            'seeds',
            'thyroid',
            'pima',
            'ionosphere',
            'spectf',
            'glass_conforming',
            'covertype',
            '# # olivetti_faces',
            'kddcup99',
            'linnerrud',
            'species_distribution'
        ]
        
        # Add transformed digits datasets if requested
        if include_transformed_digits:
            # Check if transformed datasets exist
            try:
                if os.path.exists(f"{self.data_dir}/digits_transformed_X.npy"):
                    datasets.append('digits_transformed')
            except Exception as e:
                print(f"Error checking transformed digits datasets: {e}")
        
        results = {}
        
        for dataset_name in tqdm.tqdm(datasets):
            try:
                result = self.test_dataset(dataset_name)
                if result:
                    results[dataset_name] = result
            except Exception as e:
                print(f"Error testing {dataset_name}: {e}")
        
        # Generate summary report
        self.generate_summary_report(results)
        
        return results
    
    def generate_summary_report(self, results):
        """Generate a summary report of all test results."""
        print(f"\n{'='*80}")
        print("SUMMARY REPORT - PBP Vector Approach")
        print(f"{'='*80}")
        
        if not results:
            print("No results available for summary report.")
            return
        
        # Create summary table
        summary_data = []
        for name, result in results.items():
            summary_data.append({
                'Dataset': name,
                'Original Shape': f"{result['original_shape'][1]}x{result['original_shape'][2]}",
                'Reduced Shape': f"{result['reduced_shape'][1]}",
                'Silhouette Score': f"{result['clustering_results']['silhouette_score']:.4f}",
                'Davies-Bouldin Score': f"{result['clustering_results']['davies_bouldin_score']:.4f}",
                'Description': result['metadata']['description']
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Save summary to file
        summary_df.to_csv(f"{self.results_dir}/tables/test_summary_pbp_vector.csv", index=False)
        print(f"\nSummary saved to {self.results_dir}/tables/test_summary_pbp_vector.csv")
        
        # Find best performing datasets
        best_silhouette = max(results.items(), 
                             key=lambda x: x[1]['clustering_results']['silhouette_score'])
        best_davies = min(results.items(), 
                         key=lambda x: x[1]['clustering_results']['davies_bouldin_score'])
        
        print(f"\nBest Silhouette Score: {best_silhouette[0]} ({best_silhouette[1]['clustering_results']['silhouette_score']:.4f})")
        print(f"Best Davies-Bouldin Score: {best_davies[0]} ({best_davies[1]['clustering_results']['davies_bouldin_score']:.4f})")
        
        # Performance statistics
        silhouette_scores = [r['clustering_results']['silhouette_score'] for r in results.values()]
        davies_scores = [r['clustering_results']['davies_bouldin_score'] for r in results.values()]
        
        print(f"\nPerformance Statistics:")
        print(f"Average Silhouette Score: {np.mean(silhouette_scores):.4f} ± {np.std(silhouette_scores):.4f}")
        print(f"Average Davies-Bouldin Score: {np.mean(davies_scores):.4f} ± {np.std(davies_scores):.4f}")

def main():
    """Main function to run dataset testing."""
    # Create data directory if it doesn't exist
    data_dir = './data'
    results_dir = './results'
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f"{results_dir}/figures", exist_ok=True)
    os.makedirs(f"{results_dir}/tables", exist_ok=True)
    
    # Initialize tester
    tester = DatasetTester(data_dir, results_dir)
    
    parser = argparse.ArgumentParser(description="Test datasets with PBP vector approach")
    parser.add_argument("-d", "--dataset_name", default='all', help="Name of the dataset to test")
    args = parser.parse_args()

    # Test all datasets
    if args.dataset_name == 'all':
        results = tester.test_all_datasets()
    else:
        results = tester.test_dataset(args.dataset_name)
        exit()
    
    print(f"\nTesting completed! Results for {len(results)} datasets.")
    print(f"Check {results_dir} for saved visualizations and summary.")

if __name__ == "__main__":
    main() 