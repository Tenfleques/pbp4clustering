#!/usr/bin/env python3
"""
PBP Datasets Testing Script

This script tests all datasets with the pbp_vector approach, including visualization 
and clustering analysis. Now includes aggregation function optimization.
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
import logging
import tqdm
import json
logging.getLogger('matplotlib.font_manager').disabled = True

# Import PBP modules
try:
    from src.pbp.core import pbp_vector
    PBP_AVAILABLE = True
except ImportError:
    print("Warning: pbp modules not found. Install required dependencies.")
    pbp_vector = None
    PBP_AVAILABLE = False

# Import aggregation optimization
try:
    from src.analysis.aggregation_optimization import AggregationOptimizer
    from src.pbp.aggregation_functions import get_aggregation_function, get_recommended_aggregation_functions
    AGGREGATION_OPTIMIZATION_AVAILABLE = True
except ImportError:
    print("Warning: Aggregation optimization not available. Using default sum aggregation.")
    AGGREGATION_OPTIMIZATION_AVAILABLE = False

# Import the centralized DatasetTransformer
from src.data.loader import DatasetTransformer

class DatasetTester:
    """Test datasets with pbp_vector approach."""
    
    def __init__(self, data_dir='./data', results_dir='./results', use_optimized_aggregation=True):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.results = {}
        self.transformer = DatasetTransformer()
        self.use_optimized_aggregation = use_optimized_aggregation and AGGREGATION_OPTIMIZATION_AVAILABLE
        self.aggregation_optimizer = None
        self.optimal_aggregation_functions = {}
        
        if self.use_optimized_aggregation:
            print("✓ Aggregation function optimization enabled")
            self.aggregation_optimizer = AggregationOptimizer(random_state=42)
        else:
            print("⚠ Using default sum aggregation")
        
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
    
    def get_optimal_aggregation_function(self, dataset_name, dataset):
        """Get the optimal aggregation function for a dataset."""
        if not self.use_optimized_aggregation:
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
    
    def apply_pbp_reduction(self, X, dataset_name, dataset):
        """Apply pbp_vector reduction to dataset with optimal aggregation function."""
        if not PBP_AVAILABLE:
            print("PBP module not available. Using PCA as fallback.")
            X_reduced, method_label = self.apply_pca_reduction(X)
            return X_reduced, method_label
        
        # Get optimal aggregation function
        agg_func = self.get_optimal_aggregation_function(dataset_name, dataset)
        agg_func_name = self.optimal_aggregation_functions.get(dataset_name, 'sum')
        
        print(f"Applying PBP vector reduction with {agg_func_name} aggregation...")
        reduced_samples = []
        
        # Check for cached PBP features
        cache_file = os.path.join(self.data_dir, f'{dataset_name}_pbp_features_{agg_func_name}.npy')
        if os.path.exists(cache_file):
            print(f"  Loading cached PBP features with {agg_func_name} aggregation...")
            reduced_samples = np.load(cache_file)
        else:
            print(f"  Computing PBP features with {agg_func_name} aggregation...")
            for i in tqdm.tqdm(range(X.shape[0])):
                try:
                    pbp_result = pbp_vector(X[i], agg_func)
                    reduced_samples.append(pbp_result)
                except Exception as e:
                    print(f"Error processing sample {i}: {e}")
                    # Use original sample if reduction fails
                    reduced_samples.append(X[i].flatten())
            
            reduced_samples = np.array(reduced_samples)
            
            # Save to cache
            try:
                np.save(cache_file, reduced_samples)
                print(f"  Saved PBP features to cache: {cache_file}")
            except Exception as e:
                print(f"  Warning: Could not cache PBP features: {e}")
        
        # Remove zero columns
        zero_columns = np.all(reduced_samples == 0, axis=0)
        print(f"Has zero columns: {np.sum(zero_columns)} / {reduced_samples.shape[1]}")
        reduced_samples = reduced_samples[:, ~zero_columns]

        return reduced_samples, f"PBP ({agg_func_name})"
    
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
    
    def get_pbp_component_names(self, vector_length, num_rows):
        """
        Generate meaningful names for PBP vector components based on their position.
        
        Args:
            vector_length: Length of the PBP vector (2^num_rows - 1)
            num_rows: Number of rows in the original matrix
        
        Returns:
            List of component names
        """
        from src.pbp.core import decode_var
        
        component_names = []
        for i in range(vector_length):
            decoded = decode_var(i)
            if decoded == "":
                component_names.append("Aggregated (min)")
            else:
                component_names.append(f"Aggregated ({decoded})")
        
        return component_names
    
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

        # Generate PBP component names if this is a PBP method
        if "PBP" in method_label:
            # Estimate original matrix dimensions from the method label or metadata
            # For now, we'll use a reasonable default based on the reduced shape
            estimated_rows = min(3, X_reduced.shape[1])  # Conservative estimate
            pbp_component_names = self.get_pbp_component_names(X_reduced.shape[1], estimated_rows)
        else:
            pbp_component_names = None

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        scatter1 = axes[0].scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_true, cmap='viridis', alpha=0.7)
        axes[0].set_title(f'{dataset_name} - True Labels\n({method_label})')
        
        # Use PBP component names if available, otherwise fallback to generic names
        if pbp_component_names and len(pbp_component_names) >= 2:
            axes[0].set_xlabel(pbp_component_names[0])
            axes[0].set_ylabel(pbp_component_names[1])
        else:
            axes[0].set_xlabel('Component 1')
            axes[0].set_ylabel('Component 2')
        plt.colorbar(scatter1, ax=axes[0])

        scatter2 = axes[1].scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
        axes[1].set_title(f'{dataset_name} - Predicted Clusters\n({method_label})')
        
        # Use PBP component names if available, otherwise fallback to generic names
        if pbp_component_names and len(pbp_component_names) >= 2:
            axes[1].set_xlabel(pbp_component_names[0])
            axes[1].set_ylabel(pbp_component_names[1])
        else:
            axes[1].set_xlabel('Component 1')
            axes[1].set_ylabel('Component 2')
        plt.colorbar(scatter2, ax=axes[1])

        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/figures/{dataset_name}_clustering_{method_label.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
        # plt.show()
        
        # 3D visualization if we have 3+ components
        if X_reduced.shape[1] >= 3:
            fig = plt.figure(figsize=(15, 6))
            
            ax1 = fig.add_subplot(121, projection='3d')
            scatter1 = ax1.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], 
                                  c=y_true, cmap='viridis', alpha=0.7)
            ax1.set_title(f'{dataset_name} - True Labels (3D)')
            
            # Use PBP component names if available, otherwise fallback to generic names
            if pbp_component_names and len(pbp_component_names) >= 3:
                ax1.set_xlabel(pbp_component_names[0])
                ax1.set_ylabel(pbp_component_names[1])
                ax1.set_zlabel(pbp_component_names[2])
            else:
                ax1.set_xlabel('Component 1')
                ax1.set_ylabel('Component 2')
                ax1.set_zlabel('Component 3')
            
            ax2 = fig.add_subplot(122, projection='3d')
            scatter2 = ax2.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], 
                                  c=y_pred, cmap='viridis', alpha=0.7)
            ax2.set_title(f'{dataset_name} - Predicted Clusters (3D)')
            
            # Use PBP component names if available, otherwise fallback to generic names
            if pbp_component_names and len(pbp_component_names) >= 3:
                ax2.set_xlabel(pbp_component_names[0])
                ax2.set_ylabel(pbp_component_names[1])
                ax2.set_zlabel(pbp_component_names[2])
            else:
                ax2.set_xlabel('Component 1')
                ax2.set_ylabel('Component 2')
                ax2.set_zlabel('Component 3')
            
            plt.tight_layout()
            plt.savefig(f"{self.results_dir}/figures/{dataset_name}_clustering_3d_{method_label.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
            # plt.show()
    
    def test_dataset(self, dataset_name, visualize=False):
        """Test a specific dataset with the pbp_vector approach."""
        print(f"\n{'='*50}")
        print(f"Testing dataset: {dataset_name}")
        print(f"{'='*50}")
        
        # Load dataset using the centralized DatasetTransformer
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
        
        # Apply pbp_vector reduction with optimal aggregation
        print("\nApplying pbp_vector reduction...")
        X_reduced, method_label = self.apply_pbp_reduction(X, dataset_name, dataset)
        
        print(f"Reduced shape: {X_reduced.shape}")
        print(f"Dimensionality reduction: {X.shape[1] * X.shape[2]} -> {X_reduced.shape[1]}")
        
        # Evaluate clustering
        print("\nEvaluating clustering performance...")
        clustering_results = self.evaluate_clustering(X_reduced, y)
        
        print(f"Silhouette Score: {clustering_results['silhouette_score']:.4f}")
        print(f"Davies-Bouldin Score: {clustering_results['davies_bouldin_score']:.4f}")
        
        # Get aggregation function info
        agg_func_name = self.optimal_aggregation_functions.get(dataset_name, 'sum')
        
        # Visualize results
        print("\nGenerating visualizations...")
        if visualize:
            self.visualize_results(X_reduced, y, clustering_results['y_pred'], 
                                dataset_name, metadata, method_label=method_label)
        
        # Store results
        self.results[dataset_name] = {
            'original_shape': X.shape,
            'reduced_shape': X_reduced.shape,
            'clustering_results': clustering_results,
            'metadata': metadata,
            'aggregation_function': agg_func_name,
            'method_label': method_label
        }
        
        return self.results[dataset_name]
    
    def test_all_datasets(self, visualize=False):
        """Test all available datasets."""
        # Get all available datasets from the consolidated loader
        try:
            from src.data.consolidated_loader import ConsolidatedDatasetLoader
            loader = ConsolidatedDatasetLoader()
            dataset_config = loader.get_available_datasets()
            
            # Flatten all datasets into a single list
            all_datasets = []
            for category, datasets in dataset_config.items():
                all_datasets.extend(datasets)
            
            print(f"Found {len(all_datasets)} datasets across {len(dataset_config)} categories")
            print("Testing all available datasets...")
            
        except Exception as e:
            print(f"Error getting dataset configuration: {e}")
            # Fallback to a subset of datasets
            all_datasets = [
                'iris', 'breast_cancer', 'wine', 'digits', 'diabetes',
                'sonar', 'glass', 'vehicle', 'ecoli', 'yeast',
                'seeds', 'thyroid', 'pima', 'ionosphere', 'spectf',
                'glass_conforming', 'covertype', 'kddcup99', 'linnerrud', 'species_distribution'
            ]
            print(f"Using fallback dataset list: {len(all_datasets)} datasets")
        
        results = {}
        
        for dataset_name in tqdm.tqdm(all_datasets, desc="Testing datasets"):
            try:
                result = self.test_dataset(dataset_name, visualize=visualize)
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
        print("SUMMARY REPORT - PBP Vector Approach with Aggregation Optimization")
        print(f"{'='*80}")
        
        if not results:
            print("No results available for summary report.")
            return
        
        # Create summary table
        summary_data = []
        pbp_component_info = {}
        
        for name, result in results.items():
            # Get PBP component names if this is a PBP method
            pbp_components = []
            if "PBP" in result.get('method_label', ''):
                try:
                    reduced_shape = result['reduced_shape'][1]
                    estimated_rows = min(3, reduced_shape)  # Conservative estimate
                    pbp_components = self.get_pbp_component_names(reduced_shape, estimated_rows)
                    pbp_component_info[name] = pbp_components
                except Exception as e:
                    print(f"Warning: Could not generate PBP component names for {name}: {e}")
            
            summary_data.append({
                'Dataset': name,
                'Original Shape': f"{result['original_shape'][1]}x{result['original_shape'][2]}",
                'Reduced Shape': f"{result['reduced_shape'][1]}",
                'Aggregation Function': result.get('aggregation_function', 'sum'),
                'Silhouette Score': f"{result['clustering_results']['silhouette_score']:.4f}",
                'Davies-Bouldin Score': f"{result['clustering_results']['davies_bouldin_score']:.4f}",
                'Description': result['metadata']['description'],
                'PBP Components': len(pbp_components) if pbp_components else 'N/A'
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Save summary to file
        summary_df.to_csv(f"{self.results_dir}/tables/test_summary_pbp_vector_optimized.csv", index=False)
        print(f"\nSummary saved to {self.results_dir}/tables/test_summary_pbp_vector_optimized.csv")
        
        # Print PBP component information for datasets that use PBP
        if pbp_component_info:
            print(f"\n{'='*60}")
            print("PBP COMPONENT BREAKDOWN")
            print(f"{'='*60}")
            
            for dataset_name, components in pbp_component_info.items():
                print(f"\n{dataset_name}:")
                for i, component in enumerate(components):
                    print(f"  Component {i+1}: {component}")
        
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
        
        # Aggregation function usage statistics
        if self.use_optimized_aggregation:
            agg_func_counts = {}
            for result in results.values():
                agg_func = result.get('aggregation_function', 'sum')
                agg_func_counts[agg_func] = agg_func_counts.get(agg_func, 0) + 1
            
            print(f"\nAggregation Function Usage:")
            for agg_func, count in sorted(agg_func_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {agg_func}: {count} datasets")


def main():
    """Main function to run dataset testing."""
    # Create data directory if it doesn't exist
    data_dir = './data'
    results_dir = './results'
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true", help="Visualize results")
    args = parser.parse_args()
    visualize = args.visualize
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f"{results_dir}/figures", exist_ok=True)
    os.makedirs(f"{results_dir}/tables", exist_ok=True)
    
    # Initialize tester
    tester = DatasetTester(data_dir, results_dir)
    
    # Test all datasets
    results = tester.test_all_datasets(visualize=visualize  )
    
    print(f"\nTesting completed! Results for {len(results)} datasets.")
    print(f"Check {results_dir} for saved visualizations and summary.")

if __name__ == "__main__":
    main() 