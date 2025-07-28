#!/usr/bin/env python3
"""
Test Script for Pseudo-Boolean Polynomial Dimensionality Reduction

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
import json
import logging
from tqdm import tqdm
import argparse
logging.getLogger('matplotlib.font_manager').disabled = True

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import PBP modules
try:
    from ..pbp.core import pbp_vector
    PBP_AVAILABLE = True
except ImportError:
    print("Warning: pbp modules not found. Install required dependencies.")
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

from ..data.dataset_config import get_testing_datasets

class DatasetTester:
    """Test datasets with pbp_vector approach."""
    
    def __init__(self, data_dir='./data', results_dir='./results', use_optimized_aggregation=True):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.results = {}
        self.use_optimized_aggregation = use_optimized_aggregation and AGGREGATION_OPTIMIZATION_AVAILABLE
        self.aggregation_optimizer = None
        self.optimal_aggregation_functions = {}
        
        if self.use_optimized_aggregation:
            print("✓ Aggregation function optimization enabled")
            self.aggregation_optimizer = AggregationOptimizer(random_state=42)
        else:
            print("⚠ Using default sum aggregation")
        
    def load_dataset(self, dataset_name):
        """Load dataset using the centralized ConsolidatedDatasetLoader."""
        print(f"Loading {dataset_name} using centralized loader...")
        
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
            for i in tqdm(range(X.shape[0])):
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
        from ..pbp.core import decode_var
        
        component_names = []
        for i in range(vector_length):
            decoded = decode_var(i)
            if decoded == "":
                component_names.append("Aggregated (min)")
            else:
                component_names.append(f"Aggregated ({decoded})")
        
        return component_names
    
    def select_informative_columns(self, X_reduced, y_labels, n_components=2):
        """
        Select the most informative columns for visualization.
        
        Args:
            X_reduced: Reduced feature matrix
            y_labels: Cluster labels
            n_components: Number of components to select
            
        Returns:
            tuple: (selected_indices, X_selected, component_names)
        """
        if X_reduced.shape[1] <= n_components:
            # If we have fewer columns than needed, use all
            return np.arange(X_reduced.shape[1]), X_reduced, None
        
        # Calculate informativeness scores for each column
        informativeness_scores = []
        
        for col in range(X_reduced.shape[1]):
            col_data = X_reduced[:, col]
            
            # Calculate overall variance
            overall_variance = np.var(col_data)
            
            # Calculate within-cluster variance
            within_cluster_variances = []
            for cluster_id in np.unique(y_labels):
                cluster_mask = y_labels == cluster_id
                if np.sum(cluster_mask) > 1:  # Need at least 2 points for variance
                    cluster_variance = np.var(col_data[cluster_mask])
                    within_cluster_variances.append(cluster_variance)
            
            # Calculate between-cluster variance
            cluster_means = []
            for cluster_id in np.unique(y_labels):
                cluster_mask = y_labels == cluster_id
                cluster_mean = np.mean(col_data[cluster_mask])
                cluster_means.append(cluster_mean)
            between_cluster_variance = np.var(cluster_means) if len(cluster_means) > 1 else 0
            
            # Calculate informativeness score
            # Higher score = more informative (high overall variance, low within-cluster variance, high between-cluster variance)
            if overall_variance > 0:
                within_variance = np.mean(within_cluster_variances) if within_cluster_variances else 0
                informativeness_score = (between_cluster_variance / (within_variance + 1e-8)) * overall_variance
            else:
                informativeness_score = 0
            
            informativeness_scores.append(informativeness_score)
        
        # Select top n_components columns
        top_indices = np.argsort(informativeness_scores)[::-1][:n_components]
        
        # Ensure we have at least 2 different columns for 2D plots
        if len(top_indices) < 2:
            # If we don't have enough informative columns, add the next best ones
            remaining_indices = np.setdiff1d(np.arange(X_reduced.shape[1]), top_indices)
            if len(remaining_indices) > 0:
                additional_needed = 2 - len(top_indices)
                additional_indices = remaining_indices[:additional_needed]
                top_indices = np.concatenate([top_indices, additional_indices])
        
        # Sort indices to maintain order
        top_indices = np.sort(top_indices)
        
        # Ensure we don't exceed the number of available columns
        top_indices = top_indices[:min(n_components, X_reduced.shape[1])]
        
        return top_indices, X_reduced[:, top_indices], None
    
    def create_dataset_info_table(self, dataset_name, metadata, X_shape, y_unique):
        """
        Create a comprehensive dataset information table.
        
        Args:
            dataset_name: Name of the dataset
            metadata: Dataset metadata
            X_shape: Shape of the feature matrix
            y_unique: Unique target values
            
        Returns:
            matplotlib figure with dataset information table
        """
        # Get additional dataset information from dataset_info.py if available
        try:
            from src.data.dataset_info import get_dataset_info
            dataset_info = get_dataset_info(dataset_name)
        except ImportError:
            dataset_info = None
        
        # Prepare table data
        table_data = []
        
        # Basic information
        table_data.append(['Dataset Name', dataset_name])
        table_data.append(['Description', metadata.get('description', 'No description available')])
        table_data.append(['Data Type', metadata.get('data_type', 'Unknown')])
        table_data.append(['Source', metadata.get('source', 'Unknown')])
        
        # Shape information
        table_data.append(['Original Shape', f"{X_shape[0]} samples × {X_shape[1]}×{X_shape[2]} matrices"])
        table_data.append(['Matrix Structure', f"{X_shape[1]} rows × {X_shape[2]} columns"])
        table_data.append(['Total Features', f"{X_shape[1] * X_shape[2]}"])
        
        # Target information
        table_data.append(['Number of Classes', str(len(y_unique))])
        table_data.append(['Class Distribution', f"{dict(zip(y_unique, [np.sum(y_unique == c) for c in y_unique]))}"])
        
        # Feature information
        if metadata.get('feature_names'):
            feature_names = metadata['feature_names']
            if len(feature_names) <= 5:
                table_data.append(['Feature Categories', ', '.join(feature_names)])
            else:
                table_data.append(['Feature Categories', f"{', '.join(feature_names[:3])}... (+{len(feature_names)-3} more)"])
        
        if metadata.get('measurement_names'):
            measurement_names = metadata['measurement_names']
            if len(measurement_names) <= 5:
                table_data.append(['Measurement Types', ', '.join(measurement_names)])
            else:
                table_data.append(['Measurement Types', f"{', '.join(measurement_names[:3])}... (+{len(measurement_names)-3} more)"])
        
        # PBP transformation information
        if dataset_info and 'pbp_transformation' in dataset_info:
            pbp_info = dataset_info['pbp_transformation']
            table_data.append(['PBP Matrix Shape', pbp_info.get('matrix_shape', 'Unknown')])
            table_data.append(['PBP Rationale', pbp_info.get('rationale', 'Unknown')])
        
        # Domain and application information
        if dataset_info:
            table_data.append(['Domain', dataset_info.get('domain', 'Unknown')])
            table_data.append(['Sample Count', str(dataset_info.get('sample_count', 'Unknown'))])
        
        # Create the table
        fig, ax = plt.subplots(figsize=(12, len(table_data) * 0.4 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=table_data, 
                        colLabels=['Property', 'Value'],
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.3, 0.7])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color the header
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color alternating rows
        for i in range(1, len(table_data) + 1):
            for j in range(2):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        # Set title
        plt.title(f'Dataset Information: {dataset_name}', fontsize=14, fontweight='bold', pad=20)
        
        return fig
    
    def visualize_results(self, X_reduced, y_true, y_pred, dataset_name, metadata, method_label="PBP"):
        """Visualize clustering results with dataset information table and informative column selection."""
        # Ensure y_true and y_pred are numeric and the same length as X_reduced
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.dtype == object or y_true.dtype.type is np.str_:
            y_true = pd.Categorical(y_true).codes
        if y_pred.dtype == object or y_pred.dtype.type is np.str_:
            y_pred = pd.Categorical(y_pred).codes
        
        # Ensure X_reduced is 2D for visualization
        if len(X_reduced.shape) == 3:
            # If X_reduced is 3D, flatten it for visualization
            X_flat = X_reduced.reshape(X_reduced.shape[0], -1)
        else:
            X_flat = X_reduced
            
        if len(y_true) != X_flat.shape[0]:
            print(f"Skipping visualization for {dataset_name}: y_true length mismatch ({len(y_true)} vs {X_flat.shape[0]}).")
            return
        if len(y_pred) != X_flat.shape[0]:
            print(f"Skipping visualization for {dataset_name}: y_pred length mismatch ({len(y_pred)} vs {X_flat.shape[0]}).")
            return

        # Generate PBP component names if this is a PBP method
        if "PBP" in method_label:
            # Estimate original matrix dimensions from the method label or metadata
            # For now, we'll use a reasonable default based on the reduced shape
            estimated_rows = min(3, X_flat.shape[1])  # Conservative estimate
            pbp_component_names = self.get_pbp_component_names(X_flat.shape[1], estimated_rows)
        else:
            pbp_component_names = None

        # Select informative columns for 2D visualization
        informative_indices_2d, X_2d, _ = self.select_informative_columns(X_flat, y_pred, n_components=2)
        
        # Create 2D visualization with dataset info table
        fig = plt.figure(figsize=(20, 8))
        
        # Create subplot grid: 1 row, 3 columns (table, clustering plot, space for legend)
        gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 0.1])
        
        # Dataset information table (left)
        ax_table = fig.add_subplot(gs[0])
        
        # Create table data directly instead of copying from another figure
        table_data = []
        
        # Basic information
        table_data.append(['Dataset Name', dataset_name])
        table_data.append(['Description', metadata.get('description', 'No description available')])
        table_data.append(['Data Type', metadata.get('data_type', 'Unknown')])
        table_data.append(['Source', metadata.get('source', 'Unknown')])
        
        # Shape information
        table_data.append(['Original Shape', f"{X_flat.shape[0]} samples × {X_flat.shape[1]} features"])
        table_data.append(['Reduced Shape', f"{X_2d.shape[1]} components"])
        table_data.append(['Total Features', f"{X_flat.shape[1]}"])
        
        # Target information
        unique_classes = np.unique(y_true)
        table_data.append(['Number of Classes', str(len(unique_classes))])
        
        # Create a proper class distribution subtable
        class_dist = dict(zip(unique_classes, [np.sum(y_true == c) for c in unique_classes]))
        
        # Create horizontal table format (max 10 classes)
        max_classes_to_show = 10
        sorted_classes = sorted(class_dist.items())
        
        if len(sorted_classes) <= max_classes_to_show:
            # Show all classes
            counts = [str(count) for _, count in sorted_classes]
            class_dist_text = "Count| " + " | ".join(counts)
        else:
            # Show first 10 classes and add ellipsis
            counts = [str(count) for _, count in sorted_classes[:max_classes_to_show]]
            class_dist_text = "Count| " + " | ".join(counts) + " | ..."
        
        table_data.append(['Class Distribution', class_dist_text])
        
        # Feature information
        if metadata.get('feature_names'):
            feature_names = metadata['feature_names']
            if len(feature_names) <= 5:
                table_data.append(['Feature Categories', ', '.join(feature_names)])
            else:
                table_data.append(['Feature Categories', f"{', '.join(feature_names[:3])}... (+{len(feature_names)-3} more)"])
        
        if metadata.get('measurement_names'):
            measurement_names = metadata['measurement_names']
            if len(measurement_names) <= 5:
                table_data.append(['Measurement Types', ', '.join(measurement_names)])
            else:
                table_data.append(['Measurement Types', f"{', '.join(measurement_names[:3])}... (+{len(measurement_names)-3} more)"])
        
        # Create the table
        ax_table.axis('off')
        table = ax_table.table(cellText=table_data, 
                              colLabels=['Property', 'Value'],
                              cellLoc='left',
                              loc='center',
                              colWidths=[0.3, 0.7])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8) # Fixed scaling for class distribution
        
        # Color the header
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color alternating rows
        for i in range(1, len(table_data) + 1):
            for j in range(2):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        


        ax_table.set_title(f'Dataset Information: {dataset_name}', fontsize=12, fontweight='bold', pad=10)

        # Clustering plot (middle)
        ax_cluster = fig.add_subplot(gs[1])
        scatter = ax_cluster.scatter(X_2d[:, 0], X_2d[:, 1], c=y_pred, cmap='viridis', alpha=0.7, s=50)
        ax_cluster.set_title(f'Predicted Clusters\n({method_label})', fontsize=12, fontweight='bold')
        
        # Use PBP component names if available, otherwise use informative column indices
        if pbp_component_names and len(pbp_component_names) >= max(informative_indices_2d) + 1:
            ax_cluster.set_xlabel(pbp_component_names[informative_indices_2d[0]], fontsize=10)
            ax_cluster.set_ylabel(pbp_component_names[informative_indices_2d[1]], fontsize=10)
        else:
            ax_cluster.set_xlabel(f'Component {informative_indices_2d[0] + 1}', fontsize=10)
            ax_cluster.set_ylabel(f'Component {informative_indices_2d[1] + 1}', fontsize=10)
        
        # Add colorbar (right)
        ax_cbar = fig.add_subplot(gs[2])
        cbar = plt.colorbar(scatter, ax=ax_cbar, shrink=0.8)
        cbar.set_label('Cluster', fontsize=10)
        ax_cbar.axis('off')

        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/figures/{dataset_name}_clustering_{method_label.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3D visualization if we have 3+ components
        if X_flat.shape[1] >= 3:
            # Select informative columns for 3D visualization
            informative_indices_3d, X_3d, _ = self.select_informative_columns(X_flat, y_pred, n_components=3)
            
            fig = plt.figure(figsize=(20, 8))
            
            # Create subplot grid for 3D: 1 row, 3 columns (table, 3D plot, space for legend)
            gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 0.1])
            
            # Dataset information table (left) - reuse the same table data
            ax_table = fig.add_subplot(gs[0])
            ax_table.axis('off')
            table = ax_table.table(cellText=table_data, 
                                  colLabels=['Property', 'Value'],
                                  cellLoc='left',
                                  loc='center',
                                  colWidths=[0.3, 0.7])
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.8) # Fixed scaling for class distribution
            
            # Color the header
            for i in range(2):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Color alternating rows
            for i in range(1, len(table_data) + 1):
                for j in range(2):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#f0f0f0')
            
            
            
            ax_table.set_title(f'Dataset Information: {dataset_name}', fontsize=12, fontweight='bold', pad=10)
            
            # 3D clustering plot (middle)
            ax_cluster = fig.add_subplot(gs[1], projection='3d')
            scatter = ax_cluster.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], 
                                       c=y_pred, cmap='viridis', alpha=0.7, s=50)
            ax_cluster.set_title(f'Predicted Clusters (3D)\n({method_label})', fontsize=12, fontweight='bold')
            
            # Use PBP component names if available, otherwise use informative column indices
            if pbp_component_names and len(pbp_component_names) >= max(informative_indices_3d) + 1:
                ax_cluster.set_xlabel(pbp_component_names[informative_indices_3d[0]], fontsize=10)
                ax_cluster.set_ylabel(pbp_component_names[informative_indices_3d[1]], fontsize=10)
                ax_cluster.set_zlabel(pbp_component_names[informative_indices_3d[2]], fontsize=10)
            else:
                ax_cluster.set_xlabel(f'Component {informative_indices_3d[0] + 1}', fontsize=10)
                ax_cluster.set_ylabel(f'Component {informative_indices_3d[1] + 1}', fontsize=10)
                ax_cluster.set_zlabel(f'Component {informative_indices_3d[2] + 1}', fontsize=10)
            
            # Add colorbar (right)
            ax_cbar = fig.add_subplot(gs[2])
            cbar = plt.colorbar(scatter, ax=ax_cbar, shrink=0.8)
            cbar.set_label('Cluster', fontsize=10)
            ax_cbar.axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{self.results_dir}/figures/{dataset_name}_clustering_3d_{method_label.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def test_dataset(self, dataset_name):
        """Test a specific dataset with the pbp_vector approach."""
        print(f"\n{'='*50}")
        print(f"Testing dataset: {dataset_name}")
        print(f"{'='*50}")
        
        # Load dataset using the centralized ConsolidatedDatasetLoader
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
        # self.visualize_results(X_reduced, y, clustering_results['y_pred'], 
                            #  dataset_name, metadata, method_label=method_label)
        
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
    
    def test_all_datasets(self):
        """Test all available datasets."""
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
            print("Testing all available datasets...")
            
        except Exception as e:
            print(f"Error getting dataset configuration: {e}")
            # Fallback to a subset of datasets
            all_datasets = get_testing_datasets()
            print(f"Using fallback dataset list: {len(all_datasets)} datasets")
        
        results = {}
        
        for dataset_name in tqdm(all_datasets, desc="Testing datasets"):
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