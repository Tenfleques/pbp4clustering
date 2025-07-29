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
from sklearn.metrics import (
            silhouette_score, davies_bouldin_score, calinski_harabasz_score,
            adjusted_rand_score, normalized_mutual_info_score,
            homogeneity_score, completeness_score, v_measure_score,
            fowlkes_mallows_score
        )
from sklearn.decomposition import PCA
import os
import sys
import logging
import tqdm
import json
import argparse
from src.data.dataset_config import get_all_datasets
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
            from src.data.consolidated_loader import ConsolidatedDatasetLoader
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
        """Evaluate clustering performance on both full dataset and informative indices only."""
        if n_clusters is None:
            n_clusters = len(np.unique(y_true))
        
        # Ensure we have enough samples for clustering
        if X_reduced.shape[0] < n_clusters:
            print(f"Warning: Not enough samples ({X_reduced.shape[0]}) for {n_clusters} clusters")
            return {
                'silhouette_score': 0.0,
                'davies_bouldin_score': float('inf'),
                'y_pred': np.zeros(X_reduced.shape[0]),
                'cluster_centers': np.array([]),
                'informative_indices': [],
                'informative_silhouette_score': 0.0,
                'informative_davies_bouldin_score': float('inf'),
                'informative_y_pred': np.zeros(X_reduced.shape[0]),
                'calinski_harabasz_score': 0.0,
                'informative_calinski_harabasz_score': 0.0,
                'adjusted_rand_score': 0.0,
                'informative_adjusted_rand_score': 0.0,
                'normalized_mutual_info_score': 0.0,
                'informative_normalized_mutual_info_score': 0.0,
                'homogeneity_score': 0.0,
                'informative_homogeneity_score': 0.0,
                'completeness_score': 0.0,
                'informative_completeness_score': 0.0,
                'v_measure_score': 0.0,
                'informative_v_measure_score': 0.0,
                'fowlkes_mallows_score': 0.0,
                'informative_fowlkes_mallows_score': 0.0
            }
        
        
        
        # First, evaluate clustering on the full dataset
        kmeans_full = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        y_pred_full = kmeans_full.fit_predict(X_reduced)
        
        # Calculate comprehensive metrics for full dataset
        silhouette_full = silhouette_score(X_reduced, y_pred_full)
        davies_bouldin_full = davies_bouldin_score(X_reduced, y_pred_full)
        calinski_harabasz_full = calinski_harabasz_score(X_reduced, y_pred_full)
        
        # Classification metrics (comparing clustering to true labels)
        adjusted_rand_full = adjusted_rand_score(y_true, y_pred_full)
        normalized_mutual_info_full = normalized_mutual_info_score(y_true, y_pred_full)
        homogeneity_full = homogeneity_score(y_true, y_pred_full)
        completeness_full = completeness_score(y_true, y_pred_full)
        v_measure_full = v_measure_score(y_true, y_pred_full)
        fowlkes_mallows_full = fowlkes_mallows_score(y_true, y_pred_full)
        
        # Now, select informative columns and evaluate clustering on those only
        informative_indices, X_informative, _ = self.select_informative_columns(
            X_reduced, y_pred_full, n_components=min(3, X_reduced.shape[1])
        )
        
        # Evaluate clustering on informative indices only
        kmeans_informative = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        y_pred_informative = kmeans_informative.fit_predict(X_informative)
        
        # Calculate comprehensive metrics for informative indices only
        silhouette_informative = silhouette_score(X_informative, y_pred_informative)
        davies_bouldin_informative = davies_bouldin_score(X_informative, y_pred_informative)
        calinski_harabasz_informative = calinski_harabasz_score(X_informative, y_pred_informative)
        
        # Classification metrics for informative indices
        adjusted_rand_informative = adjusted_rand_score(y_true, y_pred_informative)
        normalized_mutual_info_informative = normalized_mutual_info_score(y_true, y_pred_informative)
        homogeneity_informative = homogeneity_score(y_true, y_pred_informative)
        completeness_informative = completeness_score(y_true, y_pred_informative)
        v_measure_informative = v_measure_score(y_true, y_pred_informative)
        fowlkes_mallows_informative = fowlkes_mallows_score(y_true, y_pred_informative)
        
        # Calculate improvements
        silhouette_improvement = silhouette_informative - silhouette_full
        davies_bouldin_improvement = davies_bouldin_full - davies_bouldin_informative  # Lower is better
        calinski_harabasz_improvement = calinski_harabasz_informative - calinski_harabasz_full
        adjusted_rand_improvement = adjusted_rand_informative - adjusted_rand_full
        normalized_mutual_info_improvement = normalized_mutual_info_informative - normalized_mutual_info_full
        homogeneity_improvement = homogeneity_informative - homogeneity_full
        completeness_improvement = completeness_informative - completeness_full
        v_measure_improvement = v_measure_informative - v_measure_full
        fowlkes_mallows_improvement = fowlkes_mallows_informative - fowlkes_mallows_full
        
        print(f"  Full dataset ({X_reduced.shape[1]} features):")
        print(f"    Silhouette Score: {silhouette_full:.4f}")
        print(f"    Davies-Bouldin Score: {davies_bouldin_full:.4f}")
        print(f"    Calinski-Harabasz Score: {calinski_harabasz_full:.2f}")
        print(f"    Adjusted Rand Score: {adjusted_rand_full:.4f}")
        print(f"    Normalized Mutual Info: {normalized_mutual_info_full:.4f}")
        print(f"    V-Measure Score: {v_measure_full:.4f}")
        
        print(f"  Informative indices only ({len(informative_indices)} features):")
        print(f"    Silhouette Score: {silhouette_informative:.4f}")
        print(f"    Davies-Bouldin Score: {davies_bouldin_informative:.4f}")
        print(f"    Calinski-Harabasz Score: {calinski_harabasz_informative:.2f}")
        print(f"    Adjusted Rand Score: {adjusted_rand_informative:.4f}")
        print(f"    Normalized Mutual Info: {normalized_mutual_info_informative:.4f}")
        print(f"    V-Measure Score: {v_measure_informative:.4f}")
        
        print(f"  Improvement:")
        print(f"    Silhouette: {silhouette_improvement:+.4f}")
        print(f"    Davies-Bouldin: {davies_bouldin_improvement:+.4f}")
        print(f"    Calinski-Harabasz: {calinski_harabasz_improvement:+.2f}")
        print(f"    Adjusted Rand: {adjusted_rand_improvement:+.4f}")
        print(f"    Normalized Mutual Info: {normalized_mutual_info_improvement:+.4f}")
        print(f"    V-Measure: {v_measure_improvement:+.4f}")
        
        return {
            'silhouette_score': silhouette_full,
            'davies_bouldin_score': davies_bouldin_full,
            'calinski_harabasz_score': calinski_harabasz_full,
            'adjusted_rand_score': adjusted_rand_full,
            'normalized_mutual_info_score': normalized_mutual_info_full,
            'homogeneity_score': homogeneity_full,
            'completeness_score': completeness_full,
            'v_measure_score': v_measure_full,
            'fowlkes_mallows_score': fowlkes_mallows_full,
            'y_pred': y_pred_full,
            'cluster_centers': kmeans_full.cluster_centers_,
            'informative_indices': informative_indices,
            'informative_silhouette_score': silhouette_informative,
            'informative_davies_bouldin_score': davies_bouldin_informative,
            'informative_calinski_harabasz_score': calinski_harabasz_informative,
            'informative_adjusted_rand_score': adjusted_rand_informative,
            'informative_normalized_mutual_info_score': normalized_mutual_info_informative,
            'informative_homogeneity_score': homogeneity_informative,
            'informative_completeness_score': completeness_informative,
            'informative_v_measure_score': v_measure_informative,
            'informative_fowlkes_mallows_score': fowlkes_mallows_informative,
            'informative_y_pred': y_pred_informative,
            'X_informative': X_informative,
            'silhouette_improvement': silhouette_improvement,
            'davies_bouldin_improvement': davies_bouldin_improvement,
            'calinski_harabasz_improvement': calinski_harabasz_improvement,
            'adjusted_rand_improvement': adjusted_rand_improvement,
            'normalized_mutual_info_improvement': normalized_mutual_info_improvement,
            'homogeneity_improvement': homogeneity_improvement,
            'completeness_improvement': completeness_improvement,
            'v_measure_improvement': v_measure_improvement,
            'fowlkes_mallows_improvement': fowlkes_mallows_improvement
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
        table_data.append(['Class Distribution', f"{dict(zip(y_unique, [np.sum(y_unique == c).values for c in y_unique]))}"])
        
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
        table = ax.table(cellText=table_data, colLabels=['Property', 'Value'], 
                        cellLoc='left', loc='center', bbox=[0, 0, 1, 1])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)  # Fixed height for all cells
        
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
            
        # Handle length mismatches by truncating to the shorter length
        min_length = min(len(y_true), len(y_pred), X_flat.shape[0])
        if len(y_true) != min_length:
            print(f"Warning: Truncating y_true from {len(y_true)} to {min_length}")
            y_true = y_true[:min_length]
        if len(y_pred) != min_length:
            print(f"Warning: Truncating y_pred from {len(y_pred)} to {min_length}")
            y_pred = y_pred[:min_length]
        if X_flat.shape[0] != min_length:
            print(f"Warning: Truncating X_flat from {X_flat.shape[0]} to {min_length}")
            X_flat = X_flat[:min_length]
        
        if min_length == 0:
            print(f"Skipping visualization for {dataset_name}: no valid data after truncation")
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
        
        # Handle single component case
        if X_2d.shape[1] < 2:
            print(f"Skipping 2D visualization for {dataset_name}: only {X_2d.shape[1]} component(s) available")
            return
        
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
        table.scale(1, 1.8)  # Fixed height for all cells
        
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
        plt.show()
        plt.close()
        
        # 3D visualization if we have 3+ components
        if X_flat.shape[1] >= 3:
            # Select informative columns for 3D visualization
            informative_indices_3d, X_3d, _ = self.select_informative_columns(X_flat, y_pred, n_components=3)
            
            # Handle single component case for 3D
            if X_3d.shape[1] < 3:
                print(f"Skipping 3D visualization for {dataset_name}: only {X_3d.shape[1]} component(s) available")
                return
            
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
            table.scale(1, 1.8)
            
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
            plt.show()
            plt.close()
    
    def test_dataset(self, dataset_name, visualize=False):
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
        
        # Get aggregation function info
        agg_func_name = self.optimal_aggregation_functions.get(dataset_name, 'sum')
        
        # Visualize results
        print("\nGenerating visualizations...")
        if visualize:
            # Use informative indices for visualization if available
            informative_silhouette = clustering_results.get('informative_silhouette_score', 0.0)
            informative_davies_bouldin = clustering_results.get('informative_davies_bouldin_score', float('inf'))
            informative_v_measure = clustering_results.get('informative_v_measure_score', 0.0)
            informative_adjusted_rand = clustering_results.get('informative_adjusted_rand_score', 0.0)

            if 'X_informative' in clustering_results and clustering_results['X_informative'] is not None:
                method_label_informative = f"{method_label} Sil: {informative_silhouette:.2f} DB: {informative_davies_bouldin:.2f} V: {informative_v_measure:.2f} AR: {informative_adjusted_rand:.2f}"
                self.visualize_results(clustering_results['X_informative'], y, clustering_results['informative_y_pred'], 
                                    dataset_name, metadata, method_label=method_label_informative)
            else:
                self.visualize_results(X_reduced, y, clustering_results['y_pred'], 
                                    dataset_name, metadata, method_label=method_label)
        
        # Store results with informative indices information
        self.results[dataset_name] = {
            'original_shape': X.shape,
            'reduced_shape': X_reduced.shape,
            'clustering_results': clustering_results,
            'metadata': metadata,
            'aggregation_function': agg_func_name,
            'method_label': method_label,
            'informative_indices': clustering_results.get('informative_indices', []),
            'informative_shape': clustering_results.get('X_informative', X_reduced).shape if 'X_informative' in clustering_results else X_reduced.shape
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
            all_datasets = get_all_datasets()
            print(f"Using fallback dataset list: {len(all_datasets)} datasets")
        
        results = {}
        
        for dataset_name in tqdm.tqdm(all_datasets, desc="Testing datasets"):
            try:
                if dataset_name in ["covertype"]:
                    print("Skipping large datasets")
                    continue
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
        informative_analysis = {}
        
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
            
            # Get informative indices analysis
            clustering_results = result['clustering_results']
            informative_indices = clustering_results.get('informative_indices', [])
            informative_silhouette = clustering_results.get('informative_silhouette_score', 0.0)
            informative_davies_bouldin = clustering_results.get('informative_davies_bouldin_score', float('inf'))
            informative_v_measure = clustering_results.get('informative_v_measure_score', 0.0)
            informative_adjusted_rand = clustering_results.get('informative_adjusted_rand_score', 0.0)
            silhouette_improvement = clustering_results.get('silhouette_improvement', 0.0)
            davies_bouldin_improvement = clustering_results.get('davies_bouldin_improvement', 0.0)
            v_measure_improvement = clustering_results.get('v_measure_improvement', 0.0)
            adjusted_rand_improvement = clustering_results.get('adjusted_rand_improvement', 0.0)
            
            summary_data.append({
                'Dataset': name,
                'Original Shape': f"{result['original_shape'][1]}x{result['original_shape'][2]}",
                'Reduced Shape': f"{result['reduced_shape'][1]}",
                'Informative Features': len(informative_indices),
                'Aggregation Function': result.get('aggregation_function', 'sum'),
                'Full Silhouette': f"{clustering_results['silhouette_score']:.4f}",
                'Full Davies-Bouldin': f"{clustering_results['davies_bouldin_score']:.4f}",
                'Full V-Measure': f"{clustering_results.get('v_measure_score', 0.0):.4f}",
                'Full Adjusted Rand': f"{clustering_results.get('adjusted_rand_score', 0.0):.4f}",
                'Informative Silhouette': f"{informative_silhouette:.4f}",
                'Informative Davies-Bouldin': f"{informative_davies_bouldin:.4f}",
                'Informative V-Measure': f"{informative_v_measure:.4f}",
                'Informative Adjusted Rand': f"{informative_adjusted_rand:.4f}",
                'Silhouette Improvement': f"{silhouette_improvement:+.4f}",
                'Davies-Bouldin Improvement': f"{davies_bouldin_improvement:+.4f}",
                'V-Measure Improvement': f"{v_measure_improvement:+.4f}",
                'Adjusted Rand Improvement': f"{adjusted_rand_improvement:+.4f}",
                'Description': result['metadata']['description'],
                'PBP Components': len(pbp_components) if pbp_components else 'N/A'
            })
            
            # Store informative analysis for detailed report
            informative_analysis[name] = {
                'informative_indices': informative_indices,
                'silhouette_improvement': silhouette_improvement,
                'davies_bouldin_improvement': davies_bouldin_improvement,
                'v_measure_improvement': v_measure_improvement,
                'adjusted_rand_improvement': adjusted_rand_improvement,
                'full_silhouette': clustering_results['silhouette_score'],
                'informative_silhouette': informative_silhouette,
                'full_v_measure': clustering_results.get('v_measure_score', 0.0),
                'informative_v_measure': informative_v_measure,
                'full_adjusted_rand': clustering_results.get('adjusted_rand_score', 0.0),
                'informative_adjusted_rand': informative_adjusted_rand
            }
        
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
        
        # Find best performing datasets (both full and informative)
        best_full_silhouette = max(results.items(), 
                                  key=lambda x: x[1]['clustering_results']['silhouette_score'])
        best_full_davies = min(results.items(), 
                              key=lambda x: x[1]['clustering_results']['davies_bouldin_score'])
        
        best_informative_silhouette = max(results.items(), 
                                        key=lambda x: x[1]['clustering_results'].get('informative_silhouette_score', 0.0))
        best_informative_davies = min(results.items(), 
                                    key=lambda x: x[1]['clustering_results'].get('informative_davies_bouldin_score', float('inf')))
        
        print(f"\n{'='*60}")
        print("BEST PERFORMING DATASETS")
        print(f"{'='*60}")
        print(f"Best Full Silhouette Score: {best_full_silhouette[0]} ({best_full_silhouette[1]['clustering_results']['silhouette_score']:.4f})")
        print(f"Best Full Davies-Bouldin Score: {best_full_davies[0]} ({best_full_davies[1]['clustering_results']['davies_bouldin_score']:.4f})")
        print(f"Best Informative Silhouette Score: {best_informative_silhouette[0]} ({best_informative_silhouette[1]['clustering_results'].get('informative_silhouette_score', 0.0):.4f})")
        print(f"Best Informative Davies-Bouldin Score: {best_informative_davies[0]} ({best_informative_davies[1]['clustering_results'].get('informative_davies_bouldin_score', float('inf')):.4f})")
        
        # Analyze informative indices improvements
        print(f"\n{'='*60}")
        print("INFORMATIVE INDICES ANALYSIS")
        print(f"{'='*60}")
        
        improvements = []
        for name, analysis in informative_analysis.items():
            if (analysis['silhouette_improvement'] > 0 or analysis['davies_bouldin_improvement'] > 0 or 
                analysis['v_measure_improvement'] > 0 or analysis['adjusted_rand_improvement'] > 0):
                improvements.append({
                    'dataset': name,
                    'silhouette_improvement': analysis['silhouette_improvement'],
                    'davies_bouldin_improvement': analysis['davies_bouldin_improvement'],
                    'v_measure_improvement': analysis['v_measure_improvement'],
                    'adjusted_rand_improvement': analysis['adjusted_rand_improvement'],
                    'informative_features': len(analysis['informative_indices'])
                })
        
        if improvements:
            print("Datasets with improvement using informative indices:")
            for imp in sorted(improvements, key=lambda x: x['silhouette_improvement'], reverse=True):
                print(f"  {imp['dataset']}: Sil +{imp['silhouette_improvement']:.4f}, DB +{imp['davies_bouldin_improvement']:.4f}, V +{imp['v_measure_improvement']:.4f}, AR +{imp['adjusted_rand_improvement']:.4f} ({imp['informative_features']} features)")
        else:
            print("No datasets showed improvement with informative indices selection.")
        
        # Calculate average improvements
        avg_silhouette_improvement = np.mean([analysis['silhouette_improvement'] for analysis in informative_analysis.values()])
        avg_davies_bouldin_improvement = np.mean([analysis['davies_bouldin_improvement'] for analysis in informative_analysis.values()])
        avg_v_measure_improvement = np.mean([analysis['v_measure_improvement'] for analysis in informative_analysis.values()])
        avg_adjusted_rand_improvement = np.mean([analysis['adjusted_rand_improvement'] for analysis in informative_analysis.values()])
        
        print(f"\nAverage improvements across all datasets:")
        print(f"  Silhouette Score: {avg_silhouette_improvement:+.4f}")
        print(f"  Davies-Bouldin Score: {avg_davies_bouldin_improvement:+.4f}")
        print(f"  V-Measure Score: {avg_v_measure_improvement:+.4f}")
        print(f"  Adjusted Rand Score: {avg_adjusted_rand_improvement:+.4f}")
        
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

    def generate_biomathematics_summary_report(self, results):
        """Generate a biomathematics-specific summary report."""
        print(f"\n{'='*80}")
        print("BIOMATHEMATICS SUMMARY REPORT - Trends in Biomathematics")
        print("Modeling Health in Ecology, Social Interactions, and Cells")
        print(f"{'='*80}")
        
        if not results:
            print("No biomathematics results available for summary report.")
            return
        
        # Categorize datasets by biomathematics domain
        domain_categories = {
            'Ecology & Environmental Health': ['species_distribution', 'covertype'],
            'Medical & Health': ['breast_cancer', 'diabetes', 'thyroid', 'pima', 'geo_breast_cancer'],
            'Social Interactions & Epidemiology': ['ionosphere'],
            'Advanced Medical': ['metabolights'],
            'Core Biomathematics': ['iris', 'wine', 'digits_sklearn', 'sonar', 'glass', 'vehicle', 'seeds', 'linnerrud']
        }
        
        # Create summary table
        summary_data = []
        domain_performance = {}
        
        for name, result in results.items():
            # Determine domain
            domain = 'Other'
            for domain_name, datasets in domain_categories.items():
                if name in datasets:
                    domain = domain_name
                    break
            
            # Get clustering results and informative indices analysis
            clustering_results = result['clustering_results']
            informative_indices = clustering_results.get('informative_indices', [])
            informative_silhouette = clustering_results.get('informative_silhouette_score', 0.0)
            informative_davies_bouldin = clustering_results.get('informative_davies_bouldin_score', float('inf'))
            silhouette_improvement = clustering_results.get('silhouette_improvement', 0.0)
            davies_bouldin_improvement = clustering_results.get('davies_bouldin_improvement', 0.0)
            
            # Get PBP component names if this is a PBP method
            pbp_components = []
            if "PBP" in result.get('method_label', ''):
                try:
                    reduced_shape = result['reduced_shape'][1]
                    estimated_rows = min(3, reduced_shape)
                    pbp_components = self.get_pbp_component_names(reduced_shape, estimated_rows)
                except Exception as e:
                    print(f"Warning: Could not generate PBP component names for {name}: {e}")
            
            summary_data.append({
                'Dataset': name,
                'Domain': domain,
                'Original Shape': f"{result['original_shape'][1]}x{result['original_shape'][2]}",
                'Reduced Shape': f"{result['reduced_shape'][1]}",
                'Informative Features': len(informative_indices),
                'Aggregation Function': result.get('aggregation_function', 'sum'),
                'Full Silhouette': f"{clustering_results['silhouette_score']:.4f}",
                'Full Davies-Bouldin': f"{clustering_results['davies_bouldin_score']:.4f}",
                'Full V-Measure': f"{clustering_results.get('v_measure_score', 0.0):.4f}",
                'Full Adjusted Rand': f"{clustering_results.get('adjusted_rand_score', 0.0):.4f}",
                'Informative Silhouette': f"{informative_silhouette:.4f}",
                'Informative Davies-Bouldin': f"{informative_davies_bouldin:.4f}",
                # 'Informative V-Measure': f"{informative_v_measure:.4f}",
                # 'Informative Adjusted Rand': f"{informative_adjusted_rand:.4f}",
                'Silhouette Improvement': f"{silhouette_improvement:+.4f}",
                'Davies-Bouldin Improvement': f"{davies_bouldin_improvement:+.4f}",
                # 'V-Measure Improvement': f"{v_measure_improvement:+.4f}",
                # 'Adjusted Rand Improvement': f"{adjusted_rand_improvement:+.4f}",
                'Description': result['metadata']['description'],
                'PBP Components': len(pbp_components) if pbp_components else 'N/A'
            })
            
            # Track domain performance
            if domain not in domain_performance:
                domain_performance[domain] = []
            domain_performance[domain].append(result['clustering_results']['silhouette_score'])
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Save summary to file
        summary_df.to_csv(f"{self.results_dir}/tables/biomathematics_summary_pbp_vector_optimized.csv", index=False)
        print(f"\nBiomathematics summary saved to {self.results_dir}/tables/biomathematics_summary_pbp_vector_optimized.csv")
        
        # Domain-specific analysis
        print(f"\n{'='*60}")
        print("DOMAIN-SPECIFIC PERFORMANCE ANALYSIS")
        print(f"{'='*60}")
        
        for domain, scores in domain_performance.items():
            if scores:
                avg_score = np.mean(scores)
                std_score = np.std(scores)
                print(f"\n{domain}:")
                print(f"  Average Silhouette Score: {avg_score:.4f} ± {std_score:.4f}")
                print(f"  Best Score: {max(scores):.4f}")
                print(f"  Number of Datasets: {len(scores)}")
        
        # Find best performing datasets by domain
        print(f"\n{'='*60}")
        print("BEST PERFORMING DATASETS BY DOMAIN")
        print(f"{'='*60}")
        
        for domain, datasets in domain_categories.items():
            domain_results = {k: v for k, v in results.items() if k in datasets}
            if domain_results:
                best_dataset = max(domain_results.items(), 
                                 key=lambda x: x[1]['clustering_results']['silhouette_score'])
                print(f"\n{domain}:")
                print(f"  Best: {best_dataset[0]} (Silhouette: {best_dataset[1]['clustering_results']['silhouette_score']:.4f})")
        
        # Overall biomathematics statistics
        silhouette_scores = [r['clustering_results']['silhouette_score'] for r in results.values()]
        davies_scores = [r['clustering_results']['davies_bouldin_score'] for r in results.values()]
        
        print(f"\n{'='*60}")
        print("OVERALL BIOMATHEMATICS PERFORMANCE")
        print(f"{'='*60}")
        print(f"Total Datasets Analyzed: {len(results)}")
        print(f"Average Silhouette Score: {np.mean(silhouette_scores):.4f} ± {np.std(silhouette_scores):.4f}")
        print(f"Average Davies-Bouldin Score: {np.mean(davies_scores):.4f} ± {np.std(davies_scores):.4f}")
        print(f"Best Overall Performance: {max(results.items(), key=lambda x: x[1]['clustering_results']['silhouette_score'])[0]}")
        
        # Informative indices analysis for biomathematics
        print(f"\n{'='*60}")
        print("INFORMATIVE INDICES ANALYSIS FOR BIOMATHEMATICS")
        print(f"{'='*60}")
        
        biomathematics_improvements = []
        for name, result in results.items():
            clustering_results = result['clustering_results']
            silhouette_improvement = clustering_results.get('silhouette_improvement', 0.0)
            davies_bouldin_improvement = clustering_results.get('davies_bouldin_improvement', 0.0)
            v_measure_improvement = clustering_results.get('v_measure_improvement', 0.0)
            adjusted_rand_improvement = clustering_results.get('adjusted_rand_improvement', 0.0)
            informative_indices = clustering_results.get('informative_indices', [])
            
            if (silhouette_improvement > 0 or davies_bouldin_improvement > 0 or 
                v_measure_improvement > 0 or adjusted_rand_improvement > 0):
                biomathematics_improvements.append({
                    'dataset': name,
                    'domain': next((domain for domain, datasets in domain_categories.items() if name in datasets), 'Other'),
                    'silhouette_improvement': silhouette_improvement,
                    'davies_bouldin_improvement': davies_bouldin_improvement,
                    'v_measure_improvement': v_measure_improvement,
                    'adjusted_rand_improvement': adjusted_rand_improvement,
                    'informative_features': len(informative_indices)
                })
        
        if biomathematics_improvements:
            print("Biomathematics datasets with improvement using informative indices:")
            for imp in sorted(biomathematics_improvements, key=lambda x: x['silhouette_improvement'], reverse=True):
                print(f"  {imp['dataset']} ({imp['domain']}): Sil +{imp['silhouette_improvement']:.4f}, DB +{imp['davies_bouldin_improvement']:.4f}, V +{imp['v_measure_improvement']:.4f}, AR +{imp['adjusted_rand_improvement']:.4f} ({imp['informative_features']} features)")
        else:
            print("No biomathematics datasets showed improvement with informative indices selection.")
        
        # Calculate average improvements for biomathematics
        avg_silhouette_improvement = np.mean([r['clustering_results'].get('silhouette_improvement', 0.0) for r in results.values()])
        avg_davies_bouldin_improvement = np.mean([r['clustering_results'].get('davies_bouldin_improvement', 0.0) for r in results.values()])
        avg_v_measure_improvement = np.mean([r['clustering_results'].get('v_measure_improvement', 0.0) for r in results.values()])
        avg_adjusted_rand_improvement = np.mean([r['clustering_results'].get('adjusted_rand_improvement', 0.0) for r in results.values()])
        
        print(f"\nAverage improvements across biomathematics datasets:")
        print(f"  Silhouette Score: {avg_silhouette_improvement:+.4f}")
        print(f"  Davies-Bouldin Score: {avg_davies_bouldin_improvement:+.4f}")
        print(f"  V-Measure Score: {avg_v_measure_improvement:+.4f}")
        print(f"  Adjusted Rand Score: {avg_adjusted_rand_improvement:+.4f}")
        
        # Aggregation function usage for biomathematics
        if self.use_optimized_aggregation:
            agg_func_counts = {}
            for result in results.values():
                agg_func = result.get('aggregation_function', 'sum')
                agg_func_counts[agg_func] = agg_func_counts.get(agg_func, 0) + 1
            
            print(f"\nAggregation Function Usage in Biomathematics:")
            for agg_func, count in sorted(agg_func_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {agg_func}: {count} datasets")


def main():
    """Main function to run dataset testing."""
    # Create data directory if it doesn't exist
    data_dir = './data'
    results_dir = './results'
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true", help="Visualize results")
    parser.add_argument("--biomat", action="store_true", help="Target biomathematics datasets only")
    args = parser.parse_args()
    visualize = args.visualize
    biomat = args.biomat
    
    # Set results directory based on biomat flag
    if biomat:
        results_dir = './results/biomat'
        print("🎯 Biomathematics mode enabled - targeting health, ecology, and social interaction datasets")
        print(f"📁 Results will be saved to: {results_dir}")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f"{results_dir}/figures", exist_ok=True)
    os.makedirs(f"{results_dir}/tables", exist_ok=True)
    
    # Initialize tester
    tester = DatasetTester(data_dir, results_dir)
    
    # Test datasets based on mode
    if biomat:
        # Use biomathematics datasets only
        from src.data.dataset_config import get_biomathematics_datasets
        biomat_datasets = get_biomathematics_datasets()
        print(f"🧬 Testing {len(biomat_datasets)} biomathematics datasets...")
        
        results = {}
        for dataset_name in tqdm.tqdm(biomat_datasets, desc="Testing biomathematics datasets"):
            try:
                if dataset_name in ["covertype"]:
                    print("Skipping large datasets")
                    continue

                result = tester.test_dataset(dataset_name, visualize=visualize)
                if result:
                    results[dataset_name] = result
            except Exception as e:
                print(f"Error testing {dataset_name}: {e}")
        
        # Generate biomathematics-specific summary report
        tester.generate_biomathematics_summary_report(results)
    else:
        # Test all datasets
        results = tester.test_all_datasets(visualize=visualize)
    
    print(f"\nTesting completed! Results for {len(results)} datasets.")
    print(f"Check {results_dir} for saved visualizations and summary.")

if __name__ == "__main__":
    main() 