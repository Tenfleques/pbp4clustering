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
import time
import psutil
import threading
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
    
    def timeout_wrapper(self, func, args=(), kwargs={}, timeout_seconds=60):
        """Execute a function with timeout protection."""
        result = [None]
        exception = [None]
        
        def target():
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout_seconds)
        
        if thread.is_alive():
            print(f"  ⚠️  Function exceeded {timeout_seconds} seconds, terminating...")
            return None, f"TIMEOUT after {timeout_seconds}s"
        
        if exception[0] is not None:
            print(f"  ❌ Function failed: {exception[0]}")
            return None, f"ERROR: {exception[0]}"
        
        return result[0], "SUCCESS"
        
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
        """Apply PCA dimensionality reduction with timeout protection."""
        # Flatten matrices for PCA
        X_flat = X.reshape(X.shape[0], -1)
        
        # Check if dataset is too large for reasonable processing
        if X_flat.shape[0] * X_flat.shape[1] > 1000000:  # More than 1M elements
            print(f"  ⚠️  Large dataset detected: {X_flat.shape[0]} samples × {X_flat.shape[1]} features")
            print(f"  ⏱️  Applying timeout protection (60 seconds)")
        
        def pca_function():
            pca = PCA(n_components=min(n_components, X_flat.shape[1]))
            return pca.fit_transform(X_flat)
        
        # Use timeout wrapper
        result, status = self.timeout_wrapper(pca_function, timeout_seconds=60)
        
        if result is None:
            return None, f"PCA ({status})"
        
        print(f"  ✅ PCA completed successfully")
        return result, "PCA"
    
    def apply_tsne(self, X, n_components=3):
        """Apply t-SNE dimensionality reduction with timeout protection."""
        # Flatten matrices for t-SNE
        X_flat = X.reshape(X.shape[0], -1)
        
        # Check if dataset is too large for reasonable processing
        if X_flat.shape[0] * X_flat.shape[1] > 1000000:  # More than 1M elements
            print(f"  ⚠️  Large dataset detected: {X_flat.shape[0]} samples × {X_flat.shape[1]} features")
            print(f"  ⏱️  Applying timeout protection (60 seconds)")
        
        def tsne_function():
            tsne = TSNE(n_components=min(n_components, X_flat.shape[1]), random_state=42)
            return tsne.fit_transform(X_flat)
        
        # Use timeout wrapper
        result, status = self.timeout_wrapper(tsne_function, timeout_seconds=60)
        
        if result is None:
            return None, f"t-SNE ({status})"
        
        print(f"  ✅ t-SNE completed successfully")
        return result, "t-SNE"
    
    def apply_umap(self, X, n_components=3):
        """Apply UMAP dimensionality reduction with timeout protection."""
        # Flatten matrices for UMAP
        X_flat = X.reshape(X.shape[0], -1)
        
        # Check if dataset is too large for reasonable processing
        if X_flat.shape[0] * X_flat.shape[1] > 1000000:  # More than 1M elements
            print(f"  ⚠️  Large dataset detected: {X_flat.shape[0]} samples × {X_flat.shape[1]} features")
            print(f"  ⏱️  Applying timeout protection (60 seconds)")
        
        def umap_function():
            reducer = umap.UMAP(n_components=min(n_components, X_flat.shape[1]), random_state=42)
            return reducer.fit_transform(X_flat)
        
        # Use timeout wrapper
        result, status = self.timeout_wrapper(umap_function, timeout_seconds=60)
        
        if result is None:
            return None, f"UMAP ({status})"
        
        print(f"  ✅ UMAP completed successfully")
        return result, "UMAP"
    
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
        """Evaluate clustering performance with comprehensive metrics and informative indices analysis."""
        if len(np.unique(y_true)) <= 1:
            return {
                'silhouette_score': 0.0,
                'davies_bouldin_score': float('inf'),
                'calinski_harabasz_score': 0.0,
                'adjusted_rand_score': 0.0,
                'normalized_mutual_info_score': 0.0,
                'homogeneity_score': 0.0,
                'completeness_score': 0.0,
                'v_measure_score': 0.0,
                'fowlkes_mallows_score': 0.0,
                'method': method_name,
                'n_clusters': 1,
                'processing_time': 0.0,
                'memory_usage_mb': 0.0,
                'informative_indices': [],
                'informative_silhouette_score': 0.0,
                'informative_davies_bouldin_score': float('inf'),
                'informative_calinski_harabasz_score': 0.0,
                'informative_adjusted_rand_score': 0.0,
                'informative_normalized_mutual_info_score': 0.0,
                'informative_homogeneity_score': 0.0,
                'informative_completeness_score': 0.0,
                'informative_v_measure_score': 0.0,
                'informative_fowlkes_mallows_score': 0.0,
                'silhouette_improvement': 0.0,
                'davies_bouldin_improvement': 0.0,
                'calinski_harabasz_improvement': 0.0,
                'adjusted_rand_improvement': 0.0,
                'normalized_mutual_info_improvement': 0.0,
                'homogeneity_improvement': 0.0,
                'completeness_improvement': 0.0,
                'v_measure_improvement': 0.0,
                'fowlkes_mallows_improvement': 0.0
            }
        
        import time
        import psutil
        import os
        from sklearn.metrics import (
            silhouette_score, davies_bouldin_score, calinski_harabasz_score,
            adjusted_rand_score, normalized_mutual_info_score,
            homogeneity_score, completeness_score, v_measure_score,
            fowlkes_mallows_score
        )
        
        # Start timing and memory tracking
        start_time = time.time()
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Check timeout (1 minute = 60 seconds)
        timeout_seconds = 60
        
        try:
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=len(np.unique(y_true)), random_state=42, n_init=10)
            y_pred = kmeans.fit_predict(X_reduced)
            
            # Check timeout during metric calculation
            if time.time() - start_time > timeout_seconds:
                return {
                    'silhouette_score': 0.0,
                    'davies_bouldin_score': float('inf'),
                    'calinski_harabasz_score': 0.0,
                    'adjusted_rand_score': 0.0,
                    'normalized_mutual_info_score': 0.0,
                    'homogeneity_score': 0.0,
                    'completeness_score': 0.0,
                    'v_measure_score': 0.0,
                    'fowlkes_mallows_score': 0.0,
                    'method': method_name,
                    'n_clusters': len(np.unique(y_true)),
                    'processing_time': time.time() - start_time,
                    'memory_usage_mb': process.memory_info().rss / 1024 / 1024 - initial_memory,
                    'error': 'Runtime exceeds 1 minute',
                    'informative_indices': [],
                    'informative_silhouette_score': 0.0,
                    'informative_davies_bouldin_score': float('inf'),
                    'informative_calinski_harabasz_score': 0.0,
                    'informative_adjusted_rand_score': 0.0,
                    'informative_normalized_mutual_info_score': 0.0,
                    'informative_homogeneity_score': 0.0,
                    'informative_completeness_score': 0.0,
                    'informative_v_measure_score': 0.0,
                    'informative_fowlkes_mallows_score': 0.0,
                    'silhouette_improvement': 0.0,
                    'davies_bouldin_improvement': 0.0,
                    'calinski_harabasz_improvement': 0.0,
                    'adjusted_rand_improvement': 0.0,
                    'normalized_mutual_info_improvement': 0.0,
                    'homogeneity_improvement': 0.0,
                    'completeness_improvement': 0.0,
                    'v_measure_improvement': 0.0,
                    'fowlkes_mallows_improvement': 0.0
                }
            
            # Calculate comprehensive metrics for full dataset
            silhouette_full = silhouette_score(X_reduced, y_pred)
            davies_bouldin_full = davies_bouldin_score(X_reduced, y_pred)
            calinski_harabasz_full = calinski_harabasz_score(X_reduced, y_pred)
            
            # Classification metrics (comparing clustering to true labels)
            adjusted_rand_full = adjusted_rand_score(y_true, y_pred)
            normalized_mutual_info_full = normalized_mutual_info_score(y_true, y_pred)
            homogeneity_full = homogeneity_score(y_true, y_pred)
            completeness_full = completeness_score(y_true, y_pred)
            v_measure_full = v_measure_score(y_true, y_pred)
            fowlkes_mallows_full = fowlkes_mallows_score(y_true, y_pred)
            
            # Check timeout before informative indices analysis
            if time.time() - start_time > timeout_seconds:
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
                    'method': method_name,
                    'n_clusters': len(np.unique(y_true)),
                    'processing_time': time.time() - start_time,
                    'memory_usage_mb': process.memory_info().rss / 1024 / 1024 - initial_memory,
                    'error': 'Runtime exceeds 1 minute during informative indices analysis',
                    'informative_indices': [],
                    'informative_silhouette_score': 0.0,
                    'informative_davies_bouldin_score': float('inf'),
                    'informative_calinski_harabasz_score': 0.0,
                    'informative_adjusted_rand_score': 0.0,
                    'informative_normalized_mutual_info_score': 0.0,
                    'informative_homogeneity_score': 0.0,
                    'informative_completeness_score': 0.0,
                    'informative_v_measure_score': 0.0,
                    'informative_fowlkes_mallows_score': 0.0,
                    'silhouette_improvement': 0.0,
                    'davies_bouldin_improvement': 0.0,
                    'calinski_harabasz_improvement': 0.0,
                    'adjusted_rand_improvement': 0.0,
                    'normalized_mutual_info_improvement': 0.0,
                    'homogeneity_improvement': 0.0,
                    'completeness_improvement': 0.0,
                    'v_measure_improvement': 0.0,
                    'fowlkes_mallows_improvement': 0.0
                }
            
            # Now, select informative columns and evaluate clustering on those only
            informative_indices, X_informative, _ = self.select_informative_columns(
                X_reduced, y_pred, n_components=min(3, X_reduced.shape[1])
            )
            
            # Check timeout before informative indices clustering
            if time.time() - start_time > timeout_seconds:
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
                    'method': method_name,
                    'n_clusters': len(np.unique(y_true)),
                    'processing_time': time.time() - start_time,
                    'memory_usage_mb': process.memory_info().rss / 1024 / 1024 - initial_memory,
                    'error': 'Runtime exceeds 1 minute during informative indices clustering',
                    'informative_indices': informative_indices,
                    'informative_silhouette_score': 0.0,
                    'informative_davies_bouldin_score': float('inf'),
                    'informative_calinski_harabasz_score': 0.0,
                    'informative_adjusted_rand_score': 0.0,
                    'informative_normalized_mutual_info_score': 0.0,
                    'informative_homogeneity_score': 0.0,
                    'informative_completeness_score': 0.0,
                    'informative_v_measure_score': 0.0,
                    'informative_fowlkes_mallows_score': 0.0,
                    'silhouette_improvement': 0.0,
                    'davies_bouldin_improvement': 0.0,
                    'calinski_harabasz_improvement': 0.0,
                    'adjusted_rand_improvement': 0.0,
                    'normalized_mutual_info_improvement': 0.0,
                    'homogeneity_improvement': 0.0,
                    'completeness_improvement': 0.0,
                    'v_measure_improvement': 0.0,
                    'fowlkes_mallows_improvement': 0.0
                }
            
            # Evaluate clustering on informative indices only
            kmeans_informative = KMeans(n_clusters=len(np.unique(y_true)), random_state=42, n_init=10)
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
            davies_bouldin_improvement = davies_bouldin_full - davies_bouldin_informative
            calinski_harabasz_improvement = calinski_harabasz_informative - calinski_harabasz_full
            adjusted_rand_improvement = adjusted_rand_informative - adjusted_rand_full
            normalized_mutual_info_improvement = normalized_mutual_info_informative - normalized_mutual_info_full
            homogeneity_improvement = homogeneity_informative - homogeneity_full
            completeness_improvement = completeness_informative - completeness_full
            v_measure_improvement = v_measure_informative - v_measure_full
            fowlkes_mallows_improvement = fowlkes_mallows_informative - fowlkes_mallows_full
            
            # Calculate final timing and memory
            final_time = time.time() - start_time
            final_memory = process.memory_info().rss / 1024 / 1024 - initial_memory
            
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
                'method': method_name,
                'n_clusters': len(np.unique(y_true)),
                'processing_time': final_time,
                'memory_usage_mb': final_memory,
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
            
        except Exception as e:
            final_time = time.time() - start_time
            final_memory = process.memory_info().rss / 1024 / 1024 - initial_memory
            
            return {
                'silhouette_score': 0.0,
                'davies_bouldin_score': float('inf'),
                'calinski_harabasz_score': 0.0,
                'adjusted_rand_score': 0.0,
                'normalized_mutual_info_score': 0.0,
                'homogeneity_score': 0.0,
                'completeness_score': 0.0,
                'v_measure_score': 0.0,
                'fowlkes_mallows_score': 0.0,
                'method': method_name,
                'n_clusters': len(np.unique(y_true)),
                'processing_time': final_time,
                'memory_usage_mb': final_memory,
                'error': str(e),
                'informative_indices': [],
                'informative_silhouette_score': 0.0,
                'informative_davies_bouldin_score': float('inf'),
                'informative_calinski_harabasz_score': 0.0,
                'informative_adjusted_rand_score': 0.0,
                'informative_normalized_mutual_info_score': 0.0,
                'informative_homogeneity_score': 0.0,
                'informative_completeness_score': 0.0,
                'informative_v_measure_score': 0.0,
                'informative_fowlkes_mallows_score': 0.0,
                'silhouette_improvement': 0.0,
                'davies_bouldin_improvement': 0.0,
                'calinski_harabasz_improvement': 0.0,
                'adjusted_rand_improvement': 0.0,
                'normalized_mutual_info_improvement': 0.0,
                'homogeneity_improvement': 0.0,
                'completeness_improvement': 0.0,
                'v_measure_improvement': 0.0,
                'fowlkes_mallows_improvement': 0.0
            }
    
    def select_informative_columns(self, X_reduced, y_labels, n_components=2):
        """Select the most informative columns for visualization."""
        if X_reduced.shape[1] <= n_components:
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
                
                # Check if dimensionality reduction failed or timed out
                if X_reduced is None:
                    print(f"  ❌ {method_name} failed or timed out")
                    results[method_name] = {
                        'silhouette_score': 0,
                        'davies_bouldin_score': float('inf'),
                        'calinski_harabasz_score': 0.0,
                        'adjusted_rand_score': 0.0,
                        'normalized_mutual_info_score': 0.0,
                        'homogeneity_score': 0.0,
                        'completeness_score': 0.0,
                        'v_measure_score': 0.0,
                        'fowlkes_mallows_score': 0.0,
                        'method': method_label,
                        'processing_time': 0.0,
                        'memory_usage_mb': 0.0,
                        'error': f'{method_name} failed or timed out',
                        'informative_indices': [],
                        'informative_silhouette_score': 0.0,
                        'informative_davies_bouldin_score': float('inf'),
                        'informative_calinski_harabasz_score': 0.0,
                        'informative_adjusted_rand_score': 0.0,
                        'informative_normalized_mutual_info_score': 0.0,
                        'informative_homogeneity_score': 0.0,
                        'informative_completeness_score': 0.0,
                        'informative_v_measure_score': 0.0,
                        'informative_fowlkes_mallows_score': 0.0,
                        'silhouette_improvement': 0.0,
                        'davies_bouldin_improvement': 0.0,
                        'calinski_harabasz_improvement': 0.0,
                        'adjusted_rand_improvement': 0.0,
                        'normalized_mutual_info_improvement': 0.0,
                        'homogeneity_improvement': 0.0,
                        'completeness_improvement': 0.0,
                        'v_measure_improvement': 0.0,
                        'fowlkes_mallows_improvement': 0.0
                    }
                    continue
                
                # Evaluate clustering with comprehensive metrics
                evaluation = self.evaluate_clustering(X_reduced, y, method_label)
                results[method_name] = evaluation
                
                # Check for timeout or error
                if 'error' in evaluation:
                    print(f"  ⚠️  {evaluation['error']}")
                    print(f"  Processing Time: {evaluation['processing_time']:.2f}s")
                    print(f"  Memory Usage: {evaluation['memory_usage_mb']:.2f} MB")
                else:
                    print(f"  ✅ Silhouette Score: {evaluation['silhouette_score']:.4f}")
                    print(f"  ✅ Davies-Bouldin Score: {evaluation['davies_bouldin_score']:.4f}")
                    print(f"  ✅ V-Measure Score: {evaluation['v_measure_score']:.4f}")
                    print(f"  ✅ Adjusted Rand Score: {evaluation['adjusted_rand_score']:.4f}")
                    print(f"  📊 Processing Time: {evaluation['processing_time']:.2f}s")
                    print(f"  💾 Memory Usage: {evaluation['memory_usage_mb']:.2f} MB")
                    
                    # Show informative indices improvements if available
                    if 'informative_silhouette_score' in evaluation:
                        print(f"  🔍 Informative Indices Analysis:")
                        print(f"    - Informative Silhouette: {evaluation['informative_silhouette_score']:.4f}")
                        print(f"    - Silhouette Improvement: {evaluation['silhouette_improvement']:+.4f}")
                        print(f"    - V-Measure Improvement: {evaluation['v_measure_improvement']:+.4f}")
                        print(f"    - Informative Features: {len(evaluation['informative_indices'])}")
                
            except Exception as e:
                print(f"  ❌ Error applying {method_name}: {e}")
                results[method_name] = {
                    'silhouette_score': 0,
                    'davies_bouldin_score': float('inf'),
                    'calinski_harabasz_score': 0.0,
                    'adjusted_rand_score': 0.0,
                    'normalized_mutual_info_score': 0.0,
                    'homogeneity_score': 0.0,
                    'completeness_score': 0.0,
                    'v_measure_score': 0.0,
                    'fowlkes_mallows_score': 0.0,
                    'method': method_name,
                    'processing_time': 0.0,
                    'memory_usage_mb': 0.0,
                    'error': str(e),
                    'informative_indices': [],
                    'informative_silhouette_score': 0.0,
                    'informative_davies_bouldin_score': float('inf'),
                    'informative_calinski_harabasz_score': 0.0,
                    'informative_adjusted_rand_score': 0.0,
                    'informative_normalized_mutual_info_score': 0.0,
                    'informative_homogeneity_score': 0.0,
                    'informative_completeness_score': 0.0,
                    'informative_v_measure_score': 0.0,
                    'informative_fowlkes_mallows_score': 0.0,
                    'silhouette_improvement': 0.0,
                    'davies_bouldin_improvement': 0.0,
                    'calinski_harabasz_improvement': 0.0,
                    'adjusted_rand_improvement': 0.0,
                    'normalized_mutual_info_improvement': 0.0,
                    'homogeneity_improvement': 0.0,
                    'completeness_improvement': 0.0,
                    'v_measure_improvement': 0.0,
                    'fowlkes_mallows_improvement': 0.0
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
                if dataset_name in ["covertype"]:
                    print("Skipping large datasets")
                    continue
                results = self.compare_methods(dataset_name)
                if results:
                    all_results[dataset_name] = results
            except Exception as e:
                print(f"Error comparing {dataset_name}: {e}")
        
        # Generate comprehensive summary
        self.generate_comprehensive_summary(all_results)
        
        return all_results
    
    def run_biomathematics_comparison(self):
        """Run comprehensive comparison across biomathematics datasets only."""
        # Get biomathematics datasets
        try:
            from ..data.dataset_config import get_biomathematics_datasets
            biomat_datasets = get_biomathematics_datasets()
            print(f"🧬 Found {len(biomat_datasets)} biomathematics datasets")
            print("Running biomathematics comparison...")
        except Exception as e:
            print(f"Error getting biomathematics datasets: {e}")
            return {}
        
        all_results = {}
        
        for dataset_name in tqdm(biomat_datasets, desc="Comparing biomathematics datasets"):
            try:
                if dataset_name in ["covertype"]:
                    print("Skipping large datasets")
                    continue
                results = self.compare_methods(dataset_name)
                if results:
                    all_results[dataset_name] = results
            except Exception as e:
                print(f"Error comparing {dataset_name}: {e}")
        
        # Generate biomathematics-specific summary
        self.generate_biomathematics_comprehensive_summary(all_results)
        
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
                    'Calinski_Harabasz_Score': result.get('calinski_harabasz_score', 0.0),
                    'V_Measure_Score': result.get('v_measure_score', 0.0),
                    'Adjusted_Rand_Score': result.get('adjusted_rand_score', 0.0),
                    'Normalized_Mutual_Info_Score': result.get('normalized_mutual_info_score', 0.0),
                    'Homogeneity_Score': result.get('homogeneity_score', 0.0),
                    'Completeness_Score': result.get('completeness_score', 0.0),
                    'Fowlkes_Mallows_Score': result.get('fowlkes_mallows_score', 0.0),
                    'Processing_Time_s': result.get('processing_time', 0.0),
                    'Memory_Usage_MB': result.get('memory_usage_mb', 0.0),
                    'Informative_Features': len(result.get('informative_indices', [])),
                    'Informative_Silhouette_Score': result.get('informative_silhouette_score', 0.0),
                    'Informative_V_Measure_Score': result.get('informative_v_measure_score', 0.0),
                    'Silhouette_Improvement': result.get('silhouette_improvement', 0.0),
                    'V_Measure_Improvement': result.get('v_measure_improvement', 0.0),
                    'N_Clusters': result.get('n_clusters', 0),
                    'Error': result.get('error', '')
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary to CSV
        output_file = f"{self.results_dir}/tables/comprehensive_comparison_summary_optimized.csv"
        summary_df.to_csv(output_file, index=False)
        print(f"Summary saved to: {output_file}")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print("-" * 50)
        
        # Method rankings by different metrics
        print("\nMethod Rankings by Different Metrics:")
        print("-" * 50)
        
        # Silhouette Score rankings
        method_rankings_silhouette = summary_df.groupby('Method')['Silhouette_Score'].agg(['mean', 'std']).sort_values('mean', ascending=False)
        print("\nSilhouette Score Rankings:")
        for method, stats in method_rankings_silhouette.iterrows():
            print(f"  {method}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        # V-Measure Score rankings
        method_rankings_vmeasure = summary_df.groupby('Method')['V_Measure_Score'].agg(['mean', 'std']).sort_values('mean', ascending=False)
        print("\nV-Measure Score Rankings:")
        for method, stats in method_rankings_vmeasure.iterrows():
            print(f"  {method}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        # Processing Time rankings (lower is better)
        method_rankings_time = summary_df.groupby('Method')['Processing_Time_s'].agg(['mean', 'std']).sort_values('mean', ascending=True)
        print("\nProcessing Time Rankings (seconds, lower is better):")
        for method, stats in method_rankings_time.iterrows():
            print(f"  {method}: {stats['mean']:.2f} ± {stats['std']:.2f}")
        
        # Memory Usage rankings (lower is better)
        method_rankings_memory = summary_df.groupby('Method')['Memory_Usage_MB'].agg(['mean', 'std']).sort_values('mean', ascending=True)
        print("\nMemory Usage Rankings (MB, lower is better):")
        for method, stats in method_rankings_memory.iterrows():
            print(f"  {method}: {stats['mean']:.2f} ± {stats['std']:.2f}")
        
        # PBP performance analysis
        if self.use_optimized_aggregation:
            pbp_results = summary_df[summary_df['Method'].str.contains('PBP')]
            if not pbp_results.empty:
                print(f"\nPBP Performance Analysis:")
                print(f"  Average PBP Silhouette Score: {pbp_results['Silhouette_Score'].mean():.4f} ± {pbp_results['Silhouette_Score'].std():.4f}")
                print(f"  Average PBP V-Measure Score: {pbp_results['V_Measure_Score'].mean():.4f} ± {pbp_results['V_Measure_Score'].std():.4f}")
                print(f"  Average PBP Processing Time: {pbp_results['Processing_Time_s'].mean():.2f}s ± {pbp_results['Processing_Time_s'].std():.2f}s")
                print(f"  Average PBP Memory Usage: {pbp_results['Memory_Usage_MB'].mean():.2f} MB ± {pbp_results['Memory_Usage_MB'].std():.2f} MB")
                print(f"  Best PBP Silhouette Score: {pbp_results['Silhouette_Score'].max():.4f}")
                print(f"  PBP wins (Silhouette): {len(pbp_results[pbp_results['Silhouette_Score'] == pbp_results.groupby('Dataset')['Silhouette_Score'].transform('max')])} out of {len(pbp_results)} comparisons")
                print(f"  PBP wins (V-Measure): {len(pbp_results[pbp_results['V_Measure_Score'] == pbp_results.groupby('Dataset')['V_Measure_Score'].transform('max')])} out of {len(pbp_results)} comparisons")
        
        else:
            print("\nNo aggregation optimization data available.")
        
        # Informative indices analysis
        print(f"\nInformative Indices Analysis:")
        informative_improvements = summary_df[summary_df['Silhouette_Improvement'] > 0]
        if not informative_improvements.empty:
            print(f"  Datasets with Silhouette improvement: {len(informative_improvements)} out of {len(summary_df)}")
            print(f"  Average Silhouette improvement: {informative_improvements['Silhouette_Improvement'].mean():.4f}")
            print(f"  Average V-Measure improvement: {informative_improvements['V_Measure_Improvement'].mean():.4f}")
        else:
            print("  No datasets showed improvement with informative indices selection.")
        
        # Best performing dataset for each method
        print(f"\nBest performing dataset for each method:")
        for method in summary_df['Method'].unique():
            method_data = summary_df[summary_df['Method'] == method]
            best_dataset = method_data.loc[method_data['Silhouette_Score'].idxmax()]
            print(f"  {method}: {best_dataset['Dataset']} (Silhouette: {best_dataset['Silhouette_Score']:.4f}, V-Measure: {best_dataset['V_Measure_Score']:.4f}, Time: {best_dataset['Processing_Time_s']:.2f}s)")

    def generate_biomathematics_comprehensive_summary(self, all_results):
        """Generate a biomathematics-specific comprehensive summary."""
        print(f"\n{'='*80}")
        print("BIOMATHEMATICS COMPREHENSIVE COMPARISON SUMMARY")
        print("Trends in Biomathematics: Modeling Health in Ecology, Social Interactions, and Cells")
        print(f"{'='*80}")
        
        if not all_results:
            print("No biomathematics results available for summary.")
            return
        
        # Categorize datasets by biomathematics domain
        domain_categories = {
            'Ecology & Environmental Health': ['species_distribution'],
            'Medical & Health': ['breast_cancer', 'diabetes', 'thyroid', 'pima', 'geo_breast_cancer'],
            'Social Interactions & Epidemiology': ['ionosphere'],
            'Advanced Medical': ['metabolights'],
            'Core Biomathematics': ['iris', 'wine', 'digits_sklearn', 'sonar', 'glass', 'vehicle', 'seeds', 'linnerrud']
        }
        
        # Create summary DataFrame
        summary_data = []
        domain_performance = {}
        
        for dataset_name, results in all_results.items():
            # Determine domain
            domain = 'Other'
            for domain_name, datasets in domain_categories.items():
                if dataset_name in datasets:
                    domain = domain_name
                    break
            
            for method_name, result in results.items():
                summary_data.append({
                    'Dataset': dataset_name,
                    'Domain': domain,
                    'Method': result['method'],
                    'Silhouette_Score': result['silhouette_score'],
                    'Davies_Bouldin_Score': result['davies_bouldin_score'],
                    'Calinski_Harabasz_Score': result.get('calinski_harabasz_score', 0.0),
                    'V_Measure_Score': result.get('v_measure_score', 0.0),
                    'Adjusted_Rand_Score': result.get('adjusted_rand_score', 0.0),
                    'Normalized_Mutual_Info_Score': result.get('normalized_mutual_info_score', 0.0),
                    'Homogeneity_Score': result.get('homogeneity_score', 0.0),
                    'Completeness_Score': result.get('completeness_score', 0.0),
                    'Fowlkes_Mallows_Score': result.get('fowlkes_mallows_score', 0.0),
                    'Processing_Time_s': result.get('processing_time', 0.0),
                    'Memory_Usage_MB': result.get('memory_usage_mb', 0.0),
                    'Informative_Features': len(result.get('informative_indices', [])),
                    'Informative_Silhouette_Score': result.get('informative_silhouette_score', 0.0),
                    'Informative_V_Measure_Score': result.get('informative_v_measure_score', 0.0),
                    'Silhouette_Improvement': result.get('silhouette_improvement', 0.0),
                    'V_Measure_Improvement': result.get('v_measure_improvement', 0.0),
                    'N_Clusters': result.get('n_clusters', 0),
                    'Error': result.get('error', '')
                })
                
                # Track domain performance
                if domain not in domain_performance:
                    domain_performance[domain] = []
                domain_performance[domain].append(result['silhouette_score'])
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary to CSV
        output_file = f"{self.results_dir}/tables/biomathematics_comprehensive_comparison_summary_optimized.csv"
        summary_df.to_csv(output_file, index=False)
        print(f"Biomathematics summary saved to: {output_file}")
        
        # Print summary statistics
        print("\nBiomathematics Summary Statistics:")
        print("-" * 50)
        
        # Method rankings for biomathematics by different metrics
        print("\nMethod Rankings for Biomathematics by Different Metrics:")
        print("-" * 50)
        
        # Silhouette Score rankings
        method_rankings_silhouette = summary_df.groupby('Method')['Silhouette_Score'].agg(['mean', 'std']).sort_values('mean', ascending=False)
        print("\nSilhouette Score Rankings:")
        for method, stats in method_rankings_silhouette.iterrows():
            print(f"  {method}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        # V-Measure Score rankings
        method_rankings_vmeasure = summary_df.groupby('Method')['V_Measure_Score'].agg(['mean', 'std']).sort_values('mean', ascending=False)
        print("\nV-Measure Score Rankings:")
        for method, stats in method_rankings_vmeasure.iterrows():
            print(f"  {method}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        # Processing Time rankings (lower is better)
        method_rankings_time = summary_df.groupby('Method')['Processing_Time_s'].agg(['mean', 'std']).sort_values('mean', ascending=True)
        print("\nProcessing Time Rankings (seconds, lower is better):")
        for method, stats in method_rankings_time.iterrows():
            print(f"  {method}: {stats['mean']:.2f} ± {stats['std']:.2f}")
        
        # Memory Usage rankings (lower is better)
        method_rankings_memory = summary_df.groupby('Method')['Memory_Usage_MB'].agg(['mean', 'std']).sort_values('mean', ascending=True)
        print("\nMemory Usage Rankings (MB, lower is better):")
        for method, stats in method_rankings_memory.iterrows():
            print(f"  {method}: {stats['mean']:.2f} ± {stats['std']:.2f}")
        
        # Domain-specific analysis
        print(f"\n{'='*60}")
        print("DOMAIN-SPECIFIC PERFORMANCE IN BIOMATHEMATICS")
        print(f"{'='*60}")
        
        for domain, scores in domain_performance.items():
            if scores:
                avg_score = np.mean(scores)
                std_score = np.std(scores)
                print(f"\n{domain}:")
                print(f"  Average Silhouette Score: {avg_score:.4f} ± {std_score:.4f}")
                print(f"  Best Score: {max(scores):.4f}")
                print(f"  Number of Results: {len(scores)}")
        
        # PBP performance analysis for biomathematics
        if self.use_optimized_aggregation:
            pbp_results = summary_df[summary_df['Method'].str.contains('PBP')]
            if not pbp_results.empty:
                print(f"\nPBP Performance in Biomathematics:")
                print(f"  Average PBP Silhouette Score: {pbp_results['Silhouette_Score'].mean():.4f} ± {pbp_results['Silhouette_Score'].std():.4f}")
                print(f"  Average PBP V-Measure Score: {pbp_results['V_Measure_Score'].mean():.4f} ± {pbp_results['V_Measure_Score'].std():.4f}")
                print(f"  Average PBP Processing Time: {pbp_results['Processing_Time_s'].mean():.2f}s ± {pbp_results['Processing_Time_s'].std():.2f}s")
                print(f"  Average PBP Memory Usage: {pbp_results['Memory_Usage_MB'].mean():.2f} MB ± {pbp_results['Memory_Usage_MB'].std():.2f} MB")
                print(f"  Best PBP Silhouette Score: {pbp_results['Silhouette_Score'].max():.4f}")
                print(f"  PBP wins (Silhouette): {len(pbp_results[pbp_results['Silhouette_Score'] == pbp_results.groupby('Dataset')['Silhouette_Score'].transform('max')])} out of {len(pbp_results)} comparisons")
                print(f"  PBP wins (V-Measure): {len(pbp_results[pbp_results['V_Measure_Score'] == pbp_results.groupby('Dataset')['V_Measure_Score'].transform('max')])} out of {len(pbp_results)} comparisons")
        
        # Informative indices analysis for biomathematics
        print(f"\nInformative Indices Analysis for Biomathematics:")
        informative_improvements = summary_df[summary_df['Silhouette_Improvement'] > 0]
        if not informative_improvements.empty:
            print(f"  Datasets with Silhouette improvement: {len(informative_improvements)} out of {len(summary_df)}")
            print(f"  Average Silhouette improvement: {informative_improvements['Silhouette_Improvement'].mean():.4f}")
            print(f"  Average V-Measure improvement: {informative_improvements['V_Measure_Improvement'].mean():.4f}")
        else:
            print("  No datasets showed improvement with informative indices selection.")
        
        # Best performing dataset for each method in biomathematics
        print(f"\nBest performing biomathematics dataset for each method:")
        for method in summary_df['Method'].unique():
            method_data = summary_df[summary_df['Method'] == method]
            best_dataset = method_data.loc[method_data['Silhouette_Score'].idxmax()]
            print(f"  {method}: {best_dataset['Dataset']} (Silhouette: {best_dataset['Silhouette_Score']:.4f}, V-Measure: {best_dataset['V_Measure_Score']:.4f}, Time: {best_dataset['Processing_Time_s']:.2f}s)")
        
        # Overall biomathematics statistics
        print(f"\n{'='*60}")
        print("OVERALL BIOMATHEMATICS COMPARISON STATISTICS")
        print(f"{'='*60}")
        print(f"Total Datasets Analyzed: {len(all_results)}")
        print(f"Total Method Comparisons: {len(summary_df)}")
        print(f"Average Silhouette Score: {summary_df['Silhouette_Score'].mean():.4f} ± {summary_df['Silhouette_Score'].std():.4f}")
        print(f"Average V-Measure Score: {summary_df['V_Measure_Score'].mean():.4f} ± {summary_df['V_Measure_Score'].std():.4f}")
        print(f"Average Processing Time: {summary_df['Processing_Time_s'].mean():.2f}s ± {summary_df['Processing_Time_s'].std():.2f}s")
        print(f"Average Memory Usage: {summary_df['Memory_Usage_MB'].mean():.2f} MB ± {summary_df['Memory_Usage_MB'].std():.2f} MB")
        
        # Find best overall performance
        best_overall = summary_df.loc[summary_df['Silhouette_Score'].idxmax()]
        print(f"Best Overall Performance: {best_overall['Method']} on {best_overall['Dataset']} (Silhouette: {best_overall['Silhouette_Score']:.4f}, V-Measure: {best_overall['V_Measure_Score']:.4f}, Time: {best_overall['Processing_Time_s']:.2f}s)")


def main():
    """Main function to run comprehensive comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive comparison of dimensionality reduction methods")
    parser.add_argument("--no-optimization", action="store_true", help="Disable aggregation function optimization")
    parser.add_argument("--biomat", action="store_true", help="Target biomathematics datasets only")
    args = parser.parse_args()
    
    use_optimization = not args.no_optimization
    biomat = args.biomat
    
    # Set results directory based on biomat flag
    if biomat:
        results_dir = './results/biomat'
        print("🎯 Biomathematics mode enabled - targeting health, ecology, and social interaction datasets")
        print(f"📁 Results will be saved to: {results_dir}")
    else:
        results_dir = './results'
    
    comparison = ComprehensiveComparison('./data', results_dir, use_optimized_aggregation=use_optimization)
    
    # Run appropriate comparison based on mode
    if biomat:
        results = comparison.run_biomathematics_comparison()
    else:
        results = comparison.run_comprehensive_comparison()
    
    if results:
        print(f"\n✅ Comprehensive comparison completed successfully!")
        if biomat:
            print(f"🧬 Biomathematics results for {len(results)} datasets generated.")
        else:
            print(f"Results for {len(results)} datasets generated.")
    else:
        print(f"\n❌ Comprehensive comparison failed or no results generated.")


if __name__ == "__main__":
    main() 