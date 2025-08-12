#!/usr/bin/env python3
"""
Sorting Functions for Pseudo-Boolean Polynomial Dimensionality Reduction

This module provides various sorting functions that can be used in PBP calculations
to determine the order of data points in the polynomial basis representation.
"""

import numpy as np
import pandas as pd
from typing import Callable, Any, Dict, List
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


# Basic Sorting Functions
def ascending_sort(C: np.ndarray) -> np.ndarray:
    """Default ascending sort - smallest values first (original behavior)."""
    return C.argsort(kind='quick', axis=0)


def descending_sort(C: np.ndarray) -> np.ndarray:
    """Descending sort - largest values first."""
    return C.argsort(kind='quick', axis=0)[::-1]


def stable_sort(C: np.ndarray) -> np.ndarray:
    """Stable sort - maintains order of equal elements."""
    return C.argsort(kind='stable', axis=0)


# Statistical Sorting Functions
def median_based_sort(C: np.ndarray) -> np.ndarray:
    """Sort based on distance from median - more robust to outliers."""
    median_vals = np.median(C, axis=0)
    distances = np.abs(C - median_vals)
    return distances.argsort(kind='quick', axis=0)


def mean_based_sort(C: np.ndarray) -> np.ndarray:
    """Sort based on distance from mean."""
    mean_vals = np.mean(C, axis=0)
    distances = np.abs(C - mean_vals)
    return distances.argsort(kind='quick', axis=0)


def percentile_based_sort(C: np.ndarray, percentile: float = 75) -> np.ndarray:
    """Sort based on distance from specific percentile."""
    percentile_vals = np.percentile(C, percentile, axis=0)
    distances = np.abs(C - percentile_vals)
    return distances.argsort(kind='quick', axis=0)


# Variance-based Sorting Functions
def variance_based_sort(C: np.ndarray) -> np.ndarray:
    """Sort based on variance contribution - emphasizes variable points."""
    # For each column, sort by how much each point contributes to variance
    result = np.zeros_like(C, dtype=int)
    for col in range(C.shape[1]):
        col_data = C[:, col]
        mean_val = np.mean(col_data)
        contributions = (col_data - mean_val)**2
        result[:, col] = contributions.argsort(kind='quick')
    return result


def std_based_sort(C: np.ndarray) -> np.ndarray:
    """Sort based on standard deviation contribution."""
    # For each column, sort by absolute deviation from mean
    result = np.zeros_like(C, dtype=int)
    for col in range(C.shape[1]):
        col_data = C[:, col]
        mean_val = np.mean(col_data)
        deviations = np.abs(col_data - mean_val)
        result[:, col] = deviations.argsort(kind='quick')
    return result


# Distance-based Sorting Functions
def euclidean_distance_sort(C: np.ndarray) -> np.ndarray:
    """Sort based on Euclidean distance from centroid for each column."""
    # For each column, sort by distance from column mean
    result = np.zeros_like(C, dtype=int)
    for col in range(C.shape[1]):
        col_data = C[:, col]
        col_mean = np.mean(col_data)
        distances = np.abs(col_data - col_mean)
        result[:, col] = distances.argsort(kind='quick')
    return result


def manhattan_distance_sort(C: np.ndarray) -> np.ndarray:
    """Sort based on Manhattan distance from column mean."""
    # For each column, sort by absolute distance from column mean
    result = np.zeros_like(C, dtype=int)
    for col in range(C.shape[1]):
        col_data = C[:, col]
        col_mean = np.mean(col_data)
        distances = np.abs(col_data - col_mean)
        result[:, col] = distances.argsort(kind='quick')
    return result


def cosine_distance_sort(C: np.ndarray) -> np.ndarray:
    """Sort based on cosine distance from column mean."""
    # For each column, sort by distance from column mean (simplified)
    result = np.zeros_like(C, dtype=int)
    for col in range(C.shape[1]):
        col_data = C[:, col]
        col_mean = np.mean(col_data)
        # Use simple distance for each column
        distances = np.abs(col_data - col_mean)
        result[:, col] = distances.argsort(kind='quick')
    return result


# Rank-based Sorting Functions
def rank_based_sort(C: np.ndarray) -> np.ndarray:
    """Sort based on rank within each column."""
    # For each column, sort by rank
    result = np.zeros_like(C, dtype=int)
    for col in range(C.shape[1]):
        col_data = C[:, col]
        result[:, col] = stats.rankdata(col_data).argsort(kind='quick')
    return result


def spearman_rank_sort(C: np.ndarray) -> np.ndarray:
    """Sort based on Spearman rank correlation within each column."""
    # For each column, sort by rank correlation with column mean
    result = np.zeros_like(C, dtype=int)
    for col in range(C.shape[1]):
        col_data = C[:, col]
        col_mean = np.mean(col_data)
        # Calculate rank correlation for each point with the mean
        correlations = []
        for i in range(len(col_data)):
            corr, _ = stats.spearmanr([col_data[i]], [col_mean])
            correlations.append(corr if not np.isnan(corr) else 0)
        result[:, col] = np.array(correlations).argsort(kind='quick')[::-1]  # Higher correlation first
    return result


# Entropy-based Sorting Functions
def entropy_based_sort(C: np.ndarray) -> np.ndarray:
    """Sort based on information content within each column."""
    # For each column, sort by information content
    result = np.zeros_like(C, dtype=int)
    for col in range(C.shape[1]):
        col_data = C[:, col]
        # Calculate entropy for each value in the column
        entropies = []
        for i in range(len(col_data)):
            # Simple entropy measure: how different is this value from others
            other_values = np.delete(col_data, i)
            if len(other_values) > 0:
                # Calculate how unique this value is
                uniqueness = 1.0 / (1.0 + np.sum(np.abs(col_data[i] - other_values)))
                entropies.append(uniqueness)
            else:
                entropies.append(0)
        result[:, col] = np.array(entropies).argsort(kind='quick')[::-1]  # Higher entropy first
    return result


# Outlier-based Sorting Functions
def outlier_based_sort(C: np.ndarray) -> np.ndarray:
    """Sort based on outlier score using IQR method for each column."""
    # For each column, sort by outlier score
    result = np.zeros_like(C, dtype=int)
    for col in range(C.shape[1]):
        col_data = C[:, col]
        q1, q3 = np.percentile(col_data, [25, 75])
        iqr = q3 - q1
        if iqr > 0:
            # Calculate outlier score for each value
            outlier_scores = []
            for i in range(len(col_data)):
                value = col_data[i]
                outlier_score = 0
                if value < q1 - 1.5*iqr or value > q3 + 1.5*iqr:
                    outlier_score = abs(value - np.median(col_data)) / iqr
                outlier_scores.append(outlier_score)
        else:
            outlier_scores = [0] * len(col_data)
        result[:, col] = np.array(outlier_scores).argsort(kind='quick')[::-1]  # More outliers first
    return result


def robust_outlier_sort(C: np.ndarray) -> np.ndarray:
    """Sort based on robust outlier detection using MAD for each column."""
    # For each column, sort by robust outlier score
    result = np.zeros_like(C, dtype=int)
    for col in range(C.shape[1]):
        col_data = C[:, col]
        median = np.median(col_data)
        mad = np.median(np.abs(col_data - median))
        if mad == 0:
            outlier_scores = [0] * len(col_data)
        else:
            # Calculate outlier score for each value
            outlier_scores = []
            for i in range(len(col_data)):
                value = col_data[i]
                outlier_score = 0
                if abs(value - median) > 3 * mad:
                    outlier_score = abs(value - median) / mad
                outlier_scores.append(outlier_score)
        result[:, col] = np.array(outlier_scores).argsort(kind='quick')[::-1]  # More outliers first
    return result


# Clustering-based Sorting Functions
def kmeans_based_sort(C: np.ndarray, n_clusters: int = 2) -> np.ndarray:
    """Sort based on K-means cluster assignment for each column."""
    # For each column, apply k-means clustering and sort
    result = np.zeros_like(C, dtype=int)
    for col in range(C.shape[1]):
        col_data = C[:, col].reshape(-1, 1)
        try:
            
            kmeans = KMeans(n_clusters=min(n_clusters, len(col_data)), random_state=42, n_init=10)
            clusters = kmeans.fit_predict(col_data)
            # Sort by cluster assignment, then by distance to cluster center
            sorted_indices = []
            for cluster_id in range(len(np.unique(clusters))):
                cluster_points = np.where(clusters == cluster_id)[0]
                if len(cluster_points) > 0:
                    cluster_center = kmeans.cluster_centers_[cluster_id][0]
                    distances = np.abs(col_data[cluster_points].flatten() - cluster_center)
                    cluster_sorted = cluster_points[np.argsort(distances)]
                    sorted_indices.extend(cluster_sorted)
            result[:, col] = np.array(sorted_indices)
        except ImportError:
            # Fallback to ascending sort if sklearn not available
            result[:, col] = col_data.flatten().argsort(kind='quick')
    return result


def hierarchical_cluster_sort(C: np.ndarray) -> np.ndarray:
    """Sort based on hierarchical clustering order for each column."""
    # For each column, apply hierarchical clustering and sort
    result = np.zeros_like(C, dtype=int)
    for col in range(C.shape[1]):
        col_data = C[:, col]
        try:
            
            
            # Calculate linkage matrix for this column
            linkage_matrix = linkage(pdist(col_data.reshape(-1, 1)), method='ward')
            
            # Get dendrogram order
            dendro = dendrogram(linkage_matrix, no_plot=True)
            result[:, col] = np.array(dendro['leaves'])
        except ImportError:
            # Fallback to ascending sort if scipy not available
            result[:, col] = col_data.argsort(kind='quick')
    return result


# Adaptive Sorting Functions
def adaptive_sort(C: np.ndarray) -> np.ndarray:
    """Adaptive sorting - chooses method based on data characteristics."""
    n_samples, n_features = C.shape
    
    if n_samples < 3:
        return ascending_sort(C)
    
    # Check for outliers
    outlier_ratio = 0
    for col in range(n_features):
        q1, q3 = np.percentile(C[:, col], [25, 75])
        iqr = q3 - q1
        if iqr > 0:
            outliers = np.sum((C[:, col] < q1 - 1.5*iqr) | (C[:, col] > q3 + 1.5*iqr))
            outlier_ratio += outliers / n_samples
    outlier_ratio /= n_features
    
    # Check for skewness
    skewness = np.mean([abs(stats.skew(C[:, col])) for col in range(n_features)])
    
    # Choose sorting method based on data characteristics
    if outlier_ratio > 0.1:  # Many outliers
        return robust_outlier_sort(C)
    elif skewness > 1.0:  # Highly skewed
        return median_based_sort(C)
    elif np.std(C) / np.mean(C) > 0.5:  # High coefficient of variation
        return variance_based_sort(C)
    else:  # Normal distribution
        return ascending_sort(C)


def robust_adaptive_sort(C: np.ndarray) -> np.ndarray:
    """Robust adaptive sorting - combines multiple robust methods."""
    n_samples, n_features = C.shape
    
    if n_samples < 3:
        return ascending_sort(C)
    
    # Calculate multiple robust statistics
    median_vals = np.median(C, axis=0)
    mad_vals = np.median(np.abs(C - median_vals), axis=0)
    
    # If MAD is small relative to median, data is consistent
    consistency_ratio = np.mean(mad_vals / np.abs(median_vals))
    
    if consistency_ratio < 0.1:
        return median_based_sort(C)
    else:
        # Use robust outlier-based sorting
        return robust_outlier_sort(C)


# Dictionary of all sorting functions
SORTING_FUNCTIONS = {
    # Basic functions
    'ascending': ascending_sort,
    'descending': descending_sort,
    'stable': stable_sort,
    
    # Statistical functions
    'median_based': median_based_sort,
    'mean_based': mean_based_sort,
    'percentile_75': lambda C: percentile_based_sort(C, 75),
    'percentile_90': lambda C: percentile_based_sort(C, 90),
    
    # Variance functions
    'variance_based': variance_based_sort,
    'std_based': std_based_sort,
    
    # Distance functions
    'euclidean_distance': euclidean_distance_sort,
    'manhattan_distance': manhattan_distance_sort,
    'cosine_distance': cosine_distance_sort,
    
    # Rank functions
    'rank_based': rank_based_sort,
    'spearman_rank': spearman_rank_sort,
    
    # Entropy functions
    'entropy_based': entropy_based_sort,
    
    # Outlier functions
    'outlier_based': outlier_based_sort,
    'robust_outlier': robust_outlier_sort,
    
    # Clustering functions
    'kmeans_2': lambda C: kmeans_based_sort(C, 2),
    'kmeans_3': lambda C: kmeans_based_sort(C, 3),
    'hierarchical': hierarchical_cluster_sort,
    
    # Adaptive functions
    'adaptive': adaptive_sort,
    'robust_adaptive': robust_adaptive_sort,
}


def get_sorting_function(name: str) -> Callable[[np.ndarray], np.ndarray]:
    """
    Get a sorting function by name.
    
    Args:
        name: Name of the sorting function
        
    Returns:
        Callable: The sorting function
        
    Raises:
        ValueError: If the function name is not found
    """
    if name not in SORTING_FUNCTIONS:
        available = list(SORTING_FUNCTIONS.keys())
        raise ValueError(f"Unknown sorting function '{name}'. Available: {available}")
    
    return SORTING_FUNCTIONS[name]


def get_all_sorting_functions() -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
    """
    Get all available sorting functions.
    
    Returns:
        Dict: Dictionary mapping function names to functions
    """
    return SORTING_FUNCTIONS.copy()


def get_recommended_sorting_functions() -> List[str]:
    """
    Get a list of recommended sorting functions for clustering.
    
    Returns:
        List: List of recommended function names
    """
    return [
        'ascending',        # Default baseline
        'median_based',     # Robust to outliers
        'variance_based',   # Emphasizes variable points
        'euclidean_distance', # Distance-based
        'rank_based',       # Rank-based
        'entropy_based',    # Information content
        'robust_outlier',   # Outlier detection
        'adaptive',         # Data-driven choice
        'robust_adaptive',  # Robust data-driven choice
        'hierarchical',     # Clustering-based
    ]


def evaluate_sorting_function(sort_func: Callable[[np.ndarray], np.ndarray], 
                            test_data: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate a sorting function on test data.
    
    Args:
        sort_func: Sorting function to evaluate
        test_data: Test data array
        
    Returns:
        Dict: Evaluation metrics
    """
    try:
        result = sort_func(test_data)
        return {
            'result': result,
            'success': True,
            'error': None,
            'unique_order': len(np.unique(result)) == len(result)
        }
    except Exception as e:
        return {
            'result': None,
            'success': False,
            'error': str(e),
            'unique_order': False
        }


if __name__ == "__main__":
    # Test the sorting functions
    test_data = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 2.0],
        [7.0, .0, 9.0],
        [0.5, 1.5, 1.5],
        [3.0, 1.0, 2.0]
    ])
    
    print("Testing sorting functions:")
    print(f"Test data shape: {test_data.shape}")
    print()
    
    for name, func in SORTING_FUNCTIONS.items():
        try:
            result = func(test_data)
            print(f"{name:25s}: {result}")
        except Exception as e:
            print(f"{name:25s}: ERROR - {e}")
    
    print(f"\nRecommended functions: {get_recommended_sorting_functions()}") 