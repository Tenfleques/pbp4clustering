"""
Standard clustering pipeline module.
Provides common data processing and clustering operations.
"""

from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np
from sklearn.cluster import KMeans
from pbp_transform import matrices_to_pbp_vectors
from visualize import scatter_features
from .metrics import calculate_all_metrics


def filter_zero_columns(X: np.ndarray) -> np.ndarray:
    """
    Remove columns that are all zeros from feature matrix.
    
    Args:
        X: Feature matrix
        
    Returns:
        Filtered feature matrix
    """
    non_zero_cols = ~(np.all(X == 0, axis=0))
    if non_zero_cols.sum() != X.shape[1]:
        return X[:, non_zero_cols]
    return X


def setup_kmeans(
    n_clusters: int, 
    random_state: int = 0, 
    n_init: int = 10
) -> KMeans:
    """
    Create a configured KMeans clustering model.
    
    Args:
        n_clusters: Number of clusters
        random_state: Random seed for reproducibility
        n_init: Number of times k-means will be run
        
    Returns:
        Configured KMeans model
    """
    return KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)


def run_clustering_pipeline(
    X_matrices: np.ndarray,
    y: np.ndarray,
    agg_func: str,
    results_dir: str = "./results",
    plot: bool = True,
    dataset_name: str = "dataset",
    label_names: Optional[Dict[int, str]] = None,
    cv_splits: int = 3
) -> Dict[str, Any]:
    """
    Run the complete clustering pipeline on matrix data.
    
    Args:
        X_matrices: Input matrices of shape (N, m, n)
        y: True labels
        agg_func: Aggregation function name for PBP
        results_dir: Directory to save plots
        plot: Whether to generate scatter plot
        dataset_name: Name of the dataset for output files
        label_names: Optional mapping of label indices to names
        cv_splits: Number of CV splits for supervised metrics
        
    Returns:
        Dictionary containing all results and metrics
    """
    # Transform to PBP vectors
    X_pbp = matrices_to_pbp_vectors(X_matrices, agg=agg_func)
    X_pbp = np.asarray(X_pbp)
    
    # Filter zero columns
    X_pbp = filter_zero_columns(X_pbp)
    
    # Visualize if requested
    if plot:
        out_png = str(Path(results_dir) / f"{dataset_name}_targets_pbp_{agg_func}.png")
        scatter_features(
            X_pbp, y, out_png, 
            title=f"{dataset_name.title()} PBP (agg={agg_func})",
            label_names=label_names
        )
    
    # Clustering
    n_clusters = len(set(int(v) for v in y))
    km = setup_kmeans(n_clusters)
    pred = km.fit_predict(X_pbp)
    
    # Calculate metrics
    all_metrics = calculate_all_metrics(X_pbp, y, pred, km, cv_splits)
    
    # Return comprehensive results
    return {
        "X_pbp": X_pbp,
        "predictions": pred,
        "kmeans_model": km,
        "metrics": all_metrics,
        "n_clusters": n_clusters,
        "n_features": X_pbp.shape[1],
        "n_samples": X_pbp.shape[0],
        "agg_func": agg_func
    }


def cluster_and_predict(
    X: np.ndarray, 
    n_clusters: Optional[int] = None,
    y_true: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, KMeans]:
    """
    Perform clustering and return predictions with the model.
    
    Args:
        X: Feature matrix
        n_clusters: Number of clusters (if None, inferred from y_true)
        y_true: True labels (used to infer n_clusters if not provided)
        
    Returns:
        Tuple of (predictions, kmeans_model)
    """
    if n_clusters is None:
        if y_true is None:
            raise ValueError("Either n_clusters or y_true must be provided")
        n_clusters = len(set(int(v) for v in y_true))
    
    km = setup_kmeans(n_clusters)
    pred = km.fit_predict(X)
    return pred, km