"""
Common utility functions for the clustering project.
Provides helper functions for data processing and result formatting.
"""

from typing import Dict, Any, Optional
import numpy as np


def aggregate_rows_in_blocks(X: np.ndarray, num_blocks: int) -> np.ndarray:
    """
    Aggregate matrix rows into blocks by averaging.
    
    Args:
        X: Input matrix of shape (N, m, n)
        num_blocks: Number of blocks to create
        
    Returns:
        Aggregated matrix of shape (N, num_blocks, n)
    """
    N, m, n = X.shape
    blocks = np.array_split(np.arange(m), num_blocks)
    out = np.empty((N, num_blocks, n), dtype=X.dtype)
    for i, idxs in enumerate(blocks):
        out[:, i, :] = X[:, idxs, :].mean(axis=1)
    return out


def format_results(
    metrics_dict: Dict[str, Dict[str, Any]], 
    dataset_name: str, 
    agg_func: str,
    n_samples: int,
    n_features: int,
    n_clusters: int
) -> str:
    """
    Format metrics results into a readable string.
    
    Args:
        metrics_dict: Dictionary with 'cluster' and 'supervised' metrics
        dataset_name: Name of the dataset
        agg_func: Aggregation function used
        n_samples: Number of samples
        n_features: Number of features
        n_clusters: Number of clusters
        
    Returns:
        Formatted string with all metrics
    """
    lines = []
    lines.append(f"Metrics ({dataset_name}, PBP)")
    lines.append(f"- agg={agg_func}, n_samples={n_samples}, n_features={n_features}, n_clusters={n_clusters}")
    
    cm = metrics_dict.get("cluster", {})
    lines.append(f"- v_measure={cm.get('v_measure', np.nan):.4f}, adjusted_rand={cm.get('adjusted_rand', np.nan):.4f}")
    lines.append(f"- silhouette={cm.get('silhouette', np.nan):.4f}, calinski_harabasz={cm.get('calinski_harabasz', np.nan):.4f}, davies_bouldin={cm.get('davies_bouldin', np.nan):.4f}, inertia={cm.get('inertia', np.nan):.4f}")
    
    sm = metrics_dict.get("supervised", {})
    lines.append(f"- linear_sep_cv={sm.get('linear_sep_cv', np.nan):.4f}, cv_score={sm.get('cv_score', np.nan):.4f}, margin_score={sm.get('margin_score', np.nan):.4f}, boundary_complexity={sm.get('boundary_complexity', np.nan):.4f}")
    
    return "\n".join(lines)


def print_metrics_summary(
    cluster_metrics: Dict[str, Any],
    supervised_metrics: Dict[str, Any],
    metadata: Dict[str, Any]
):
    """
    Print a formatted summary of all metrics.
    
    Args:
        cluster_metrics: Clustering metrics dictionary
        supervised_metrics: Supervised learning metrics dictionary
        metadata: Additional metadata (dataset_name, agg_func, n_samples, etc.)
    """
    dataset_name = metadata.get("dataset_name", "Dataset")
    agg_func = metadata.get("agg_func", "unknown")
    n_samples = metadata.get("n_samples", 0)
    n_features = metadata.get("n_features", 0)
    n_clusters = metadata.get("n_clusters", 0)
    
    metrics_dict = {
        "cluster": cluster_metrics,
        "supervised": supervised_metrics
    }
    
    formatted = format_results(
        metrics_dict, dataset_name, agg_func,
        n_samples, n_features, n_clusters
    )
    print(formatted)


def format_float(value: float, precision: int = 4) -> str:
    """
    Format a float value with specified precision, handling NaN.
    
    Args:
        value: Float value to format
        precision: Number of decimal places
        
    Returns:
        Formatted string
    """
    if np.isnan(value):
        return "NaN"
    return f"{value:.{precision}f}"