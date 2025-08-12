"""
Metrics calculation module for clustering and supervised learning evaluation.
Consolidates all metrics calculations used across the project.
"""

from typing import Any, Dict, Optional
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    v_measure_score,
    adjusted_rand_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier


def safe_cluster_metrics(
    X: np.ndarray, 
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    kmeans_model: Optional[KMeans] = None
) -> Dict[str, Any]:
    """
    Calculate clustering metrics with safe error handling.
    
    Args:
        X: Feature matrix
        y_true: True labels
        y_pred: Predicted cluster labels
        kmeans_model: Optional KMeans model for inertia
        
    Returns:
        Dictionary with clustering metrics
    """
    metrics: Dict[str, Any] = {
        "v_measure": np.nan,
        "adjusted_rand": np.nan,
        "silhouette": np.nan,
        "calinski_harabasz": np.nan,
        "davies_bouldin": np.nan,
        "inertia": np.nan,
    }
    
    # Always safe metrics
    try:
        metrics["v_measure"] = float(v_measure_score(y_true, y_pred))
    except Exception:
        pass
        
    try:
        metrics["adjusted_rand"] = float(adjusted_rand_score(y_true, y_pred))
    except Exception:
        pass
    
    # Silhouette score requires 2+ clusters and proper data
    try:
        n_unique_pred = len(set(y_pred))
        if n_unique_pred >= 2 and n_unique_pred < X.shape[0]:
            metrics["silhouette"] = float(silhouette_score(X, y_pred))
    except Exception:
        pass
    
    # Calinski-Harabasz score
    try:
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(X, y_pred))
    except Exception:
        pass
    
    # Davies-Bouldin score
    try:
        metrics["davies_bouldin"] = float(davies_bouldin_score(X, y_pred))
    except Exception:
        pass
    
    # KMeans inertia if model provided
    if kmeans_model is not None:
        try:
            metrics["inertia"] = float(getattr(kmeans_model, "inertia_", np.nan))
        except Exception:
            pass
    
    return metrics


def safe_supervised_metrics(
    X: np.ndarray, 
    y: np.ndarray, 
    cv_splits: int = 3
) -> Dict[str, Any]:
    """
    Calculate supervised learning metrics with safe error handling.
    
    Args:
        X: Feature matrix
        y: True labels
        cv_splits: Number of cross-validation splits
        
    Returns:
        Dictionary with supervised learning metrics
    """
    metrics: Dict[str, Any] = {
        "linear_sep_cv": np.nan,
        "cv_score": np.nan,
        "margin_score": np.nan,
        "boundary_complexity": np.nan,
    }
    
    # Setup cross-validation (adapt splits to class counts when possible)
    try:
        # Compute minimum class count to determine feasible n_splits
        _, counts = np.unique(y, return_counts=True)
        min_class_count = int(counts.min()) if counts.size > 0 else 0
        effective_splits = min(int(cv_splits), max(0, min_class_count))
        if effective_splits < 2:
            return metrics
        cv = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=0)
    except Exception:
        # If not enough samples for stratified, return NaN metrics
        return metrics
    
    # Linear separability (LinearSVC accuracy)
    try:
        lin_svm = make_pipeline(
            StandardScaler(with_mean=True), 
            LinearSVC(dual=False, max_iter=2000, random_state=0)
        )
        lin_scores = cross_val_score(lin_svm, X, y, cv=cv, scoring="accuracy", n_jobs=1)
        metrics["linear_sep_cv"] = float(np.mean(lin_scores))
    except Exception:
        pass
    
    # KNN accuracy (5 neighbors)
    try:
        knn = make_pipeline(
            StandardScaler(with_mean=True), 
            KNeighborsClassifier(n_neighbors=5)
        )
        knn_scores = cross_val_score(knn, X, y, cv=cv, scoring="accuracy", n_jobs=1)
        metrics["cv_score"] = float(np.mean(knn_scores))
    except Exception:
        pass
    
    # Margin score (LogisticRegression confidence)
    try:
        lr = make_pipeline(
            StandardScaler(with_mean=True), 
            LogisticRegression(max_iter=200, n_jobs=1)
        )
        proba = cross_val_predict(lr, X, y, cv=cv, method="predict_proba", n_jobs=1)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            part_sorted = np.sort(proba, axis=1)
            margins = part_sorted[:, -1] - part_sorted[:, -2]
            metrics["margin_score"] = float(np.mean(margins))
    except Exception:
        pass
    
    # Boundary complexity (1-NN error rate)
    try:
        knn1 = make_pipeline(
            StandardScaler(with_mean=True), 
            KNeighborsClassifier(n_neighbors=1)
        )
        knn1_scores = cross_val_score(knn1, X, y, cv=cv, scoring="accuracy", n_jobs=1)
        metrics["boundary_complexity"] = float(1.0 - np.mean(knn1_scores))
    except Exception:
        pass
    
    return metrics


def calculate_all_metrics(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    kmeans_model: Optional[KMeans] = None,
    cv_splits: int = 3
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate both clustering and supervised metrics.
    
    Args:
        X: Feature matrix
        y_true: True labels
        y_pred: Predicted cluster labels
        kmeans_model: Optional KMeans model for inertia
        cv_splits: Number of cross-validation splits
        
    Returns:
        Dictionary with 'cluster' and 'supervised' metrics
    """
    return {
        "cluster": safe_cluster_metrics(X, y_true, y_pred, kmeans_model),
        "supervised": safe_supervised_metrics(X, y_true, cv_splits)
    }