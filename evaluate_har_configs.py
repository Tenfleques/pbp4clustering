#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
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

from datasets.har_loader import (
    load_har_six_axis,
    load_har_axis_feature_format,
    load_har_axis_feature_columns,
)
from pbp_transform import matrices_to_pbp_vectors


def _append_result_row(out_csv: str, row: Dict[str, Any]) -> None:
    """Append a single result row to CSV immediately (create file with header if missing)."""
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = out_path.exists()
    pd.DataFrame([row]).to_csv(out_path, mode="a", header=not file_exists, index=False)


def _safe_cluster_metrics(X: np.ndarray, labels: np.ndarray, km: KMeans) -> Dict[str, Any]:
    m: Dict[str, Any] = {"silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan, "inertia": np.nan}
    try:
        # KMeans inertia
        m["inertia"] = float(getattr(km, "inertia_", np.nan))
    except Exception:
        pass
    try:
        # Silhouette requires at least 2 labels and less than n_samples
        if len(set(labels)) >= 2 and len(set(labels)) < X.shape[0]:
            m["silhouette"] = float(silhouette_score(X, labels))
    except Exception:
        pass
    try:
        m["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
    except Exception:
        pass
    try:
        m["davies_bouldin"] = float(davies_bouldin_score(X, labels))
    except Exception:
        pass
    return m


def _safe_supervised_metrics(X: np.ndarray, y: np.ndarray, cv_splits: int = 3) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "linear_sep_cv": np.nan,
        "cv_score": np.nan,
        "margin_score": np.nan,
        "boundary_complexity": np.nan,
    }
    if len(set(y)) < 2 or X.shape[0] <= cv_splits:
        return metrics
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=0)

    # Linear separability (Linear SVM CV accuracy)
    try:
        lin_svm = make_pipeline(StandardScaler(with_mean=True), LinearSVC(dual=False, max_iter=2000, random_state=0))
        lin_scores = cross_val_score(lin_svm, X, y, cv=cv, scoring="accuracy", n_jobs=1)
        metrics["linear_sep_cv"] = float(np.mean(lin_scores))
    except Exception:
        pass

    # Generic CV score (KNN k=5 accuracy)
    try:
        knn = make_pipeline(StandardScaler(with_mean=True), KNeighborsClassifier(n_neighbors=5))
        knn_scores = cross_val_score(knn, X, y, cv=cv, scoring="accuracy", n_jobs=1)
        metrics["cv_score"] = float(np.mean(knn_scores))
    except Exception:
        pass

    # Margin score (mean top1 - top2 probability gap from LogisticRegression)
    try:
        lr = make_pipeline(StandardScaler(with_mean=True), LogisticRegression(max_iter=200, n_jobs=1))
        # cross_val_predict on pipeline with predict_proba
        proba = cross_val_predict(lr, X, y, cv=cv, method="predict_proba", n_jobs=1)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            part_sorted = np.sort(proba, axis=1)
            margins = part_sorted[:, -1] - part_sorted[:, -2]
            metrics["margin_score"] = float(np.mean(margins))
    except Exception:
        pass

    # Boundary complexity (1 - CV accuracy of 1-NN)
    try:
        knn1 = make_pipeline(StandardScaler(with_mean=True), KNeighborsClassifier(n_neighbors=1))
        knn1_scores = cross_val_score(knn1, X, y, cv=cv, scoring="accuracy", n_jobs=1)
        metrics["boundary_complexity"] = float(1.0 - np.mean(knn1_scores))
    except Exception:
        pass

    return metrics


def map_labels_walking_idle(y: np.ndarray) -> np.ndarray:
    walking_set = {1, 2, 3}
    return np.array([1 if int(lbl) in walking_set else 0 for lbl in y], dtype=int)


def build_config_grid() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []

    # Six-axis: m=6, try optional row reduction
    for rows in [None, 3, 2]:
        for label_mode in ["multiclass", "walking_idle"]:
            configs.append({
                "format": "six_axis",
                "include_body_acc": None,
                "time_agg": None,
                "rows": rows,
                "label_mode": label_mode,
            })

    # Axis-feature-format (concatenate windows across features): m=3; still allow rows reduction
    for include_body_acc in [False, True]:
        for rows in [None, 2]:
            for label_mode in ["multiclass", "walking_idle"]:
                configs.append({
                    "format": "axis_feature_format",
                    "include_body_acc": include_body_acc,
                    "time_agg": None,
                    "rows": rows,
                    "label_mode": label_mode,
                })

    # Axis-feature-columns (aggregate time to feature columns)
    time_aggs = [
        "mean", "median", "sum", "min", "max", "std", "var",
        "rms", "absmean", "mad", "iqr", "energy", "entropy",
    ]
    for include_body_acc in [False, True]:
        for time_agg in time_aggs:
            for label_mode in ["multiclass", "walking_idle"]:
                configs.append({
                    "format": "axis_feature_columns",
                    "include_body_acc": include_body_acc,
                    "time_agg": time_agg,
                    "rows": None,  # Already m=3
                    "label_mode": label_mode,
                })

    return configs


def build_top_plots_configs() -> List[Dict[str, Any]]:
    """Build configurations matching those used in run_top_plots.sh"""
    configs: List[Dict[str, Any]] = []
    
    # Six-axis format configurations (14 total)
    six_axis_pbp_aggs = [
        "max", "sum", "mean", "entropy", "rms", "range", "trimmed_mean",
        "robust_adaptive", "std", "median", "adaptive", "iqr", "gini"
    ]
    
    for pbp_agg in six_axis_pbp_aggs:
        configs.append({
            "format": "six_axis",
            "include_body_acc": True,
            "time_agg": None,
            "rows": None,
            "pbp_agg": pbp_agg,
        })
    
    # Axis-feature-columns format (1 configuration)
    configs.append({
        "format": "axis_feature_columns",
        "include_body_acc": False,
        "time_agg": "max",
        "rows": None,
        "pbp_agg": "max",
    })
    
    # Axis-feature-format format (1 configuration)
    configs.append({
        "format": "axis_feature_format",
        "include_body_acc": True,
        "time_agg": None,
        "rows": None,
        "pbp_agg": "adaptive",
    })
    
    return configs


def evaluate_top_plots_configs(data_dir: str, out_csv: str, limit: Optional[int] = None) -> None:
    """Evaluate only the configurations used in run_top_plots.sh, iterating label mode only"""
    configs = build_top_plots_configs()
    
    for cfg in configs:
        fmt = cfg["format"]
        include_body_acc = cfg["include_body_acc"]
        time_agg = cfg["time_agg"]
        rows = cfg["rows"]
        pbp_agg = cfg["pbp_agg"]
        
        X_mats, y, meta = load_matrices_and_labels(data_dir, fmt, include_body_acc, time_agg)
        
        if limit is not None and limit > 0:
            X_mats = X_mats[:limit]
            y = y[:limit]
        
        # Precompute PBP once per config
        X_pbp = matrices_to_pbp_vectors(X_mats, agg=pbp_agg, rows_target=rows)
        X_pbp = np.asarray(X_pbp)
        non_zero_cols = ~(np.all(X_pbp == 0, axis=0))
        X_pbp = X_pbp[:, non_zero_cols]
        
        for label_mode in ["multiclass", "walking_idle"]:
            if label_mode == "walking_idle":
                y_eval = map_labels_walking_idle(y)
            else:
                y_eval = y
            
            n_clusters = len(set(int(v) for v in y_eval))
            if n_clusters < 2:
                continue
            
            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
            pred = km.fit_predict(X_pbp)
            
            v_score = v_measure_score(y_eval, pred)
            ari = adjusted_rand_score(y_eval, pred)
            cluster_m = _safe_cluster_metrics(X_pbp, pred, km)
            sup_m = _safe_supervised_metrics(X_pbp, y_eval, cv_splits=3)
            
            row: Dict[str, Any] = {
                "format": fmt,
                "include_body_acc": include_body_acc,
                "time_agg": time_agg,
                "rows": rows,
                "label_mode": label_mode,
                "pbp_agg": pbp_agg,
                "n_samples": int(X_pbp.shape[0]),
                "n_features": int(X_pbp.shape[1]),
                "v_measure": float(v_score),
                "adjusted_rand": float(ari),
            }
            row.update(cluster_m)
            row.update(sup_m)
            _append_result_row(out_csv, row)
            
            print(
                f"Top plots config: fmt={fmt}, lbl={label_mode}, pbp={pbp_agg}, body_acc={include_body_acc}, "
                f"v={v_score:.4f}, ari={ari:.4f}, sil={cluster_m['silhouette']}"
            )
    
    print(f"Finished top plots evaluation. Results appended to {out_csv}")


def load_matrices_and_labels(
    data_dir: str,
    fmt: str,
    include_body_acc: Optional[bool],
    time_agg: Optional[str],
):
    if fmt == "six_axis":
        return load_har_six_axis(data_dir)
    if fmt == "axis_feature_format":
        return load_har_axis_feature_format(data_dir, include_body_acc=bool(include_body_acc))
    if fmt == "axis_feature_columns":
        assert time_agg is not None
        return load_har_axis_feature_columns(
            data_dir,
            include_body_acc=bool(include_body_acc),
            time_agg=time_agg,
        )
    raise ValueError(f"Unknown format: {fmt}")


def evaluate_configs(data_dir: str, out_csv: str, limit: Optional[int] = None) -> None:
    grid = build_config_grid()
    # Aggregation functions to evaluate inside PBP
    pbp_aggs = [
        "sum",
        "mean",
        "median",
        "trimmed_mean",
        "rms",
        "adaptive",
        "robust_adaptive",
        "std",
        "var",
        "max",
        "min",
        "iqr",
        "range",
        "entropy",
        "gini",
    ]

    for cfg in grid:
        fmt = cfg["format"]
        include_body_acc = cfg["include_body_acc"]
        time_agg = cfg["time_agg"]
        rows = cfg["rows"]
        label_mode = cfg["label_mode"]

        X_mats, y, meta = load_matrices_and_labels(data_dir, fmt, include_body_acc, time_agg)

        if limit is not None and limit > 0:
            X_mats = X_mats[:limit]
            y = y[:limit]

        if label_mode == "walking_idle":
            y_eval = map_labels_walking_idle(y)
        else:
            y_eval = y

        n_clusters = len(set(int(v) for v in y_eval))
        if n_clusters < 2:
            continue

        for pbp_agg in pbp_aggs:
            X_pbp = matrices_to_pbp_vectors(X_mats, agg=pbp_agg, rows_target=rows)
            X_pbp = np.asarray(X_pbp)
            # Drop all-zero columns if any
            non_zero_cols = ~(np.all(X_pbp == 0, axis=0))
            X_pbp = X_pbp[:, non_zero_cols]

            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
            pred = km.fit_predict(X_pbp)

            v_score = v_measure_score(y_eval, pred)
            ari = adjusted_rand_score(y_eval, pred)
            cluster_m = _safe_cluster_metrics(X_pbp, pred, km)
            sup_m = _safe_supervised_metrics(X_pbp, y_eval, cv_splits=3)

            row: Dict[str, Any] = {
                "format": fmt,
                "include_body_acc": include_body_acc,
                "time_agg": time_agg,
                "rows": rows,
                "label_mode": label_mode,
                "pbp_agg": pbp_agg,
                "n_samples": int(X_pbp.shape[0]),
                "n_features": int(X_pbp.shape[1]),
                "v_measure": float(v_score),
                "adjusted_rand": float(ari),
            }
            row.update(cluster_m)
            row.update(sup_m)
            _append_result_row(out_csv, row)
            # Optional progress print for visibility on long runs
            print(
                f"Appended: fmt={fmt}, lbl={label_mode}, rows={rows}, tagg={time_agg}, pbp={pbp_agg}, "
                f"v={v_score:.4f}, ari={ari:.4f}, sil={cluster_m['silhouette']}, ch={cluster_m['calinski_harabasz']}, "
                f"db={cluster_m['davies_bouldin']}, inrt={cluster_m['inertia']}, lin={sup_m['linear_sep_cv']}, "
                f"cv={sup_m['cv_score']}, margin={sup_m['margin_score']}, bc={sup_m['boundary_complexity']}"
            )

    print(f"Finished. Results appended to {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate HAR configurations with KMeans and separation scores")
    parser.add_argument("--data-dir", default="./data/har", help="Download/cache directory for HAR")
    parser.add_argument("--out-csv", default="./results/har_eval.csv", help="Path to write CSV results")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for quick runs")
    parser.add_argument("--top-plots-only", action="store_true", 
                       help="Evaluate only configurations used in run_top_plots.sh")
    args = parser.parse_args()

    if args.top_plots_only:
        evaluate_top_plots_configs(args.data_dir, args.out_csv, limit=args.limit)
    else:
        evaluate_configs(args.data_dir, args.out_csv, limit=args.limit)


if __name__ == "__main__":
    main()


