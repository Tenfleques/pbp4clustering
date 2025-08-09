#!/usr/bin/env python3
import argparse
from typing import Any, Dict
from pathlib import Path

import numpy as np
from datasets.penguins_loader import load_penguins_matrices
from pbp_transform import matrices_to_pbp_vectors
from visualize import scatter_features
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


def main():
    parser = argparse.ArgumentParser(description="Run PBP on Palmer Penguins (2x2) and report metrics/plot.")
    parser.add_argument("--data-dir", default="./data/penguins", help="Directory containing penguins.csv")
    parser.add_argument("--agg", default="sum", help="Aggregation function name for PBP")
    parser.add_argument("--results-dir", default="./results", help="Directory to write plot outputs")
    parser.add_argument("--plot", dest="plot", action="store_true", default=True, help="Generate scatter plot of PBP features")
    parser.add_argument("--no-plot", dest="plot", action="store_false", help="Disable plotting")
    args = parser.parse_args()

    X_mats, y, meta = load_penguins_matrices(args.data_dir)
    print(f"Loaded Penguins: matrices={X_mats.shape}, labels={y.shape}, m×n={meta['matrix_shape']}")

    X_pbp = matrices_to_pbp_vectors(X_mats, agg=args.agg)
    X_pbp = np.asarray(X_pbp)
    print(f"PBP vectors: {X_pbp.shape} (agg={args.agg})")

    non_zero_cols = ~(np.all(X_pbp == 0, axis=0))
    if non_zero_cols.sum() != X_pbp.shape[1]:
        X_pbp = X_pbp[:, non_zero_cols]

    if args.plot:
        out_png = str(Path(args.results_dir) / f"penguins_targets_pbp_{args.agg}.png")
        scatter_features(X_pbp, y, out_png, title=f"Penguins PBP (agg={args.agg}) - Species", label_names=meta.get("labels_map"))
        print(f"Saved: {out_png}")

    n_clusters = len(set(int(v) for v in y))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    pred = km.fit_predict(X_pbp)

    def safe_cluster_metrics() -> Dict[str, Any]:
        m: Dict[str, Any] = {
            "v_measure": float(v_measure_score(y, pred)),
            "adjusted_rand": float(adjusted_rand_score(y, pred)),
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
            "inertia": float(getattr(km, "inertia_", np.nan)),
        }
        try:
            if len(set(pred)) >= 2 and len(set(pred)) < X_pbp.shape[0]:
                m["silhouette"] = float(silhouette_score(X_pbp, pred))
        except Exception:
            pass
        try:
            m["calinski_harabasz"] = float(calinski_harabasz_score(X_pbp, pred))
        except Exception:
            pass
        try:
            m["davies_bouldin"] = float(davies_bouldin_score(X_pbp, pred))
        except Exception:
            pass
        return m

    def safe_supervised_metrics(cv_splits: int = 5) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "linear_sep_cv": np.nan,
            "cv_score": np.nan,
            "margin_score": np.nan,
            "boundary_complexity": np.nan,
        }
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=0)
        try:
            lin_svm = make_pipeline(StandardScaler(with_mean=True), LinearSVC(dual=False, max_iter=2000, random_state=0))
            lin_scores = cross_val_score(lin_svm, X_pbp, y, cv=cv, scoring="accuracy", n_jobs=1)
            metrics["linear_sep_cv"] = float(np.mean(lin_scores))
        except Exception:
            pass
        try:
            knn = make_pipeline(StandardScaler(with_mean=True), KNeighborsClassifier(n_neighbors=5))
            knn_scores = cross_val_score(knn, X_pbp, y, cv=cv, scoring="accuracy", n_jobs=1)
            metrics["cv_score"] = float(np.mean(knn_scores))
        except Exception:
            pass
        try:
            lr = make_pipeline(StandardScaler(with_mean=True), LogisticRegression(max_iter=200, n_jobs=1))
            proba = cross_val_predict(lr, X_pbp, y, cv=cv, method="predict_proba", n_jobs=1)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                part_sorted = np.sort(proba, axis=1)
                margins = part_sorted[:, -1] - part_sorted[:, -2]
                metrics["margin_score"] = float(np.mean(margins))
        except Exception:
            pass
        try:
            knn1 = make_pipeline(StandardScaler(with_mean=True), KNeighborsClassifier(n_neighbors=1))
            knn1_scores = cross_val_score(knn1, X_pbp, y, cv=cv, scoring="accuracy", n_jobs=1)
            metrics["boundary_complexity"] = float(1.0 - np.mean(knn1_scores))
        except Exception:
            pass
        return metrics

    cm = safe_cluster_metrics()
    sm = safe_supervised_metrics(cv_splits=5)

    print("Metrics (Penguins, PBP)")
    print(f"- agg={args.agg}, n_samples={X_pbp.shape[0]}, n_features={X_pbp.shape[1]}, n_clusters={n_clusters}")
    print(f"- v_measure={cm['v_measure']}, adjusted_rand={cm['adjusted_rand']}")
    print(f"- silhouette={cm['silhouette']}, calinski_harabasz={cm['calinski_harabasz']}, davies_bouldin={cm['davies_bouldin']}, inertia={cm['inertia']}")
    print(f"- linear_sep_cv={sm['linear_sep_cv']}, cv_score={sm['cv_score']}, margin_score={sm['margin_score']}, boundary_complexity={sm['boundary_complexity']}")


if __name__ == "__main__":
    main()


