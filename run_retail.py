#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Any, Dict
from visualize import scatter_features
import numpy as np
from datasets.online_retail_loader import load_online_retail_matrices
from pbp_transform import matrices_to_pbp_vectors
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
    parser = argparse.ArgumentParser(description="Run PBP on UCI Online Retail II per-customer matrices and report metrics.")
    parser.add_argument("--data-dir", default="./data/retail", help="Download/cache directory")
    parser.add_argument("--results-dir", default="./results", help="Directory to write outputs")
    parser.add_argument("--agg", default="sum", help="Aggregation function name")
    parser.add_argument("--rows", type=int, default=None, help="Optionally reduce rows (m) to this value via PCA before PBP (e.g., 2 or 3)")
    parser.add_argument("--top-k-countries", type=int, default=6, help="Keep K most frequent countries; map others to 'Other'")
    parser.add_argument("--months-agg", type=int, choices=[4, 6], default=4, help="Aggregate months into blocks of this size (mean); 4->3 cols, 6->2 cols")
    parser.add_argument("--rows-from-months", action="store_true", help="After months aggregation, transpose so rows = aggregated months (3 or 2) and columns = features (6)")
    args = parser.parse_args()

    X_mats, y, meta = load_online_retail_matrices(args.data_dir)
    print(f"Loaded Retail: matrices={X_mats.shape}, labels={y.shape}, m×n={meta['matrix_shape']}")

    # Optional: aggregate months into larger windows along the time axis (columns)
    if args.months_agg is not None:
        N, m, n = X_mats.shape
        b = int(args.months_agg)
        if n % b == 0:
            X_mats = X_mats.reshape(N, m, n // b, b).mean(axis=3)
            print(f"Aggregated months by {b}: new shape={X_mats.shape}")
            if args.rows_from_months:
                # Make rows equal to aggregated months; columns become features
                X_mats = np.swapaxes(X_mats, 1, 2)
                print(f"Transposed to rows-from-months: shape={X_mats.shape}")
        else:
            print(f"Warning: months ({n}) not divisible by {b}; skipping months aggregation")

    # Optional: keep only top-K most frequent countries
    if args.top_k_countries is not None and args.top_k_countries > 0:
        k = int(args.top_k_countries)
        counts = np.bincount(y)
        top_ids = np.argsort(counts)[::-1][:k]
        remap: Dict[int, int] = {int(cid): i for i, cid in enumerate(top_ids)}
        other_id_new = k
        y = np.array([remap.get(int(lbl), other_id_new) for lbl in y], dtype=int)

        # Best-effort label names for reporting
        id_to_country = {v: k for k, v in (meta.get("label_map") or {}).items()}
        kept = [id_to_country.get(int(cid), str(cid)) for cid in top_ids]
        print(f"Top-K countries kept (K={k}): {kept}; others mapped to 'Other'")

    X_pbp = matrices_to_pbp_vectors(X_mats, agg=args.agg, rows_target=args.rows)
    X_pbp = np.asarray(X_pbp)
    print(f"PBP vectors: {X_pbp.shape} (agg={args.agg})")

    # Optional: drop all-zero columns
    non_zero_cols = ~(np.all(X_pbp == 0, axis=0))
    if non_zero_cols.sum() != X_pbp.shape[1]:
        X_pbp = X_pbp[:, non_zero_cols]

    out_png = str(Path(args.results_dir) / f"retail_targets_pbp_{args.agg}.png")
    scatter_features(X_pbp, y, out_png, title=f"Retail PBP (agg={args.agg}) - True Targets", label_names=meta.get("labels_map"))
    print(f"Saved: {out_png}")

    n_clusters = len(set(int(v) for v in y))
    if n_clusters < 2:
        print("Not enough classes for clustering metrics.")
        return

    # KMeans clustering
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    pred = km.fit_predict(X_pbp)

    # Clustering metrics
    def safe_cluster_metrics() -> Dict[str, Any]:
        m: Dict[str, Any] = {
            "v_measure": np.nan,
            "adjusted_rand": np.nan,
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
            "inertia": np.nan,
        }
        try:
            m["v_measure"] = float(v_measure_score(y, pred))
            m["adjusted_rand"] = float(adjusted_rand_score(y, pred))
        except Exception:
            pass
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
        try:
            m["inertia"] = float(getattr(km, "inertia_", np.nan))
        except Exception:
            pass
        return m

    # Separability metrics via CV
    def safe_supervised_metrics(cv_splits: int = 3) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "linear_sep_cv": np.nan,
            "cv_score": np.nan,
            "margin_score": np.nan,
            "boundary_complexity": np.nan,
        }
        if len(set(y)) < 2 or X_pbp.shape[0] <= cv_splits:
            return metrics
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
    sm = safe_supervised_metrics(cv_splits=3)

    print("Metrics (Retail, PBP)")
    print(f"- agg={args.agg}, n_samples={X_pbp.shape[0]}, n_features={X_pbp.shape[1]}, n_clusters={n_clusters}")
    print(f"- v_measure={cm['v_measure']}, adjusted_rand={cm['adjusted_rand']}")
    print(f"- silhouette={cm['silhouette']}, calinski_harabasz={cm['calinski_harabasz']}, davies_bouldin={cm['davies_bouldin']}, inertia={cm['inertia']}")
    print(f"- linear_sep_cv={sm['linear_sep_cv']}, cv_score={sm['cv_score']}, margin_score={sm['margin_score']}, boundary_complexity={sm['boundary_complexity']}")


if __name__ == "__main__":
    main() 