#!/usr/bin/env python3
"""
Run dimensionality reduction baselines (PCA, t-SNE, UMAP) on selected datasets
and report the same metrics used for PBP (clustering + supervised probes).

Usage examples:
  .venv/bin/python run_dimred_baselines.py --dataset iris --method pca --no-plot
  .venv/bin/python run_dimred_baselines.py --dataset wdbc --method umap --plot
"""

from __future__ import annotations

import argparse
from typing import Dict, Tuple, Optional

import numpy as np
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    import umap  # type: ignore
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

# Dataset loaders (use raw/tabular loaders and shape into 2D arrays)
from datasets.iris_loader import load_iris_matrices
from datasets.wdbc_loader import load_wdbc_matrices
from datasets.htru2_loader import load_htru2_matrices
from datasets.seeds_loader import load_seeds_matrices
from datasets.penguins_loader import load_penguins_matrices
from datasets.banknote_loader import load_banknote_matrices
from datasets.ionosphere_loader import load_ionosphere_matrices
from datasets.sonar_loader import load_sonar_matrices
from datasets.parkinsons_loader import load_parkinsons_matrices
from datasets.pima_loader import load_pima_matrices
from datasets.vehicle_loader import load_vehicle_matrices
from datasets.spectroscopy_coffee_loader import load_coffee_spectra

from src.metrics import calculate_all_metrics
from sklearn.preprocessing import StandardScaler
from visualize import scatter_features


SUPPORTED = {
    "iris": load_iris_matrices,
    "wdbc": load_wdbc_matrices,
    "htru2": load_htru2_matrices,
    "seeds": load_seeds_matrices,
    "penguins": load_penguins_matrices,
    "banknote": load_banknote_matrices,
    "ionosphere": load_ionosphere_matrices,
    "sonar": load_sonar_matrices,
    "parkinsons": load_parkinsons_matrices,
    "pima": load_pima_matrices,
    "vehicle": load_vehicle_matrices,
    "spectroscopy": load_coffee_spectra,
}


def matrices_to_flat_features(X_mats: np.ndarray) -> np.ndarray:
    """Flatten (N, m, n) matrices into (N, m*n) and z-score features for DR baselines."""
    assert X_mats.ndim == 3
    N, m, n = X_mats.shape
    X = X_mats.reshape(N, m * n)
    # Standard baseline practice: z-score features before PCA/t-SNE/UMAP
    X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    return X


def run_baseline(
    X_mats: np.ndarray,
    y: np.ndarray,
    method: str,
    out_dir: str,
    dataset_name: str,
    plot: bool,
    label_names: Optional[Dict[int, str]] = None,
) -> Dict[str, Dict[str, float]]:
    X_flat = matrices_to_flat_features(X_mats)

    # Pick embedding dimensionality to 3 to match PBP small d
    if method == "pca":
        emb = PCA(n_components=3, random_state=0).fit_transform(X_flat)
    elif method == "tsne":
        emb = TSNE(n_components=3, random_state=0, init="pca", learning_rate="auto").fit_transform(X_flat)
    elif method == "umap":
        if not HAVE_UMAP:
            raise RuntimeError("umap-learn not installed. Please pip install umap-learn.")
        reducer = umap.UMAP(n_components=3, random_state=0)
        emb = reducer.fit_transform(X_flat)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Cluster and supervised metrics
    # n_clusters from labels
    n_clusters = len(set(int(v) for v in y))
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    pred = km.fit_predict(emb)
    metrics = calculate_all_metrics(emb, y, pred, km, cv_splits=3)

    if plot:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_png = str(Path(out_dir) / f"{dataset_name}_{method}_3d.png")
        scatter_features(
            emb, y, out_png,
            title=f"{dataset_name} {method.upper()} (3D)",
            label_names=label_names,
        )

    return metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Run PCA/t-SNE/UMAP baselines with PBP metrics")
    p.add_argument("--dataset", required=True, choices=sorted(SUPPORTED.keys()))
    p.add_argument("--method", required=True, choices=["pca", "tsne", "umap"]) 
    p.add_argument("--results-dir", default="./baseline_results")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--no-plot", dest="plot", action="store_false")
    p.set_defaults(plot=True)
    return p


def main() -> int:
    args = build_parser().parse_args()
    loader = SUPPORTED[args.dataset]

    # Load matrices normally (as used in PBP runners)
    import inspect
    sig = inspect.signature(loader)
    if "data_dir" in sig.parameters:
        X_mats, y, meta = loader()
    else:
        X_mats, y, meta = loader()
    label_names = meta.get("labels_map") if isinstance(meta, dict) else None

    metrics = run_baseline(
        X_mats=X_mats,
        y=y,
        method=args.method,
        out_dir=args.results_dir,
        dataset_name=args.dataset,
        plot=args.plot,
        label_names=label_names,
    )

    # Print in the same format as PBP results
    cluster = metrics["cluster"]
    supervised = metrics["supervised"]
    print(f"Metrics ({args.dataset}, {args.method.upper()})")
    print(f"- n_samples={X_mats.shape[0]}, n_features=3, n_clusters={len(set(int(v) for v in y))}")
    print(f"- v_measure={cluster.get('v_measure', np.nan):.4f}, adjusted_rand={cluster.get('adjusted_rand', np.nan):.4f}")
    print(f"- silhouette={cluster.get('silhouette', np.nan):.4f}, calinski_harabasz={cluster.get('calinski_harabasz', np.nan):.4f}, davies_bouldin={cluster.get('davies_bouldin', np.nan):.4f}, inertia={cluster.get('inertia', np.nan):.4f}")
    print(f"- linear_sep_cv={supervised.get('linear_sep_cv', np.nan):.4f}, cv_score={supervised.get('cv_score', np.nan):.4f}, margin_score={supervised.get('margin_score', np.nan):.4f}, boundary_complexity={supervised.get('boundary_complexity', np.nan):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


