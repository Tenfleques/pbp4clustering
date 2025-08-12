#!/usr/bin/env python3
"""
Refactored Retail dataset runner with complex preprocessing.
"""

from typing import Dict
from src.base_runner import BaseRunner
from src.cli_args import get_base_parser, add_data_dir_arg
from datasets.online_retail_loader import load_online_retail_matrices
import numpy as np


class RetailRunner(BaseRunner):
    """Custom runner for Retail dataset with complex preprocessing options."""
    
    def __init__(self):
        super().__init__("retail")
    
    def load_data(self, args):
        X_mats, y, meta = load_online_retail_matrices(args.data_dir)
        return X_mats, y, meta
    
    def preprocess_matrices(self, X_matrices, args):
        X_mats = X_matrices.copy()
        
        # Optional: aggregate months into larger windows
        if args.months_agg is not None:
            N, m, n = X_mats.shape
            b = int(args.months_agg)
            if n % b == 0:
                X_mats = X_mats.reshape(N, m, n // b, b).mean(axis=3)
                print(f"Aggregated months by {b}: new shape={X_mats.shape}")
                if args.rows_from_months:
                    X_mats = np.swapaxes(X_mats, 1, 2)
                    print(f"Transposed to rows-from-months: shape={X_mats.shape}")
            else:
                print(f"Warning: months ({n}) not divisible by {b}; skipping months aggregation")
        
        # Optional: PCA reduction on rows
        if args.rows is not None:
            from sklearn.decomposition import PCA
            N, m, n = X_mats.shape
            target_m = int(args.rows)
            if target_m < m:
                X_flat = X_mats.reshape(N, m * n)
                pca = PCA(n_components=target_m, random_state=0)
                X_reduced = pca.fit_transform(X_flat.T).T
                X_mats = X_reduced.reshape(N, target_m, n)
                print(f"Reduced rows from {m} to {target_m} via PCA: shape={X_mats.shape}")
        
        return X_mats
    
    def run(self, args):
        # Load and preprocess data
        X_matrices, y_orig, metadata = self.load_data(args)
        print(f"Loaded {self.dataset_name}: matrices={X_matrices.shape}, labels={y_orig.shape}")
        if "matrix_shape" in metadata:
            print(f"  Matrix shape: {metadata['matrix_shape']}")
        
        # Handle top-K countries filtering
        y = y_orig.copy()
        if args.top_k_countries is not None and args.top_k_countries > 0:
            k = int(args.top_k_countries)
            counts = np.bincount(y)
            top_ids = np.argsort(counts)[::-1][:k]
            remap: Dict[int, int] = {int(cid): i for i, cid in enumerate(top_ids)}
            other_id_new = k
            y = np.array([remap.get(int(lbl), other_id_new) for lbl in y], dtype=int)
            
            # Update labels for display
            id_to_country = {v: k for k, v in (metadata.get("label_map") or {}).items()}
            kept = [id_to_country.get(int(cid), str(cid)) for cid in top_ids]
            kept.append("Other")
            metadata["labels_map"] = {i: name for i, name in enumerate(kept)}
            print(f"Kept top {k} countries: {kept[:-1]}")
        
        # Apply preprocessing
        X_matrices = self.preprocess_matrices(X_matrices, args)
        
        # Continue with standard pipeline
        from pbp_transform import matrices_to_pbp_vectors
        from src.pipeline import filter_zero_columns, cluster_and_predict
        from src.metrics import calculate_all_metrics
        from src.utils import print_metrics_summary
        from visualize import scatter_features
        from pathlib import Path
        
        X_pbp = matrices_to_pbp_vectors(X_matrices, agg=args.agg)
        X_pbp = np.asarray(X_pbp)
        print(f"PBP vectors: {X_pbp.shape} (agg={args.agg})")
        
        X_pbp = filter_zero_columns(X_pbp)
        
        if args.plot:
            out_png = str(Path(args.results_dir) / f"retail_targets_pbp_{args.agg}.png")
            title = f"Retail PBP (agg={args.agg}) - Countries"
            scatter_features(X_pbp, y, out_png, title=title, label_names=metadata.get("labels_map"))
            print(f"Saved: {out_png}")
        
        n_clusters = len(set(int(v) for v in y))
        pred, km = cluster_and_predict(X_pbp, n_clusters=n_clusters)
        
        cv_splits = getattr(args, "cv_splits", 3)
        all_metrics = calculate_all_metrics(X_pbp, y, pred, km, cv_splits)
        
        print_metrics_summary(
            all_metrics["cluster"],
            all_metrics["supervised"],
            {
                "dataset_name": "Retail",
                "agg_func": args.agg,
                "n_samples": X_pbp.shape[0],
                "n_features": X_pbp.shape[1],
                "n_clusters": n_clusters
            }
        )


def main():
    parser = get_base_parser(
        description="Run PBP on UCI Online Retail II per-customer matrices and report metrics."
    )
    parser = add_data_dir_arg(parser, default="./data/retail")
    parser.add_argument("--rows", type=int, default=None,
                        help="Optionally reduce rows (m) to this value via PCA before PBP")
    parser.add_argument("--top-k-countries", type=int, default=6,
                        help="Keep K most frequent countries; map others to 'Other'")
    parser.add_argument("--months-agg", type=int, choices=[4, 6], default=4,
                        help="Aggregate months into blocks of this size (mean)")
    parser.add_argument("--rows-from-months", action="store_true",
                        help="After months aggregation, transpose so rows = aggregated months")
    parser.set_defaults(cv_splits=3)  # Retail uses 3 CV splits
    args = parser.parse_args()
    
    runner = RetailRunner()
    runner.run(args)


if __name__ == "__main__":
    main()