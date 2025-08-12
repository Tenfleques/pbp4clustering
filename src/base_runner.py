"""
Base runner class for standardized clustering experiments.
Provides a template for all dataset-specific runners.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
import hashlib
import json
import pickle
import numpy as np
from pathlib import Path

from pbp_transform import matrices_to_pbp_vectors
from visualize import scatter_features
from .pipeline import filter_zero_columns, cluster_and_predict
from .metrics import calculate_all_metrics
from .utils import print_metrics_summary


class BaseRunner(ABC):
    """
    Abstract base class for clustering experiment runners.
    """
    
    def __init__(self, dataset_name: str):
        """
        Initialize the base runner.
        
        Args:
            dataset_name: Name of the dataset
        """
        self.dataset_name = dataset_name
    
    @abstractmethod
    def load_data(self, args) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Load dataset and return matrices, labels, and metadata.
        
        Args:
            args: Command-line arguments
            
        Returns:
            Tuple of (X_matrices, y_labels, metadata_dict)
        """
        raise NotImplementedError("Subclasses must implement load_data method")
    
    def preprocess_matrices(self, X_matrices: np.ndarray, args) -> np.ndarray:
        """
        Optional preprocessing step for matrices before PBP transform.
        Override in subclasses if needed.
        
        Args:
            X_matrices: Input matrices
            args: Command-line arguments
            
        Returns:
            Preprocessed matrices
        """
        return X_matrices
    
    def run(self, args):
        """
        Run the complete clustering pipeline.
        
        Args:
            args: Command-line arguments with at least:
                - agg: Aggregation function name
                - results_dir: Output directory
                - plot: Whether to generate plots
                - cv_splits: Number of CV splits
        """
        # Load data
        X_matrices, y, metadata = self.load_data(args)
        print(f"Loaded {self.dataset_name}: matrices={X_matrices.shape}, labels={y.shape}")
        if "matrix_shape" in metadata:
            print(f"  Matrix shape: {metadata['matrix_shape']}")

        # Cache setup: key derived from dataset + args
        try:
            args_dict = dict(vars(args))
        except Exception:
            args_dict = {}
        # Include dataset name for disambiguation
        cache_fingerprint = {
            "dataset": self.dataset_name,
            "args": args_dict,
        }
        cache_key_raw = json.dumps(cache_fingerprint, sort_keys=True, default=str)
        cache_key = hashlib.sha1(cache_key_raw.encode("utf-8")).hexdigest()
        results_dir = Path(getattr(args, "results_dir", "./results"))
        cache_dir = results_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{self.dataset_name}_{cache_key}.pkl"

        if cache_path.exists():
            with cache_path.open("rb") as f:
                cached = pickle.load(f)
            X_pbp = cached["X_pbp"]
            pred = cached["predictions"]
            all_metrics = cached["metrics"]
            metadata_cached = cached.get("metadata", metadata)
            # Regenerate plot if requested
            if getattr(args, "plot", True):
                out_png = str(results_dir / f"{self.dataset_name}_targets_pbp_{args.agg}.png")
                title = f"{self.dataset_name.title()} PBP (agg={args.agg})"
                if "labels_map" in metadata_cached:
                    title += " - " + metadata_cached.get("title_suffix", "Classes")
                scatter_features(
                    X_pbp, y, out_png,
                    title=title,
                    label_names=metadata_cached.get("labels_map")
                )
                print(f"Saved: {out_png}")
            print(f"Loaded cached results: {cache_path}")
            n_clusters = len(set(int(v) for v in y))
            print_metrics_summary(
                all_metrics.get("cluster", {}),
                all_metrics.get("supervised", {}),
                {
                    "dataset_name": self.dataset_name.title(),
                    "agg_func": args.agg,
                    "n_samples": X_pbp.shape[0],
                    "n_features": X_pbp.shape[1],
                    "n_clusters": n_clusters,
                },
            )
            return {
                "X_pbp": X_pbp,
                "predictions": pred,
                "metrics": all_metrics,
                "metadata": metadata_cached,
            }
        
        # Optional preprocessing
        X_matrices = self.preprocess_matrices(X_matrices, args)
        
        # PBP transformation
        sort_arg = getattr(args, "sort", None)
        X_pbp = matrices_to_pbp_vectors(X_matrices, agg=args.agg, sort=sort_arg)
        X_pbp = np.asarray(X_pbp)
        print(f"PBP vectors: {X_pbp.shape} (agg={args.agg})")
        
        # Filter zero columns
        X_pbp = filter_zero_columns(X_pbp)
        
        # Visualization
        if args.plot:
            out_png = str(Path(args.results_dir) / f"{self.dataset_name}_targets_pbp_{args.agg}.png")
            title = f"{self.dataset_name.title()} PBP (agg={args.agg})"
            if "labels_map" in metadata:
                title += " - " + metadata.get("title_suffix", "Classes")
            
            scatter_features(
                X_pbp, y, out_png,
                title=title,
                label_names=metadata.get("labels_map")
            )
            print(f"Saved: {out_png}")
        
        # Clustering
        n_clusters = len(set(int(v) for v in y))
        pred, km = cluster_and_predict(X_pbp, n_clusters=n_clusters)
        
        # Calculate metrics
        cv_splits = getattr(args, "cv_splits", 3)
        all_metrics = calculate_all_metrics(X_pbp, y, pred, km, cv_splits)
        
        # Print results
        print_metrics_summary(
            all_metrics["cluster"],
            all_metrics["supervised"],
            {
                "dataset_name": self.dataset_name.title(),
                "agg_func": args.agg,
                "n_samples": X_pbp.shape[0],
                "n_features": X_pbp.shape[1],
                "n_clusters": n_clusters
            }
        )

        # Save cache
        cache_payload = {
            "X_pbp": X_pbp,
            "predictions": pred,
            "metrics": all_metrics,
            "metadata": metadata,
        }
        try:
            with cache_path.open("wb") as f:
                pickle.dump(cache_payload, f)
            print(f"Saved cache: {cache_path}")
        except Exception:
            pass
        
        return {
            "X_pbp": X_pbp,
            "predictions": pred,
            "metrics": all_metrics,
            "metadata": metadata
        }


class SimpleRunner(BaseRunner):
    """
    Simple runner for datasets with straightforward loading.
    """
    
    def __init__(self, dataset_name: str, loader_func):
        """
        Initialize simple runner with a loader function.
        
        Args:
            dataset_name: Name of the dataset
            loader_func: Function to load the data
        """
        super().__init__(dataset_name)
        self.loader_func = loader_func
    
    def load_data(self, args) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Load data using the provided loader function.
        """
        # Check if loader needs data_dir argument
        import inspect
        sig = inspect.signature(self.loader_func)
        
        if "data_dir" in sig.parameters and hasattr(args, "data_dir"):
            return self.loader_func(data_dir=args.data_dir)
        elif len(sig.parameters) == 0:
            return self.loader_func()
        else:
            # Try calling with no arguments
            return self.loader_func()