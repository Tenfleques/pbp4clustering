#!/usr/bin/env python3
from typing import Tuple, Dict, Any
from pathlib import Path
import io

import numpy as np
import pandas as pd
import requests


def load_seeds_matrices(data_dir: str = "./data/seeds") -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load UCI Seeds dataset from a local CSV and build (N, 3, 2) matrices.

    Expected CSV columns (order-insensitive if header present):
      area, perimeter, compactness, length, width, asymmetry, groove_length, class

    Rows: [dimensions(length,width), shape(area,perimeter), surface(asymmetry, groove_length)]
    Labels: seed class (1,2,3) mapped to 0..2
    """
    csv_path = Path(data_dir) / "seeds.csv"
    if not csv_path.exists():
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        # UCI Seeds dataset (Komań Wheat) direct data has whitespace-separated values, no header
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        tmp = pd.read_csv(io.StringIO(resp.text), header=None, sep="\s+")
        tmp.columns = [
            "area", "perimeter", "compactness", "length", "width",
            "asymmetry", "groove_length", "class",
        ]
        tmp.to_csv(csv_path, index=False)

    df = pd.read_csv(csv_path)
    # Try to normalize column names
    cols = {c.lower().strip().replace(" ", "_") for c in df.columns}
    # If no header, re-read without header and assign standard names
    if not {"area", "perimeter", "compactness", "length", "width", "asymmetry", "groove_length", "class"}.issubset(cols):
        df = pd.read_csv(csv_path, header=None, delim_whitespace=False)
        df.columns = [
            "area", "perimeter", "compactness", "length", "width",
            "asymmetry", "groove_length", "class",
        ]

    # Ensure correct order
    X = df[["area", "perimeter", "compactness", "length", "width", "asymmetry", "groove_length"]].to_numpy(dtype=np.float32)
    y_raw = df["class"].to_numpy()
    # Map classes to 0..k-1
    classes = sorted(np.unique(y_raw))
    class_to_id = {int(c): i for i, c in enumerate(classes)}
    y = np.array([class_to_id[int(v)] for v in y_raw], dtype=int)

    N = X.shape[0]
    X_mats = np.empty((N, 3, 2), dtype=np.float32)
    # Row 0: dimensions (length, width)
    X_mats[:, 0, 0] = X[:, 3]
    X_mats[:, 0, 1] = X[:, 4]
    # Row 1: shape (area, perimeter)
    X_mats[:, 1, 0] = X[:, 0]
    X_mats[:, 1, 1] = X[:, 1]
    # Row 2: surface (asymmetry, groove_length)
    X_mats[:, 2, 0] = X[:, 5]
    X_mats[:, 2, 1] = X[:, 6]

    meta: Dict[str, Any] = {
        "n_samples": int(N),
        "matrix_shape": (3, 2),
        "rows": ["dimensions", "shape", "surface"],
        "cols": ["feature_1", "feature_2"],
        "labels_map": {v: k for k, v in class_to_id.items()},
    }
    return X_mats, y, meta


if __name__ == "__main__":
    X_mats, y, meta = load_seeds_matrices()
    print(X_mats.shape)
    print(y.shape)
    print(meta)