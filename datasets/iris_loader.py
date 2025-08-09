#!/usr/bin/env python3
from typing import Tuple, Dict, Any

import numpy as np
from sklearn import datasets


def load_iris_matrices() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load Iris and build per-sample matrices with rows (Sepal, Petal) and columns (Length, Width).

    Output shape: (N, 2, 2)
    Labels: species
    """
    data = datasets.load_iris()
    X = data.data.astype(np.float32)  # columns: sepal length, sepal width, petal length, petal width
    y = data.target.astype(int)

    # Build (N, 2, 2): rows [Sepal, Petal], cols [Length, Width]
    N = X.shape[0]
    X_mats = np.empty((N, 2, 2), dtype=np.float32)
    # Sepal: length (col 0), width (col 1)
    X_mats[:, 0, 0] = X[:, 0]
    X_mats[:, 0, 1] = X[:, 1]
    # Petal: length (col 2), width (col 3)
    X_mats[:, 1, 0] = X[:, 2]
    X_mats[:, 1, 1] = X[:, 3]

    labels_map = {i: name for i, name in enumerate(data.target_names)}
    meta: Dict[str, Any] = {
        "n_samples": int(N),
        "matrix_shape": (2, 2),
        "rows": ["Sepal", "Petal"],
        "cols": ["Length", "Width"],
        "labels_map": labels_map,
    }
    return X_mats, y, meta


