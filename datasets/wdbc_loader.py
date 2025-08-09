#!/usr/bin/env python3
from typing import Tuple, Dict, Any

import numpy as np
from sklearn.datasets import load_breast_cancer


def load_wdbc_matrices() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load Breast Cancer Wisconsin (Diagnostic) and build (N, 3, 10) matrices.

    Rows: [mean, se, worst]
    Columns: 10 base features (radius, texture, perimeter, area, smoothness,
             compactness, concavity, concave_points, symmetry, fractal_dimension)
    """
    data = load_breast_cancer()
    X = data.data.astype(np.float32)  # shape (N, 30)
    y = data.target.astype(int)

    # Group into 3 rows × 10 columns
    N = X.shape[0]
    X_mats = X.reshape(N, 3, 10)

    labels_map = {i: name for i, name in enumerate(data.target_names)}
    meta: Dict[str, Any] = {
        "n_samples": int(N),
        "matrix_shape": (3, 10),
        "rows": ["mean", "se", "worst"],
        "cols": [
            "radius", "texture", "perimeter", "area", "smoothness",
            "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension",
        ],
        "labels_map": labels_map,
    }
    return X_mats, y, meta


