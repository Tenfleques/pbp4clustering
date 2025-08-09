#!/usr/bin/env python3
from typing import Tuple, Dict, Any
from pathlib import Path
import io

import numpy as np
import pandas as pd
import requests


def load_parkinsons_matrices(data_dir: str = "./data/parkinsons") -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load UCI Parkinson's dataset and build (N, 2, 11) matrices by splitting 22 numeric features into two rows.

    Excludes 'name' and 'status' columns. Labels are 'status' (0 healthy, 1 Parkinson's).
    """
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(data_dir) / "parkinsons.csv"
    if not csv_path.exists():
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        tmp = pd.read_csv(io.StringIO(resp.text))
        tmp.to_csv(csv_path, index=False)

    df = pd.read_csv(csv_path)
    if "status" not in df.columns:
        raise ValueError("parkinsons.csv missing 'status' column")

    y = df["status"].astype(int).to_numpy()
    # Drop non-feature columns
    Xdf = df.drop(columns=[c for c in ["name", "status"] if c in df.columns])
    X = Xdf.to_numpy(dtype=np.float32)
    N, F = X.shape
    if F % 2 != 0:
        # If odd, drop last column to make even split
        X = X[:, :F-1]
        F = X.shape[1]
    half = F // 2
    X_mats = np.empty((N, 2, half), dtype=np.float32)
    X_mats[:, 0, :] = X[:, :half]
    X_mats[:, 1, :] = X[:, half:half*2]

    meta: Dict[str, Any] = {
        "n_samples": int(N),
        "matrix_shape": (2, int(half)),
        "rows": ["group1", "group2"],
        "cols": [f"f{i+1}" for i in range(half)],
        "labels_map": {0: "healthy", 1: "parkinsons"},
    }
    return X_mats, y, meta


