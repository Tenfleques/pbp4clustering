#!/usr/bin/env python3
from typing import Tuple, Dict, Any
from pathlib import Path
import io

import numpy as np
import pandas as pd
import requests


def load_sonar_matrices(data_dir: str = "./data/sonar") -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load UCI Sonar dataset and build (N, 2, 30) matrices by splitting 60 features into two frequency halves.
    Labels: 'R'/'M' → 0/1
    """
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(data_dir) / "sonar.csv"
    if not csv_path.exists():
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        tmp = pd.read_csv(io.StringIO(resp.text), header=None)
        tmp.to_csv(csv_path, index=False)

    df = pd.read_csv(csv_path, header=None)
    X = df.iloc[:, :-1].to_numpy(dtype=np.float32)
    y_raw = df.iloc[:, -1].astype(str).to_numpy()
    y = np.array([0 if v == 'R' else 1 for v in y_raw], dtype=int)

    N, F = X.shape  # F should be 60
    assert F >= 60, "Unexpected Sonar feature count"
    X = X[:, :60]
    X_mats = np.empty((N, 2, 30), dtype=np.float32)
    X_mats[:, 0, :] = X[:, :30]
    X_mats[:, 1, :] = X[:, 30:60]

    meta: Dict[str, Any] = {
        "n_samples": int(N),
        "matrix_shape": (2, 30),
        "rows": ["low_freq", "high_freq"],
        "cols": [f"b{i+1}" for i in range(30)],
        "labels_map": {0: "rock", 1: "mine"},
    }
    return X_mats, y, meta


