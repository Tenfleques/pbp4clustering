#!/usr/bin/env python3
from typing import Tuple, Dict, Any
from pathlib import Path
import io

import numpy as np
import pandas as pd
import requests


def load_ionosphere_matrices(data_dir: str = "./data/ionosphere") -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load UCI Ionosphere dataset and build (N, 2, 17) matrices by splitting 34 attributes into two pulses.
    Labels: 'g' (good) / 'b' (bad) → 1/0
    """
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(data_dir) / "ionosphere.csv"
    if not csv_path.exists():
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        tmp = pd.read_csv(io.StringIO(resp.text), header=None)
        tmp.to_csv(csv_path, index=False)

    df = pd.read_csv(csv_path, header=None)
    X = df.iloc[:, :-1].to_numpy(dtype=np.float32)
    y_raw = df.iloc[:, -1].astype(str).to_numpy()
    y = np.array([1 if v == 'g' else 0 for v in y_raw], dtype=int)

    # Some columns can be all zero; keep first 34 features for split
    F = min(34, X.shape[1])
    X = X[:, :F]
    if F % 2 == 1:
        F -= 1
        X = X[:, :F]
    half = F // 2

    N = X.shape[0]
    X_mats = np.empty((N, 2, half), dtype=np.float32)
    X_mats[:, 0, :] = X[:, :half]
    X_mats[:, 1, :] = X[:, half:half*2]

    meta: Dict[str, Any] = {
        "n_samples": int(N),
        "matrix_shape": (2, int(half)),
        "rows": ["pulse1", "pulse2"],
        "cols": [f"a{i+1}" for i in range(half)],
        "labels_map": {0: "bad", 1: "good"},
    }
    return X_mats, y, meta


