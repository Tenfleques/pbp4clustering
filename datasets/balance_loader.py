#!/usr/bin/env python3
from typing import Tuple, Dict, Any
from pathlib import Path
import io

import numpy as np
import pandas as pd
import requests


def load_balance_matrices(data_dir: str = "./data/balance") -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load Balance Scale dataset and build (N, 2, 2) matrices.

    Rows: Left, Right
    Cols: (weight, distance)
    Labels: class (L, B, R) mapped to 0..2
    """
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(data_dir) / "balance.csv"
    if not csv_path.exists():
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        tmp = pd.read_csv(io.StringIO(resp.text), header=None)
        tmp.columns = ["class", "l_weight", "l_distance", "r_weight", "r_distance"]
        tmp.to_csv(csv_path, index=False)

    df = pd.read_csv(csv_path)
    y_raw = df["class"].astype(str).to_numpy()
    classes = {"L": 0, "B": 1, "R": 2}
    y = np.array([classes[c] for c in y_raw], dtype=int)

    N = df.shape[0]
    X_mats = np.empty((N, 2, 2), dtype=np.float32)
    X_mats[:, 0, 0] = df["l_weight"].to_numpy(np.float32)
    X_mats[:, 0, 1] = df["l_distance"].to_numpy(np.float32)
    X_mats[:, 1, 0] = df["r_weight"].to_numpy(np.float32)
    X_mats[:, 1, 1] = df["r_distance"].to_numpy(np.float32)

    meta: Dict[str, Any] = {
        "n_samples": int(N),
        "matrix_shape": (2, 2),
        "rows": ["left", "right"],
        "cols": ["weight", "distance"],
        "labels_map": {0: "L", 1: "B", 2: "R"},
    }
    return X_mats, y, meta


