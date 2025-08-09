#!/usr/bin/env python3
from typing import Tuple, Dict, Any
from pathlib import Path
import io

import numpy as np
import pandas as pd
import requests


def load_banknote_matrices(data_dir: str = "./data/banknote") -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load Banknote Authentication and build (N, 2, 2) matrices.

    Rows: group1 (variance, skewness), group2 (kurtosis, entropy)
    Labels: class (0/1)
    """
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(data_dir) / "banknote.csv"
    if not csv_path.exists():
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        tmp = pd.read_csv(io.StringIO(resp.text), header=None)
        tmp.columns = ["variance", "skewness", "kurtosis", "entropy", "class"]
        tmp.to_csv(csv_path, index=False)

    df = pd.read_csv(csv_path)
    y = df["class"].astype(int).to_numpy()
    N = df.shape[0]
    X_mats = np.empty((N, 2, 2), dtype=np.float32)
    X_mats[:, 0, 0] = df["variance"].to_numpy(np.float32)
    X_mats[:, 0, 1] = df["skewness"].to_numpy(np.float32)
    X_mats[:, 1, 0] = df["kurtosis"].to_numpy(np.float32)
    X_mats[:, 1, 1] = df["entropy"].to_numpy(np.float32)

    meta: Dict[str, Any] = {
        "n_samples": int(N),
        "matrix_shape": (2, 2),
        "rows": ["group1", "group2"],
        "cols": ["f1", "f2"],
        "labels_map": {0: "authentic", 1: "forgery"},
    }
    return X_mats, y, meta


