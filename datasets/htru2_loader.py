#!/usr/bin/env python3
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from pathlib import Path


HTRU2_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00372/HTRU2.zip"


def load_htru2_matrices(data_dir: str = "./data/htru2") -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load HTRU2 and build (N, 2, 4) matrices: rows [Integrated_profile, DM_SNR_curve], cols [mean, std, skewness, kurtosis].
    """
    # Prefer local numpy copy if present; otherwise expect CSV named HTRU_2.csv in data_dir
    csv_path = Path(data_dir) / "HTRU_2.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected HTRU_2.csv at {csv_path}. Please download and place it there.")

    df = pd.read_csv(csv_path, header=None)
    # Columns 0-3: profile stats; 4-7: DM-SNR stats; 8: class (0/1)
    X = df.iloc[:, 0:8].to_numpy(dtype=np.float32)
    y = df.iloc[:, 8].to_numpy(dtype=int)

    N = X.shape[0]
    X_mats = np.empty((N, 2, 4), dtype=np.float32)
    X_mats[:, 0, :] = X[:, 0:4]
    X_mats[:, 1, :] = X[:, 4:8]

    meta: Dict[str, Any] = {
        "n_samples": int(N),
        "matrix_shape": (2, 4),
        "rows": ["Integrated_profile", "DM_SNR_curve"],
        "cols": ["mean", "std", "skewness", "kurtosis"],
        "labels_map": {0: "non_pulsar", 1: "pulsar"},
    }
    return X_mats, y, meta


if __name__ == "__main__":
    X_mats, y, meta = load_htru2_matrices()
    print(X_mats.shape)
    print(y.shape)
    print(meta)