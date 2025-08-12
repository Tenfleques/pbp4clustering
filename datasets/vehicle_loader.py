#!/usr/bin/env python3
from typing import Tuple, Dict, Any
from pathlib import Path
import io

import numpy as np
import pandas as pd
import requests


def load_vehicle_matrices(
    data_dir: str = "./data/vehicle",
    mode: str = "2x9",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load Statlog (Vehicle Silhouettes) and build per-sample matrices.

    - mode="2x9" (default): rows = [first 9 features, last 9 features], shape (N, 2, 9)
    - mode="3x6": rows = 3 equal groups of 6 features, shape (N, 3, 6)

    Labels: {van, saab, bus, opel}
    """
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    raw_path = Path(data_dir) / "vehicle.dat"
    csv_path = Path(data_dir) / "vehicle.csv"

    if not csv_path.exists():
        # Try primary URL; if 404, fallback mirror
        content = None
        urls = [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xaa.dat",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xab.dat",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xac.dat",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xad.dat",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/vehicle.dat",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/vehicle/vehicle.dat",

        ]
        for url in urls:
            try:
                resp = requests.get(url, timeout=120)
                if resp.status_code == 200 and resp.text.strip():
                    content = resp.text
                    break
            except Exception:
                continue
        if content is None:
            raise FileNotFoundError("Could not download vehicle.dat from UCI; please place it manually under data/vehicle")

        tmp = pd.read_csv(io.StringIO(content), header=None, delim_whitespace=True)
        if tmp.shape[1] != 19:
            tmp = pd.read_csv(io.StringIO(content), header=None)
        cols = [f"f{i+1}" for i in range(tmp.shape[1] - 1)] + ["class"]
        tmp.columns = cols
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        tmp.to_csv(csv_path, index=False)

    df = pd.read_csv(csv_path)
    if "class" not in df.columns:
        # If saved without header earlier
        df.columns = [f"f{i+1}" for i in range(df.shape[1] - 1)] + ["class"]

    y_raw = df["class"].astype(str).to_numpy()
    classes = sorted(set(y_raw.tolist()))
    class_to_id = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_id[c] for c in y_raw], dtype=int)

    X = df.drop(columns=["class"]).to_numpy(dtype=np.float32)
    N, F = X.shape
    if mode == "2x9":
        if F < 18:
            raise ValueError(f"Expected at least 18 features for 2x9 mode, found {F}")
        X = X[:, :18]
        X_mats = np.empty((N, 2, 9), dtype=np.float32)
        X_mats[:, 0, :] = X[:, :9]
        X_mats[:, 1, :] = X[:, 9:18]
        shape = (2, 9)
        rows = ["group1", "group2"]
        cols = [f"c{i+1}" for i in range(9)]
    elif mode == "3x6":
        if F < 18:
            raise ValueError(f"Expected at least 18 features for 3x6 mode, found {F}")
        X = X[:, :18]
        X_mats = np.empty((N, 3, 6), dtype=np.float32)
        X_mats[:, 0, :] = X[:, 0:6]
        X_mats[:, 1, :] = X[:, 6:12]
        X_mats[:, 2, :] = X[:, 12:18]
        shape = (3, 6)
        rows = ["group1", "group2", "group3"]
        cols = [f"c{i+1}" for i in range(6)]
    else:
        raise ValueError("mode must be one of: '2x9', '3x6'")

    meta: Dict[str, Any] = {
        "n_samples": int(N),
        "matrix_shape": shape,
        "rows": rows,
        "cols": cols,
        "labels_map": {i: c for c, i in class_to_id.items()},
    }
    return X_mats, y, meta


