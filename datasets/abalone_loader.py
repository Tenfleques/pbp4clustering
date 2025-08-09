#!/usr/bin/env python3
from typing import Tuple, Dict, Any
from pathlib import Path
import io

import numpy as np
import pandas as pd
import requests


def load_abalone_matrices(
    data_dir: str = "./data/abalone",
    option: str = "A",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load Abalone CSV and build matrices per the specified option.

    Expected columns: Sex, Length, Diameter, Height, Whole weight, Shucked weight,
                      Viscera weight, Shell weight, Rings

    Option A: (N, 3, 1) with rows [length, diameter, height]
    Option B: (N, 2, 4) with rows [physical dims length,diameter,height,rings] vs [weights whole,shucked,viscera,shell]
    Labels: rings (integer age proxy)
    """
    csv_path = Path(data_dir) / "abalone.csv"
    if not csv_path.exists():
        # Download from UCI and write a CSV with headers
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        # Build DataFrame from raw without header
        tmp = pd.read_csv(io.StringIO(resp.text), header=None)
        tmp.columns = [
            "Sex", "length", "diameter", "height",
            "whole_weight", "shucked_weight", "viscera_weight", "shell_weight",
            "rings",
        ]
        tmp.to_csv(csv_path, index=False)

    df = pd.read_csv(csv_path)
    # Normalize column names if necessary
    mapper = {
        "Length": "length", "Diameter": "diameter", "Height": "height",
        "Whole weight": "whole_weight", "Shucked weight": "shucked_weight",
        "Viscera weight": "viscera_weight", "Shell weight": "shell_weight",
        "Rings": "rings",
    }
    df = df.rename(columns=mapper)
    req = [
        "length", "diameter", "height",
        "whole_weight", "shucked_weight", "viscera_weight", "shell_weight",
        "rings",
    ]
    if not set(req).issubset(set(df.columns)):
        raise ValueError("abalone.csv missing required columns after renaming")

    y = df["rings"].to_numpy(dtype=int)

    if option.upper() == "A":
        X = df[["length", "diameter", "height"]].to_numpy(dtype=np.float32)
        N = X.shape[0]
        X_mats = X.reshape(N, 3, 1)
        meta = {
            "n_samples": int(N),
            "matrix_shape": (3, 1),
            "rows": ["length", "diameter", "height"],
            "cols": ["value"],
            "labels_map": None,
        }
    elif option.upper() == "B":
        dims = df[["length", "diameter", "height", "rings"]].to_numpy(dtype=np.float32)
        weights = df[["whole_weight", "shucked_weight", "viscera_weight", "shell_weight"]].to_numpy(dtype=np.float32)
        # Standardize within each group to comparable scales
        dims = (dims - dims.mean(axis=0)) / (dims.std(axis=0) + 1e-8)
        weights = (weights - weights.mean(axis=0)) / (weights.std(axis=0) + 1e-8)
        N = dims.shape[0]
        X_mats = np.empty((N, 2, 4), dtype=np.float32)
        X_mats[:, 0, :] = dims
        X_mats[:, 1, :] = weights
        meta = {
            "n_samples": int(N),
            "matrix_shape": (2, 4),
            "rows": ["dims", "weights"],
            "cols": ["c1", "c2", "c3", "c4"],
            "labels_map": None,
        }
    else:
        raise ValueError("option must be 'A' or 'B'")

    return X_mats, y, meta


