#!/usr/bin/env python3
from typing import Tuple, Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd
import requests


def load_penguins_matrices(data_dir: str = "./data/penguins") -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load Palmer Penguins CSV and build (N, 2, 2) matrices.

    Expected columns: species, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g

    Rows: [Bill, Body]
    Cols: [Bill: length, depth], [Body: flipper_length, body_mass] (after standardization to comparable units)
    """
    csv_path = Path(data_dir) / "penguins.csv"
    if not csv_path.exists():
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        # Use palmerpenguins GitHub CSV as a reliable source
        url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with open(csv_path, "wb") as f:
            f.write(resp.content)

    df = pd.read_csv(csv_path)
    req = ["species", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    if not set(req).issubset(set(df.columns)):
        raise ValueError(f"penguins.csv missing required columns; found: {df.columns}")

    # Drop rows with missing
    df = df.dropna(subset=req)

    y_raw = df["species"].astype(str).to_numpy()
    classes = sorted(set(y_raw.tolist()))
    class_to_id = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_id[s] for s in y_raw], dtype=int)

    # Standardize body pair to be comparable with bill pair (simple z-score per column)
    cols_num = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    Z = df[cols_num].to_numpy(dtype=np.float32)
    Z = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-8)

    N = Z.shape[0]
    X_mats = np.empty((N, 2, 2), dtype=np.float32)
    # Row 0: Bill (length, depth)
    X_mats[:, 0, 0] = Z[:, 0]
    X_mats[:, 0, 1] = Z[:, 1]
    # Row 1: Body (flipper_length, body_mass)
    X_mats[:, 1, 0] = Z[:, 2]
    X_mats[:, 1, 1] = Z[:, 3]

    meta: Dict[str, Any] = {
        "n_samples": int(N),
        "matrix_shape": (2, 2),
        "rows": ["Bill", "Body"],
        "cols": ["length", "depth"],
        "labels_map": {i: c for c, i in class_to_id.items()},
    }
    return X_mats, y, meta


