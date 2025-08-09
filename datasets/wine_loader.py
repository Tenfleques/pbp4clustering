import io
import zipfile
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import requests

# UCI Wine Quality direct CSV URLs
WINE_RED_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
WINE_WHITE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"


def _download_csv(url: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
    return out_path


def _load_csv(path: Path) -> pd.DataFrame:
    # CSV uses semicolons as separators
    return pd.read_csv(path, sep=';')


def load_wine_quality(data_dir: str = "./data/wine", merge_types: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load UCI Wine Quality (red and white), build (N, 11, 1) matrices with true quality labels.

    Returns:
        X_mats: (N, 11, 1) float32 (features as rows, single column)
        y: (N,) integer quality scores
        metadata: dict with feature_names and counts
    """
    data_root = Path(data_dir)
    red_path = _download_csv(WINE_RED_URL, data_root / "winequality-red.csv")
    white_path = _download_csv(WINE_WHITE_URL, data_root / "winequality-white.csv")

    red_df = _load_csv(red_path)
    white_df = _load_csv(white_path)

    red_df["type"] = "red"
    white_df["type"] = "white"

    df = pd.concat([red_df, white_df], axis=0, ignore_index=True) if merge_types else red_df

    feature_cols = [c for c in df.columns if c not in ("quality", "type")]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["quality"].to_numpy(dtype=int)

    # Build matrices: features as rows, a single column per sample
    # Shape: (N, F, 1)
    X_mats = X.reshape(X.shape[0], X.shape[1], 1)

    metadata = {
        "n_samples": int(X_mats.shape[0]),
        "matrix_shape": (int(X_mats.shape[1]), int(X_mats.shape[2])),
        "feature_names": feature_cols,
        "label_name": "quality",
        "types": (df["type"].value_counts().to_dict() if merge_types else {"red": len(df)})
    }
    return X_mats, y, metadata 