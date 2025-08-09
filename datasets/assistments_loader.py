from pathlib import Path
from typing import Tuple, Dict, Any
import io

import numpy as np
import pandas as pd
import requests

# Combined dataset link (as referenced by ASSISTments site may redirect; users might need to manually download if blocked)
ASSIST_COMBINED_URL = "https://drive.google.com/uc?export=download&id=0B2X0QD6q79ZJNEdiMHkyb0RNQlE"


def _download_csv(url: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        resp = requests.get(url, timeout=180)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
    return out_path


def load_assistments_metadata(data_dir: str = "./data/assistments") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Download ASSISTments 2009-10 combined dataset (CSV/TSV) and return raw DataFrame for further processing.
    """
    root = Path(data_dir)
    raw_path = root / "assistments_2009_2010_combined.csv"
    try:
        _download_csv(ASSIST_COMBINED_URL, raw_path)
    except Exception:
        # Fallback: note to user to manually place file
        pass

    # Try reading as CSV or TSV
    if raw_path.exists():
        try:
            df = pd.read_csv(raw_path)
        except Exception:
            df = pd.read_csv(raw_path, sep='\t')
    else:
        # Placeholder empty df if download failed (manual placement required)
        df = pd.DataFrame()

    meta = {
        "columns": list(df.columns),
        "n_rows": int(len(df)),
        "note": "If the download failed due to Google Drive restrictions, place the combined CSV at clustering/data/assistments/assistments_2009_2010_combined.csv",
    }
    return df, meta


def build_assistments_matrices(df: pd.DataFrame, max_skills: int = 64, time_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Construct (N, skills*time_bins, 1) matrices per student-session with target as correctness proportion or next-problem correctness.
    This is a placeholder implementation that expects columns like: user_id, skill, correct, timestamp.
    """
    if df.empty:
        return np.empty((0, 1, 1), dtype=np.float32), np.empty((0,), dtype=int), {"note": "empty df"}

    # Minimal, schematic transformation (user-level aggregates)
    # Select/rename expected columns if present
    user_col = next((c for c in df.columns if c.lower() in ("user_id", "student")), None)
    skill_col = next((c for c in df.columns if "skill" in c.lower()), None)
    correct_col = next((c for c in df.columns if "correct" in c.lower()), None)
    time_col = next((c for c in df.columns if "time" in c.lower()), None)

    if not all([user_col, skill_col, correct_col]):
        return np.empty((0, 1, 1), dtype=np.float32), np.empty((0,), dtype=int), {"note": "required columns not found"}

    # Limit to top skills
    top_skills = df[skill_col].value_counts().head(max_skills).index.tolist()
    df = df[df[skill_col].isin(top_skills)].copy()

    # Bin time if available
    if time_col is not None:
        df["_time_bin"] = pd.qcut(df[time_col].rank(method="first"), q=time_bins, labels=False, duplicates='drop')
    else:
        df["_time_bin"] = 0

    # Aggregate correctness per user-skill-bin
    pivot = df.pivot_table(index=[user_col], columns=[skill_col, "_time_bin"], values=correct_col, aggfunc='mean', fill_value=0.0)
    pivot = pivot.sort_index(axis=1)

    X_flat = pivot.to_numpy(dtype=np.float32)
    # Fill missing columns to a full grid (max_skills x time_bins)
    expected_cols = [(s, b) for s in top_skills for b in range(df["_time_bin"].max() + 1)]
    # Align if necessary (skipped for brevity)

    # Build matrices (features×1)
    X_mats = X_flat.reshape(X_flat.shape[0], X_flat.shape[1], 1)
    # Example target: overall avg correctness per user (bucketed)
    y_cont = df.groupby(user_col)[correct_col].mean().reindex(pivot.index).fillna(0.0).to_numpy()
    y = pd.qcut(y_cont, q=3, labels=False, duplicates='drop').astype(int)

    metadata = {
        "n_samples": int(X_mats.shape[0]),
        "matrix_shape": (int(X_mats.shape[1]), 1),
        "skills": top_skills,
        "time_bins": int(df["_time_bin"].max() + 1),
    }
    return X_mats, y, metadata 