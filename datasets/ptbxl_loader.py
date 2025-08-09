from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import json
import ast

import numpy as np
import pandas as pd
import requests

PTBXL_BASE = "https://physionet.org/files/ptb-xl/1.0.3/"


def _download_file(url: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
    return out_path


def download_ptbxl_metadata(data_dir: str = "./data/ptbxl") -> Dict[str, str]:
    """
    Download PTB-XL metadata CSV files (database and statements).
    """
    root = Path(data_dir)
    db_csv = _download_file(PTBXL_BASE + "ptbxl_database.csv", root / "ptbxl_database.csv")
    scp_csv = _download_file(PTBXL_BASE + "scp_statements.csv", root / "scp_statements.csv")
    return {"ptbxl_database": str(db_csv), "scp_statements": str(scp_csv)}


def _load_superclass_map(scp_statements_csv: Path) -> Dict[str, str]:
    scp = pd.read_csv(scp_statements_csv)
    scp = scp.set_index("Unnamed: 0")
    return scp["diagnostic_class"].dropna().to_dict()


def _extract_superclass_labels(scp_codes_str: str, scp_to_super: Dict[str, str]) -> List[str]:
    try:
        codes = ast.literal_eval(scp_codes_str)
    except Exception:
        return []
    labels: List[str] = []
    for code in codes.keys():
        super_c = scp_to_super.get(code)
        if super_c:
            labels.append(super_c)
    return sorted(list(set(labels)))


def load_ptbxl_metadata_only(
    data_dir: str = "./data/ptbxl",
    target_superclass: bool = True,
    folds: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load PTB-XL metadata and prepare a frame with filenames and labels.

    Returns a DataFrame with columns: filename_lr, filename_hr, y_labels (list of superclasses or scp_codes)
    """
    paths = download_ptbxl_metadata(data_dir)
    db = pd.read_csv(paths["ptbxl_database"])  # columns include filename_hr, filename_lr, scp_codes, strat_fold

    meta: Dict[str, Any] = {}
    if target_superclass:
        scp_map = _load_superclass_map(Path(paths["scp_statements"]))
        db["y_labels"] = db["scp_codes"].apply(lambda s: _extract_superclass_labels(s, scp_map))
        meta["label_type"] = "superclass"
    else:
        db["y_labels"] = db["scp_codes"].apply(lambda s: list(ast.literal_eval(s).keys()))
        meta["label_type"] = "scp_code"

    if folds:
        db = db[db["strat_fold"].isin(folds)].reset_index(drop=True)
        meta["folds"] = folds

    meta["n_records"] = int(len(db))
    meta["paths"] = paths
    return db, meta


def load_ptbxl_waveform_matrices(
    db: pd.DataFrame,
    data_dir: str = "./data/ptbxl",
    leads: Optional[List[str]] = None,
    sample_rate: int = 100,
    window_samples: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load WFDB waveforms for given records into matrices of shape (N, L, T) with labels.

    Requires the WFDB package and that waveform files are available locally under data_dir/records{100,500}/...
    """
    try:
        import wfdb  # type: ignore
        from wfdb import processing  # noqa: F401
    except Exception as e:
        raise ImportError("wfdb is required to load PTB-XL waveforms. Install with: pip install wfdb") from e

    # Choose lead names to extract
    if leads is None:
        leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    # Build label map from unique superclass combinations
    unique_labelsets = sorted(set(tuple(lbls) for lbls in db["y_labels"]))
    label_to_id = {lab: i for i, lab in enumerate(unique_labelsets)}

    for _, row in db.iterrows():
        # Prefer low-rate files for speed
        rel = row.get("filename_lr") or row.get("filename_hr")
        if not rel:
            continue
        record_path = Path(data_dir) / rel
        # WFDB expects path without extension
        rec_no_ext = str(record_path).rsplit(".", 1)[0]
        sig, fields = wfdb.rdsamp(rec_no_ext)
        # sig: (T, num_channels), fields["sig_name"] is channel names
        sig_names = list(fields.get("sig_name", []))
        # Extract requested leads, resample if necessary
        lead_indices = [sig_names.index(l) for l in leads if l in sig_names]
        if not lead_indices:
            continue
        sig_leads = sig[:, lead_indices].T  # (L, T)
        fs = int(fields.get("fs", sample_rate))
        # Resample to target sample_rate if different
        if fs != sample_rate:
            from scipy.signal import resample

            T_new = int(sig_leads.shape[1] * (sample_rate / fs))
            sig_leads = resample(sig_leads, T_new, axis=1)
        # Truncate or pad to window_samples
        T = sig_leads.shape[1]
        if T >= window_samples:
            sig_leads = sig_leads[:, :window_samples]
        else:
            pad = np.zeros((sig_leads.shape[0], window_samples - T), dtype=sig_leads.dtype)
            sig_leads = np.concatenate([sig_leads, pad], axis=1)
        X_list.append(sig_leads.astype(np.float32))
        y_list.append(label_to_id.get(tuple(row["y_labels"]), -1))

    X = np.stack(X_list, axis=0) if X_list else np.empty((0, len(leads), window_samples), dtype=np.float32)
    y = np.array(y_list, dtype=int)

    metadata = {
        "n_samples": int(X.shape[0]),
        "matrix_shape": (X.shape[1], X.shape[2]),
        "leads": leads,
        "sample_rate": sample_rate,
        "window_samples": window_samples,
        "label_map": {i: list(l) for i, l in enumerate(unique_labelsets)},
    }
    return X, y, metadata 