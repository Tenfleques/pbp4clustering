from pathlib import Path
from typing import Tuple, Dict, Any
import io

import numpy as np
import pandas as pd
import requests

COFFEE_URL = "https://raw.githubusercontent.com/zwbben/SpectroscopyData/master/FTIR_Spectra_instant_coffee.csv"


def _download_csv(url: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
    return out_path


def load_coffee_spectra(data_dir: str = "./data/spectroscopy") -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load FTIR coffee spectra (arabica vs robusta) and build (N, B, 1) matrices with class labels.

    The CSV layout from the source has:
      - Row 0: header with sample indices (1..N)
      - First data row: 'Group Code:' followed by numeric codes per sample (1/2)
      - Second data row: 'Wavenumbers' row where each subsequent cell is class name per sample
      - Remaining rows: spectral bands as first column; per-sample intensities in subsequent columns
    """
    out_path = _download_csv(COFFEE_URL, Path(data_dir) / "FTIR_Spectra_instant_coffee.csv")
    raw = pd.read_csv(out_path)

    # The first column is an index-like label; columns '1','2',... correspond to samples
    cols = [c for c in raw.columns if c != 'Sample Number:']

    # First row contains group code (1/2), second row contains class names per sample
    group_row = raw.iloc[0][cols]
    class_row = raw.iloc[1][cols]

    # Remaining rows are spectra: first column has wavenumber, sample columns have intensities
    spectra = raw.iloc[2:].reset_index(drop=True)
    wavenumbers = spectra['Sample Number:'].astype(str).tolist()

    # Build matrix samples × bands by stacking columns
    sample_ids = cols
    X_list = []
    for sid in sample_ids:
        X_list.append(pd.to_numeric(spectra[sid], errors='coerce').to_numpy(dtype=np.float32))
    X = np.stack(X_list, axis=0)  # (N, B)

    # Labels from class_row (strings like 'Arabica','Robusta')
    y_raw = class_row.values.astype(str)
    classes = sorted(set(y_raw.tolist()))
    class_to_id = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_id.get(lbl, -1) for lbl in y_raw], dtype=int)

    # Filter any -1 labels (should not happen)
    keep = y >= 0
    X = X[keep]
    y = y[keep]

    # Build matrices (bands×1)
    X_mats = X.reshape(X.shape[0], X.shape[1], 1)

    metadata = {
        "n_samples": int(X_mats.shape[0]),
        "matrix_shape": (int(X_mats.shape[1]), 1),
        "bands": wavenumbers,
        "label_map": class_to_id,
    }
    return X_mats, y, metadata 