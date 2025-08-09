#!/usr/bin/env python3
from typing import Tuple, Dict, Any
from pathlib import Path
import io

import numpy as np
import pandas as pd
import requests


def load_pima_matrices(data_dir: str = "./data/pima") -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load Pima Indians Diabetes and build (N, 2, 4) matrices.

    Rows: Biochem (glucose, insulin, bmi, skin_thickness)
          Demographic (pregnancies, blood_pressure, diabetes_pedigree_function, age)
    Labels: outcome (0/1)
    """
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(data_dir) / "pima.csv"
    if not csv_path.exists():
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        tmp = pd.read_csv(io.StringIO(resp.text), header=None)
        tmp.columns = [
            "pregnancies", "glucose", "blood_pressure", "skin_thickness",
            "insulin", "bmi", "diabetes_pedigree_function", "age", "outcome",
        ]
        tmp.to_csv(csv_path, index=False)

    df = pd.read_csv(csv_path)
    y = df["outcome"].astype(int).to_numpy()

    biochem = df[["glucose", "insulin", "bmi", "skin_thickness"]].to_numpy(dtype=np.float32)
    demo = df[["pregnancies", "blood_pressure", "diabetes_pedigree_function", "age"]].to_numpy(dtype=np.float32)
    N = df.shape[0]
    X_mats = np.empty((N, 2, 4), dtype=np.float32)
    X_mats[:, 0, :] = biochem
    X_mats[:, 1, :] = demo

    meta: Dict[str, Any] = {
        "n_samples": int(N),
        "matrix_shape": (2, 4),
        "rows": ["biochem", "demographic"],
        "cols": ["f1", "f2", "f3", "f4"],
        "labels_map": {0: "no_diabetes", 1: "diabetes"},
    }
    return X_mats, y, meta


