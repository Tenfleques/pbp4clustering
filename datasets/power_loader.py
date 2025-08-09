#!/usr/bin/env python3
from typing import Tuple, Dict, Any
from pathlib import Path
import io
import zipfile

import numpy as np
import pandas as pd
import requests


def load_household_power_matrices(
    data_dir: str = "./data/power",
    days_limit: int = 60,
    hours_block: int = 4,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load Individual household electric power consumption dataset and build (N_days, 3, 24/hours_block) matrices.

    Rows: [Sub_metering_1, Sub_metering_2, Sub_metering_3]
    Columns: 4-hour blocks (default) per day (6 blocks)
    Labels: day of week (0=Mon..6=Sun)
    """
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    zip_path = Path(data_dir) / "household_power_consumption.zip"
    txt_path = Path(data_dir) / "household_power_consumption.txt"
    if not txt_path.exists():
        if not zip_path.exists():
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/individual-household-electric-power-consumption.zip"
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            zip_path.write_bytes(resp.content)
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if name.endswith("household_power_consumption.txt"):
                    zf.extract(name, path=data_dir)
                    # Move to desired path if inside a folder
                    src = Path(data_dir) / name
                    src.rename(txt_path)
                    break

    # Read file
    df = pd.read_csv(
        txt_path,
        sep=";",
        na_values=['?'],
        low_memory=False,
    )
    # Parse datetime
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["datetime"]).reset_index(drop=True)

    # Keep required sub meterings and ensure numeric
    for col in ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort and limit to first `days_limit` days with complete data
    df = df.sort_values("datetime")
    df["date"] = df["datetime"].dt.date
    dates = sorted(df["date"].unique().tolist())
    if days_limit is not None:
        dates = dates[:days_limit]
    df = df[df["date"].isin(dates)]

    # Build daily matrices with 4-hour blocks
    def build_day(group: pd.DataFrame) -> np.ndarray:
        # Set hour block index
        hours = group["datetime"].dt.hour.values
        block_idx = (hours // hours_block).astype(int)
        num_blocks = 24 // hours_block
        day_mat = np.zeros((3, num_blocks), dtype=np.float32)
        counts = np.zeros((3, num_blocks), dtype=np.int32)
        vals = group[["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]].to_numpy(dtype=np.float32)
        for i in range(len(group)):
            b = block_idx[i]
            for r in range(3):
                v = vals[i, r]
                if not np.isnan(v):
                    day_mat[r, b] += v
                    counts[r, b] += 1
        # Convert to means where available
        with np.errstate(invalid='ignore'):
            day_mat = np.divide(day_mat, counts, out=np.zeros_like(day_mat), where=counts > 0)
        return day_mat

    day_groups = df.groupby("date", sort=True)
    mats_list = []
    labels = []
    for d, g in day_groups:
        mat = build_day(g)
        if mat.shape[1] == 24 // hours_block:  # ensure full width
            mats_list.append(mat)
            labels.append(pd.Timestamp(d).weekday())

    X = np.stack(mats_list, axis=0)  # (N_days, 3, num_blocks)
    y = np.array(labels, dtype=int)

    meta: Dict[str, Any] = {
        "n_samples": int(X.shape[0]),
        "matrix_shape": (int(X.shape[1]), int(X.shape[2])),
        "rows": ["sub_metering_1", "sub_metering_2", "sub_metering_3"],
        "cols": [f"block_{i}" for i in range(X.shape[2])],
        "labels_map": {i: name for i, name in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])},
    }
    return X.astype(np.float32), y, meta


