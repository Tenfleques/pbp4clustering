import os
import io
import zipfile
import requests
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any

UCI_HAR_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"


def _download_and_extract_har(target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / "UCI_HAR_Dataset.zip"
    extract_root = target_dir / "UCI HAR Dataset"

    if not extract_root.exists():
        # Download if not present
        if not zip_path.exists():
            resp = requests.get(UCI_HAR_URL, stream=True, timeout=120)
            resp.raise_for_status()
            with zip_path.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        # Extract
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(target_dir)
    return extract_root


def _load_inertial_block(split_dir: Path, kind: str) -> np.ndarray:
    # kind in {"train", "test"}
    inertial = split_dir / "Inertial Signals"
    # We use total acceleration (acc) and body gyroscope (gyro) on x,y,z => 6 signals
    files = [
        inertial / f"total_acc_x_{kind}.txt",
        inertial / f"total_acc_y_{kind}.txt",
        inertial / f"total_acc_z_{kind}.txt",
        inertial / f"body_gyro_x_{kind}.txt",
        inertial / f"body_gyro_y_{kind}.txt",
        inertial / f"body_gyro_z_{kind}.txt",
    ]
    # Each file: shape (num_samples, 128)
    signals = [np.loadtxt(str(fp)) for fp in files]
    # Stack into (num_samples, 6, 128)
    stacked = np.stack(signals, axis=1)
    return stacked


def _load_labels(split_dir: Path, kind: str) -> np.ndarray:
    y_path = split_dir / f"y_{kind}.txt"
    return np.loadtxt(str(y_path), dtype=int)


def _load_activity_labels(root: Path) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    with (root / "activity_labels.txt").open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                mapping[int(parts[0])] = parts[1]
    return mapping


def load_har_six_axis(data_dir: str = "./data/har") -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load UCI HAR dataset and return 6-axis (acc+gyro) matrices per sample.

    Returns:
        X: (N, 6, 128) float32
        y: (N,) int labels in [1..6]
        metadata: dict with label_names and paths
    """
    root = _download_and_extract_har(Path(data_dir))
    train_dir = root / "train"
    test_dir = root / "test"

    X_train = _load_inertial_block(train_dir, "train")
    X_test = _load_inertial_block(test_dir, "test")
    y_train = _load_labels(train_dir, "train")
    y_test = _load_labels(test_dir, "test")

    X = np.concatenate([X_train, X_test], axis=0).astype(np.float32)
    y = np.concatenate([y_train, y_test], axis=0).astype(int)

    labels_map = _load_activity_labels(root)
    metadata = {
        "root": str(root),
        "n_samples": int(X.shape[0]),
        "matrix_shape": (int(X.shape[1]), int(X.shape[2])),
        "labels_map": labels_map,
    }
    return X, y, metadata 


def load_har_axis_feature_format(
    data_dir: str = "./data/har",
    include_body_acc: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load UCI HAR and return matrices grouped by axis with features concatenated along columns.

    Output shape: (N, 3, n_concat) where rows correspond to axes [x, y, z], and columns
    are time-series values concatenated per feature (e.g., total_acc, body_gyro[, body_acc]).

    By default, uses features: [total_acc, body_gyro]. Set include_body_acc=True to add body_acc.
    """
    root = _download_and_extract_har(Path(data_dir))
    train_dir = root / "train"
    test_dir = root / "test"

    # Load base 6 signals (total_acc x/y/z, body_gyro x/y/z)
    X_train_6 = _load_inertial_block(train_dir, "train")  # (n_train, 6, 128)
    X_test_6 = _load_inertial_block(test_dir, "test")    # (n_test, 6, 128)

    # Optionally load body_acc x/y/z
    def _load_body_acc(split_dir: Path, kind: str) -> np.ndarray:
        inertial = split_dir / "Inertial Signals"
        files = [
            inertial / f"body_acc_x_{kind}.txt",
            inertial / f"body_acc_y_{kind}.txt",
            inertial / f"body_acc_z_{kind}.txt",
        ]
        signals = [np.loadtxt(str(fp)) for fp in files]
        # Shape (num_samples, 3, 128) in x,y,z order
        return np.stack(signals, axis=1)

    body_acc_train = _load_body_acc(train_dir, "train") if include_body_acc else None
    body_acc_test = _load_body_acc(test_dir, "test") if include_body_acc else None

    # Split 6-signal bundles into feature blocks of shape (num_samples, 3, 128)
    def _split_features(X6: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        total_acc = X6[:, 0:3, :]  # x,y,z
        body_gyro = X6[:, 3:6, :]  # x,y,z
        return total_acc, body_gyro

    tot_train, gyro_train = _split_features(X_train_6)
    tot_test, gyro_test = _split_features(X_test_6)

    # Build concatenated columns per axis
    def _concat_by_axis(feature_blocks: Tuple[np.ndarray, ...]) -> np.ndarray:
        # feature_blocks: tuple of arrays each (N, 3, 128)
        N = feature_blocks[0].shape[0]
        axis_rows = []
        for axis_idx in range(3):  # x, y, z
            parts = [blk[:, axis_idx, :] for blk in feature_blocks]  # each (N, 128)
            axis_concat = np.concatenate(parts, axis=1)  # (N, 128 * num_features)
            axis_rows.append(axis_concat)
        # Stack rows into (N, 3, n_concat)
        return np.stack(axis_rows, axis=1)

    feature_blocks_train = (tot_train, gyro_train)
    feature_blocks_test = (tot_test, gyro_test)
    feature_order = ["total_acc", "body_gyro"]
    if include_body_acc:
        feature_blocks_train = feature_blocks_train + (body_acc_train,)
        feature_blocks_test = feature_blocks_test + (body_acc_test,)
        feature_order.append("body_acc")

    X_train = _concat_by_axis(feature_blocks_train).astype(np.float32)
    X_test = _concat_by_axis(feature_blocks_test).astype(np.float32)

    # Labels
    y_train = _load_labels(train_dir, "train")
    y_test = _load_labels(test_dir, "test")
    y = np.concatenate([y_train, y_test], axis=0).astype(int)

    # Combine
    X = np.concatenate([X_train, X_test], axis=0)

    labels_map = _load_activity_labels(root)
    metadata = {
        "root": str(root),
        "n_samples": int(X.shape[0]),
        "matrix_shape": (int(X.shape[1]), int(X.shape[2])),  # (3, n_concat)
        "labels_map": labels_map,
        "axis_order": ["x", "y", "z"],
        "feature_order": feature_order,
        "window_length": 128,
    }
    return X, y, metadata


def load_har_axis_feature_columns(
    data_dir: str = "./data/har",
    include_body_acc: bool = False,
    time_agg: str = "mean",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load UCI HAR and return matrices shaped (N, 3, F) where rows are axes [x, y, z]
    and columns are feature families as standalone columns (e.g., total_acc, body_gyro[, body_acc]).

    Each feature family is aggregated over the time dimension (length 128) using `time_agg` (default: mean).
    """
    root = _download_and_extract_har(Path(data_dir))
    train_dir = root / "train"
    test_dir = root / "test"

    X_train_6 = _load_inertial_block(train_dir, "train")  # (n_train, 6, 128)
    X_test_6 = _load_inertial_block(test_dir, "test")    # (n_test, 6, 128)

    def _load_body_acc(split_dir: Path, kind: str) -> np.ndarray:
        inertial = split_dir / "Inertial Signals"
        files = [
            inertial / f"body_acc_x_{kind}.txt",
            inertial / f"body_acc_y_{kind}.txt",
            inertial / f"body_acc_z_{kind}.txt",
        ]
        signals = [np.loadtxt(str(fp)) for fp in files]
        return np.stack(signals, axis=1)  # (N, 3, 128)

    body_acc_train = _load_body_acc(train_dir, "train") if include_body_acc else None
    body_acc_test = _load_body_acc(test_dir, "test") if include_body_acc else None

    def _split_features(X6: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        total_acc = X6[:, 0:3, :]  # x,y,z
        body_gyro = X6[:, 3:6, :]  # x,y,z
        return total_acc, body_gyro

    tot_train, gyro_train = _split_features(X_train_6)
    tot_test, gyro_test = _split_features(X_test_6)

    def _agg_time(block: np.ndarray) -> np.ndarray:
        # block: (N, 3, 128) -> (N, 3)
        if time_agg == "mean":
            return block.mean(axis=2)
        if time_agg == "median":
            return np.median(block, axis=2)
        if time_agg == "sum":
            return block.sum(axis=2)
        if time_agg == "min":
            return block.min(axis=2)
        if time_agg == "max":
            return block.max(axis=2)
        if time_agg == "std":
            return block.std(axis=2)
        if time_agg == "var":
            return block.var(axis=2)
        if time_agg == "rms":
            return np.sqrt(np.mean(block * block, axis=2))
        if time_agg == "absmean":
            return np.mean(np.abs(block), axis=2)
        if time_agg == "mad":
            median_vals = np.median(block, axis=2, keepdims=True)
            return np.median(np.abs(block - median_vals), axis=2)
        if time_agg == "iqr":
            q75 = np.percentile(block, 75, axis=2)
            q25 = np.percentile(block, 25, axis=2)
            return q75 - q25
        if time_agg == "energy":
            return np.sum(block * block, axis=2)
        if time_agg == "entropy":
            # Approximate Shannon entropy over time using per-sample histograms
            N = block.shape[0]
            out = np.zeros((N, 3), dtype=block.dtype)
            num_bins = 32
            eps = 1e-12
            for i in range(N):
                for axis_idx in range(3):
                    series = block[i, axis_idx, :]
                    vmin = series.min()
                    vmax = series.max()
                    if vmax - vmin < eps:
                        out[i, axis_idx] = 0.0
                        continue
                    counts, _ = np.histogram(series, bins=num_bins, range=(vmin, vmax))
                    p = counts.astype(np.float64)
                    p_sum = p.sum()
                    if p_sum == 0:
                        out[i, axis_idx] = 0.0
                    else:
                        p /= p_sum
                        out[i, axis_idx] = float(-(p * np.log(p + eps)).sum())
            return out
        raise ValueError(f"Unsupported time_agg: {time_agg}")

    features_train = [_agg_time(tot_train), _agg_time(gyro_train)]  # each (N, 3)
    features_test = [_agg_time(tot_test), _agg_time(gyro_test)]
    feature_order = ["total_acc", "body_gyro"]
    if include_body_acc:
        features_train.append(_agg_time(body_acc_train))
        features_test.append(_agg_time(body_acc_test))
        feature_order.append("body_acc")

    # Stack features along columns: (N, 3, F)
    X_train = np.stack(features_train, axis=2).astype(np.float32)
    X_test = np.stack(features_test, axis=2).astype(np.float32)

    y_train = _load_labels(train_dir, "train")
    y_test = _load_labels(test_dir, "test")
    y = np.concatenate([y_train, y_test], axis=0).astype(int)

    X = np.concatenate([X_train, X_test], axis=0)

    labels_map = _load_activity_labels(root)
    metadata = {
        "root": str(root),
        "n_samples": int(X.shape[0]),
        "matrix_shape": (int(X.shape[1]), int(X.shape[2])),  # (3, F)
        "labels_map": labels_map,
        "axis_order": ["x", "y", "z"],
        "feature_order": feature_order,
        "time_agg": time_agg,
    }
    return X, y, metadata