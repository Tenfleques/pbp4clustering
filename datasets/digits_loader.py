#!/usr/bin/env python3
from typing import Tuple, Dict, Any, List

import numpy as np
from sklearn.datasets import load_digits


def _entropy(values: np.ndarray, num_bins: int = 16) -> float:
    if values.size == 0:
        return 0.0
    hist, _ = np.histogram(values, bins=num_bins, range=(0.0, 16.0))
    p = hist.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p /= s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _center_of_mass(block: np.ndarray) -> Tuple[float, float]:
    # block intensities; coordinates origin at (0,0) top-left
    h, w = block.shape
    if h == 0 or w == 0:
        return 0.0, 0.0
    y_coords = np.arange(h).reshape(h, 1)
    x_coords = np.arange(w).reshape(1, w)
    mass = block.astype(np.float64)
    total = mass.sum()
    if total <= 1e-12:
        return 0.0, 0.0
    cy = float((mass * y_coords).sum() / total)
    cx = float((mass * x_coords).sum() / total)
    # Normalize to [0,1] within block
    return cy / max(h - 1, 1), cx / max(w - 1, 1)


def _h_symmetry(block: np.ndarray) -> float:
    # Horizontal symmetry: 1 - normalized L1 distance between left and flipped-right halves
    h, w = block.shape
    mid = w // 2
    left = block[:, :mid]
    right = block[:, w - mid:][:, ::-1]
    denom = np.abs(left).sum() + np.abs(right).sum() + 1e-8
    diff = np.abs(left - right).sum()
    score = 1.0 - (diff / denom)
    return float(score)


def _compute_block_features(block: np.ndarray, measures: List[str]) -> List[float]:
    vals: List[float] = []
    block = block.astype(np.float32)
    if 'mean' in measures:
        vals.append(float(block.mean()))
    if 'std' in measures:
        vals.append(float(block.std()))
    if 'sum' in measures:
        vals.append(float(block.sum()))
    if 'nonzero_frac' in measures:
        nz = float((block > 0).sum())
        vals.append(nz / float(block.size))
    if 'entropy' in measures:
        vals.append(_entropy(block.ravel()))
    if 'grad_mag' in measures:
        # Simple gradient magnitude proxy via finite diffs
        dy = np.diff(block, axis=0)
        dx = np.diff(block, axis=1)
        gm = float(np.abs(dy).sum() + np.abs(dx).sum())
        vals.append(gm)
    if 'centroid' in measures:
        cy, cx = _center_of_mass(block)
        vals.extend([cy, cx])
    if 'h_symmetry' in measures:
        vals.append(_h_symmetry(block))
    return vals


def load_digits_matrices(
    data_dir: str = "./data/digits",  # unused; kept for signature consistency
    rows: int = 2,
    col_blocks: int = 4,
    measures: List[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load UCI/Sklearn 8x8 digits and build (N, rows, col_blocks * num_measures) matrices.

    - rows: 2 → split into top 4 vs bottom 4 rows. 3 → [3,2,3] row bands.
    - col_blocks: one of {2,4,8}. Splits 8 columns into equal-width blocks.
    - measures: per-block measures concatenated along the column axis.

    Default measures: ['mean', 'std', 'nonzero_frac', 'entropy']
    """
    if measures is None:
        measures = ['mean', 'std', 'nonzero_frac', 'entropy']

    ds = load_digits()
    X = ds.images.astype(np.float32)  # (N, 8, 8), values 0..16
    y = ds.target.astype(int)

    assert rows in (2, 3), "rows must be 2 or 3"
    assert col_blocks in (2, 4, 8), "col_blocks must be 2, 4, or 8"

    N = X.shape[0]
    # Row band splits
    if rows == 2:
        band_sizes = [4, 4]
    else:
        band_sizes = [3, 2, 3]
    # Column block width
    block_w = 8 // col_blocks

    num_meas = 0
    # Precompute length of feature vector per block
    tmp_feats = _compute_block_features(np.zeros((band_sizes[0], block_w), dtype=np.float32), measures)
    num_meas = len(tmp_feats)
    out_cols = col_blocks * num_meas

    X_mats = np.empty((N, rows, out_cols), dtype=np.float32)

    for i in range(N):
        img = X[i]
        r0 = 0
        row_features: List[List[float]] = []
        for bsz in band_sizes:
            band = img[r0:r0 + bsz, :]
            r0 += bsz
            block_feats: List[float] = []
            c0 = 0
            for _ in range(col_blocks):
                block = band[:, c0:c0 + block_w]
                c0 += block_w
                feats = _compute_block_features(block, measures)
                block_feats.extend(feats)
            row_features.append(block_feats)
        # Assign
        for r_idx, feats in enumerate(row_features):
            X_mats[i, r_idx, :] = np.asarray(feats, dtype=np.float32)

    labels_map = {int(k): str(k) for k in np.unique(y)}
    meta: Dict[str, Any] = {
        "n_samples": int(N),
        "matrix_shape": (rows, out_cols),
        "rows": [f"band_{i+1}" for i in range(rows)],
        "cols": [f"b{b+1}_{m}" for b in range(col_blocks) for m in measures],
        "labels_map": labels_map,
        "measures": measures,
        "col_blocks": int(col_blocks),
    }
    return X_mats, y, meta


