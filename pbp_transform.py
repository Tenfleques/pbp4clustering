from typing import Tuple, Optional
import numpy as np
from src.core import pbp_vector
from src.aggregation_functions import get_aggregation_function
from src.sorting_functions import get_sorting_function


def _reduce_rows_pca(matrix: np.ndarray, k: int) -> np.ndarray:
    """
    Reduce the row dimension of a single (m, n) matrix to (k, n) using per-sample PCA via SVD.

    This projects the matrix onto its top-k left singular vectors, preserving the temporal axis (n).
    """
    m, n = matrix.shape
    if k >= m:
        return matrix
    # Use SVD for stable PCA-like projection per sample
    # X = U S Vh; project rows onto top-k components: X_k = (U[:, :k].T @ X) = S_k Vh_k
    # This yields shape (k, n)
    U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
    k = int(k)
    Uk = U[:, :k]
    Xk = Uk.T @ matrix
    return Xk.astype(matrix.dtype, copy=False)


def matrices_to_pbp_vectors(
    matrices: np.ndarray,
    agg: str = "sum",
    dtype=np.float32,
    rows_target: Optional[int] = None,
    row_reduce_method: str = "pca",
    sort: Optional[str] = None,
) -> np.ndarray:
    """
    Transform (N, m, n) matrices into PBP vectors of length (2^m - 1).

    Args:
        matrices: array of shape (N, m, n)
        agg: aggregation function name from src.pbp.aggregation_functions
        dtype: output dtype

    Returns:
        X_pbp: (N, 2^m - 1)
    """
    assert matrices.ndim == 3, "Expected (N, m, n)"
    num_samples, m, n = matrices.shape
    agg_func = get_aggregation_function(agg)
    sort_func = None
    if sort is not None:
        sort_func = get_sorting_function(sort)

    # Optionally reduce row dimension (m -> rows_target)
    if rows_target is not None and rows_target < m:
        if row_reduce_method == "pca":
            reduced = np.empty((num_samples, int(rows_target), n), dtype=matrices.dtype)
            for i in range(num_samples):
                reduced[i] = _reduce_rows_pca(matrices[i], int(rows_target))
            matrices = reduced
            m = int(rows_target)
        else:
            raise ValueError(f"Unsupported row_reduce_method: {row_reduce_method}")

    sample_vec = pbp_vector(matrices[0], agg_func, sort_func)
    vec_len = sample_vec.shape[0]

    X_pbp = np.empty((num_samples, vec_len), dtype=dtype)
    X_pbp[0] = sample_vec.astype(dtype, copy=False)

    for i in range(1, num_samples):
        X_pbp[i] = pbp_vector(matrices[i], agg_func, sort_func).astype(dtype, copy=False)
    return X_pbp 