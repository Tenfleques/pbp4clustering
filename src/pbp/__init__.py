"""
PBP Core Implementation

This module contains the core Pseudo-Boolean Polynomial (PBP) implementation
for dimensionality reduction and clustering.
"""

# Import functions from the core module
from .core import (
    pbp_vector,
    create_pbp,
    reduce_pbp_pandas,
    create_coeffs_matrix,
    create_variable_matrix,
    create_perm,
    calculate_degree,
    bin_encoder,
    decode_var,
    truncate_pBp,
    trunc_driver,
    to_string
)

__all__ = [
    "pbp_vector",
    "create_pbp",
    "reduce_pbp_pandas",
    "create_coeffs_matrix",
    "create_variable_matrix",
    "create_perm",
    "calculate_degree",
    "bin_encoder",
    "decode_var",
    "truncate_pBp",
    "trunc_driver",
    "to_string"
] 