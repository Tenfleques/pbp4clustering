#!/usr/bin/env python3
"""
PBP Runner - Entry point for PBP functionality

This script provides access to the PBP core functionality from the src structure.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.pbp.core import (
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

# Re-export all functions
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

if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    c = np.array([
        [7, 8, 2, 10, 3],
        [4, 12, 1, 8, 4],
        [5, 3, 0, 6, 9],
        [9, 6, 7, 1, 5]
    ])

    v = pbp_vector(c)
    print("PBP Vector Result:")
    print(v)
    

    c = np.array([
        [5.4, 3.4],
        [1.7, 0.2],
    ])
    v = pbp_vector(c)
    print("\nPBP Vector Result (2x2):")
    print(v)

    c = np.array([
        [5.1, 3.7],
        [1.5, 0.4],
    ])
    v = pbp_vector(c)
    print("\nPBP Vector Result (2x2):")
    print(v) 