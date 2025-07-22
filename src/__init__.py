"""
PBP Clustering and Dimensionality Reduction Package

This package provides implementations and analysis tools for the Pseudo-Boolean
Polynomial (PBP) method for dimensionality reduction and clustering.
"""

__version__ = "1.0.0"
__author__ = "Tendai Chikake"
__email__ = "tendai.chikake@example.com"

# Import actual functionality from source files
from . import pbp
from . import data
from . import analysis

__all__ = ["pbp", "data", "analysis"] 