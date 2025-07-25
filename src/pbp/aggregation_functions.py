#!/usr/bin/env python3
"""
Aggregation Functions for Pseudo-Boolean Polynomial Dimensionality Reduction

This module provides various aggregation functions that can be used in PBP calculations
to combine similar terms in the polynomial in ways that improve clustering performance.
"""

import numpy as np
import pandas as pd
from typing import Callable, Any, Dict, List
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# Basic Aggregation Functions
def sum_agg(x: pd.Series) -> float:
    """Default sum aggregation - adds all coefficients for similar terms."""
    return x.sum()


def mean_agg(x: pd.Series) -> float:
    """Mean aggregation - averages coefficients for similar terms."""
    return x.mean()


def median_agg(x: pd.Series) -> float:
    """Median aggregation - uses median of coefficients for similar terms."""
    return x.median()


def max_agg(x: pd.Series) -> float:
    """Maximum aggregation - uses maximum coefficient for similar terms."""
    return x.max()


def min_agg(x: pd.Series) -> float:
    """Minimum aggregation - uses minimum coefficient for similar terms."""
    return x.min()


# Statistical Aggregation Functions
def std_agg(x: pd.Series) -> float:
    """Standard deviation aggregation - captures variability in coefficients."""
    return x.std() if len(x) > 1 else 0.0


def var_agg(x: pd.Series) -> float:
    """Variance aggregation - captures spread in coefficients."""
    return x.var() if len(x) > 1 else 0.0


def skew_agg(x: pd.Series) -> float:
    """Skewness aggregation - captures asymmetry in coefficient distribution."""
    if len(x) < 3:
        return 0.0
    return stats.skew(x) if not np.isnan(stats.skew(x)) else 0.0


def kurtosis_agg(x: pd.Series) -> float:
    """Kurtosis aggregation - captures tail behavior of coefficient distribution."""
    if len(x) < 4:
        return 0.0
    return stats.kurtosis(x) if not np.isnan(stats.kurtosis(x)) else 0.0


# Robust Aggregation Functions
def trimmed_mean_agg(x: pd.Series, trim_percent: float = 0.1) -> float:
    """Trimmed mean aggregation - removes outliers before averaging."""
    if len(x) < 3:
        return x.mean()
    trim_count = max(1, int(len(x) * trim_percent))
    sorted_x = np.sort(x)
    return np.mean(sorted_x[trim_count:-trim_count])


def winsorized_mean_agg(x: pd.Series, winsorize_percent: float = 0.1) -> float:
    """Winsorized mean aggregation - caps outliers instead of removing them."""
    if len(x) < 3:
        return x.mean()
    winsorize_count = max(1, int(len(x) * winsorize_percent))
    sorted_x = np.sort(x)
    lower_bound = sorted_x[winsorize_count]
    upper_bound = sorted_x[-(winsorize_count + 1)]
    winsorized_x = np.clip(x, lower_bound, upper_bound)
    return winsorized_x.mean()


# Weighted Aggregation Functions
def weighted_sum_agg(x: pd.Series, weights: List[float] = None) -> float:
    """Weighted sum aggregation - applies weights to coefficients."""
    if weights is None or len(weights) != len(x):
        weights = np.ones(len(x))
    return np.average(x, weights=weights)


def exponential_weighted_agg(x: pd.Series, alpha: float = 0.5) -> float:
    """Exponential weighted aggregation - gives more weight to recent/larger values."""
    if len(x) == 1:
        return x.iloc[0]
    weights = np.exp(alpha * np.arange(len(x)))
    return np.average(x, weights=weights)


# Non-linear Aggregation Functions
def root_mean_square_agg(x: pd.Series) -> float:
    """Root mean square aggregation - emphasizes larger coefficients."""
    return np.sqrt(np.mean(x**2))


def geometric_mean_agg(x: pd.Series) -> float:
    """Geometric mean aggregation - multiplicative combination."""
    if (x <= 0).any():
        return x.mean()  # Fallback to arithmetic mean
    return stats.gmean(x)


def harmonic_mean_agg(x: pd.Series) -> float:
    """Harmonic mean aggregation - emphasizes smaller coefficients."""
    if (x <= 0).any():
        return x.mean()  # Fallback to arithmetic mean
    return stats.hmean(x)


# Percentile-based Aggregation Functions
def percentile_agg(x: pd.Series, percentile: float = 75) -> float:
    """Percentile aggregation - uses specific percentile of coefficients."""
    return np.percentile(x, percentile)


def iqr_agg(x: pd.Series) -> float:
    """Interquartile range aggregation - captures middle 50% spread."""
    return np.percentile(x, 75) - np.percentile(x, 25)


def range_agg(x: pd.Series) -> float:
    """Range aggregation - captures total spread of coefficients."""
    return x.max() - x.min()


# Entropy-based Aggregation Functions
def entropy_agg(x: pd.Series) -> float:
    """Entropy aggregation - measures information content in coefficients."""
    if len(x) < 2:
        return 0.0
    # Normalize to probabilities
    x_norm = np.abs(x) / np.sum(np.abs(x))
    x_norm = x_norm[x_norm > 0]  # Remove zeros
    if len(x_norm) < 2:
        return 0.0
    return -np.sum(x_norm * np.log2(x_norm))


def gini_agg(x: pd.Series) -> float:
    """Gini coefficient aggregation - measures inequality in coefficients."""
    if len(x) < 2:
        return 0.0
    x_sorted = np.sort(x)
    n = len(x_sorted)
    cumsum = np.cumsum(x_sorted)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n


# Adaptive Aggregation Functions
def adaptive_agg(x: pd.Series) -> float:
    """Adaptive aggregation - chooses method based on data characteristics."""
    if len(x) == 1:
        return x.iloc[0]
    
    # Check for outliers
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    outlier_threshold = 1.5 * iqr
    outliers = ((x < q1 - outlier_threshold) | (x > q3 + outlier_threshold)).sum()
    
    # Check for skewness
    skewness = abs(stats.skew(x)) if len(x) >= 3 else 0
    
    # Choose aggregation method based on data characteristics
    if outliers > len(x) * 0.2:  # Many outliers
        return trimmed_mean_agg(x, 0.2)
    elif skewness > 1.0:  # Highly skewed
        return median_agg(x)
    elif x.std() / x.mean() > 0.5:  # High coefficient of variation
        return root_mean_square_agg(x)
    else:  # Normal distribution
        return mean_agg(x)


def robust_adaptive_agg(x: pd.Series) -> float:
    """Robust adaptive aggregation - combines multiple robust methods."""
    if len(x) == 1:
        return x.iloc[0]
    
    # Calculate multiple robust statistics
    median_val = x.median()
    mad = np.median(np.abs(x - median_val))  # Median absolute deviation
    
    # If MAD is small relative to median, data is consistent
    if mad / abs(median_val) < 0.1:
        return median_val
    else:
        # Use trimmed mean for more robust estimation
        return trimmed_mean_agg(x, 0.25)


# Dictionary of all aggregation functions
AGGREGATION_FUNCTIONS = {
    # Basic functions
    'sum': sum_agg,
    'mean': mean_agg,
    'median': median_agg,
    'max': max_agg,
    'min': min_agg,
    
    # Statistical functions
    'std': std_agg,
    'var': var_agg,
    'skew': skew_agg,
    'kurtosis': kurtosis_agg,
    
    # Robust functions
    'trimmed_mean': trimmed_mean_agg,
    'winsorized_mean': winsorized_mean_agg,
    
    # Weighted functions
    'weighted_sum': weighted_sum_agg,
    'exponential_weighted': exponential_weighted_agg,
    
    # Non-linear functions
    'rms': root_mean_square_agg,
    'geometric_mean': geometric_mean_agg,
    'harmonic_mean': harmonic_mean_agg,
    
    # Percentile functions
    'percentile_75': lambda x: percentile_agg(x, 75),
    'percentile_90': lambda x: percentile_agg(x, 90),
    'iqr': iqr_agg,
    'range': range_agg,
    
    # Entropy functions
    'entropy': entropy_agg,
    'gini': gini_agg,
    
    # Adaptive functions
    'adaptive': adaptive_agg,
    'robust_adaptive': robust_adaptive_agg,
}


def get_aggregation_function(name: str) -> Callable[[pd.Series], Any]:
    """
    Get an aggregation function by name.
    
    Args:
        name: Name of the aggregation function
        
    Returns:
        Callable: The aggregation function
        
    Raises:
        ValueError: If the function name is not found
    """
    if name not in AGGREGATION_FUNCTIONS:
        available = list(AGGREGATION_FUNCTIONS.keys())
        raise ValueError(f"Unknown aggregation function '{name}'. Available: {available}")
    
    return AGGREGATION_FUNCTIONS[name]


def get_all_aggregation_functions() -> Dict[str, Callable[[pd.Series], Any]]:
    """
    Get all available aggregation functions.
    
    Returns:
        Dict: Dictionary mapping function names to functions
    """
    return AGGREGATION_FUNCTIONS.copy()


def get_recommended_aggregation_functions() -> List[str]:
    """
    Get a list of recommended aggregation functions for clustering.
    
    Returns:
        List: List of recommended function names
    """
    return [
        'sum',           # Default baseline
        'mean',          # Standard average
        'median',        # Robust to outliers
        'trimmed_mean',  # Robust with outlier removal
        'rms',           # Emphasizes larger values
        'adaptive',      # Data-driven choice
        'robust_adaptive', # Robust data-driven choice
        'entropy',       # Information content
        'gini',          # Inequality measure
        'iqr',           # Spread measure
    ]


def evaluate_aggregation_function(agg_func: Callable[[pd.Series], Any], 
                                test_data: List[float]) -> Dict[str, float]:
    """
    Evaluate an aggregation function on test data.
    
    Args:
        agg_func: Aggregation function to evaluate
        test_data: List of test values
        
    Returns:
        Dict: Evaluation metrics
    """
    try:
        result = agg_func(pd.Series(test_data))
        return {
            'result': result,
            'success': True,
            'error': None
        }
    except Exception as e:
        return {
            'result': None,
            'success': False,
            'error': str(e)
        }


if __name__ == "__main__":
    # Test the aggregation functions
    test_data = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]  # Includes outlier
    
    print("Testing aggregation functions:")
    print(f"Test data: {test_data}")
    print()
    
    for name, func in AGGREGATION_FUNCTIONS.items():
        try:
            result = func(pd.Series(test_data))
            print(f"{name:20s}: {result:.4f}")
        except Exception as e:
            print(f"{name:20s}: ERROR - {e}")
    
    print(f"\nRecommended functions: {get_recommended_aggregation_functions()}") 