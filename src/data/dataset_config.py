#!/usr/bin/env python3
"""
Centralized Dataset Configuration

This module serves as the single source of truth for all dataset names used throughout the codebase.
All scripts should import from this module instead of hardcoding dataset names.
"""

# Standard datasets (sklearn built-in)
STANDARD_DATASETS = [
    'iris',
    'breast_cancer', 
    'wine',
    'digits_sklearn',
    'diabetes'
]

# UCI datasets
UCI_DATASETS = [
    'sonar',
    'glass',
    'vehicle',
    
    'yeast',
    'seeds',
    'thyroid',
    'ionosphere',
    
]

# Large datasets
LARGE_DATASETS = [
    'covertype',
    'kddcup99'
]

# Advanced datasets
ADVANCED_AIDS_DATASETS = [
    
]

ADVANCED_GENETIC_DATASETS = [
    
]

ADVANCED_MEDICAL_DATASETS = [
    'metabolights_study',
    'physionet_ecg',
    'mimic_icu'
]

ADVANCED_NCI_DATASETS = [
    'nci_chemical'
]

# Special datasets
SPECIAL_DATASETS = [
    'glass_conforming',
    'linnerrud',
    'species_distribution'
]

# Biomathematics datasets - Trends in Biomathematics: Modeling Health in Ecology, Social Interactions, and Cells
BIOMATHEMATICS_DATASETS = [
    # Ecology and Environmental Health
    'species_distribution',      # Species distribution modeling
    # 'covertype',                # Forest cover type (environmental health)
    
    # Medical and Health
    'breast_cancer',            # Cancer diagnosis (health modeling)
    'diabetes',                 # Diabetes prediction (health modeling)
    'thyroid',                  # Thyroid function (health modeling)
    'pima',                     # Pima diabetes (health modeling)
    'geo_breast_cancer',        # GEO Breast Cancer gene expression dataset
    
    # Social Interactions and Epidemiology
    'ionosphere',               # Ionosphere radar (signal processing for health monitoring)
    
    # Advanced Medical Datasets
    'metabolights',             # Metabolomics (biochemical health markers)
    # 'physionet_ecg',           # ECG signals (cardiovascular health) - not available
    # 'mimic_icu',               # ICU data (critical care health) - not available
    
    # Chemical and Drug Response (commented out - not available in current config)
    # 'nci_chemical',            # Chemical compounds (drug discovery)
    
    # Core Biomathematics Datasets
    'iris',                     # Biological classification (ecology)
    'wine',                     # Chemical analysis (biochemistry)
    'digits_sklearn',           # Pattern recognition (medical imaging)
    'sonar',                    # Signal processing (medical diagnostics)
    'glass',                    # Material properties (biomaterials)
    'vehicle',                  # Pattern recognition (medical imaging)
    'seeds',                    # Agricultural health (food security)
    'linnerrud'                 # Physical exercise (health fitness)
]

# All datasets by category
DATASET_CATEGORIES = {
    'standard': STANDARD_DATASETS,
    'uci': UCI_DATASETS,
    'large': LARGE_DATASETS,
    'advanced_aids': ADVANCED_AIDS_DATASETS,
    'advanced_genetic': ADVANCED_GENETIC_DATASETS,
    'advanced_medical': ADVANCED_MEDICAL_DATASETS,
    'advanced_nci': ADVANCED_NCI_DATASETS,
    'special': SPECIAL_DATASETS,
    'biomathematics': BIOMATHEMATICS_DATASETS
}

# Flattened list of all datasets
ALL_DATASETS = []
for category, datasets in DATASET_CATEGORIES.items():
    ALL_DATASETS.extend(datasets)

# Core datasets (most commonly used for testing)
CORE_DATASETS = [
    'iris',
    'breast_cancer',
    'wine',
    'digits_sklearn',
    'diabetes',
    'sonar',
    'glass',
    'vehicle',
    
    'yeast'
]

# Optimization datasets (subset used for comprehensive optimization)
OPTIMIZATION_DATASETS = [
    'iris',
    'breast_cancer',
    'wine',
    'digits_sklearn',
    'diabetes',
    'sonar',
    'glass',
    'vehicle',
    
    'yeast',
    'seeds',
    'thyroid',
    'ionosphere',
    'covertype',
    'kddcup99'
]

# Testing datasets (subset used for quick testing)
TESTING_DATASETS = [
    'iris',
    'breast_cancer',
    'wine',
    'digits_sklearn',
    'diabetes',
    'sonar',
    'glass',
    'vehicle',
    
    'yeast',
    'seeds',
    'thyroid',
    'pima',
    'ionosphere',
    
    'glass_conforming',
    'covertype',
    'kddcup99',
    'linnerrud',
    'species_distribution'
]

# Comparison datasets (subset used for method comparison)
COMPARISON_DATASETS = [
    'iris',
    'breast_cancer',
    'geo_breast_cancer',
    'wine',
    'digits_sklearn',
    'diabetes',
    'sonar',
    'glass',
    'vehicle',

    'seeds',
    'thyroid',
    'pima',
    'ionosphere',
    
    'glass_conforming',
    'covertype',
    'kddcup99',
    'linnerrud',
    'species_distribution'
]

def get_datasets_by_category(category):
    """
    Get datasets by category.
    
    Args:
        category (str): Category name ('standard', 'uci', 'large', etc.)
    
    Returns:
        list: List of dataset names in the specified category
    """
    return DATASET_CATEGORIES.get(category, [])

def get_all_datasets():
    """
    Get all available datasets.
    
    Returns:
        list: List of all dataset names
    """
    return ALL_DATASETS.copy()

def get_core_datasets():
    """
    Get core datasets for basic testing.
    
    Returns:
        list: List of core dataset names
    """
    return CORE_DATASETS.copy()

def get_optimization_datasets():
    """
    Get datasets for comprehensive optimization.
    
    Returns:
        list: List of optimization dataset names
    """
    return OPTIMIZATION_DATASETS.copy()

def get_testing_datasets():
    """
    Get datasets for testing.
    
    Returns:
        list: List of testing dataset names
    """
    return TESTING_DATASETS.copy()

def get_comparison_datasets():
    """Get datasets for comparison analysis."""
    return [
        'iris',
        'breast_cancer',
        'wine',
        'digits_sklearn',
        'diabetes',
        'sonar',
        'glass',
        'vehicle',
        'ecoli',
        'yeast',
        'seeds',
        'thyroid',
        'pima',
        'ionosphere'
    ]

def get_biomathematics_datasets():
    """Get datasets related to Trends in Biomathematics: Modeling Health in Ecology, Social Interactions, and Cells."""
    return BIOMATHEMATICS_DATASETS

def get_dataset_categories():
    """
    Get all dataset categories.
    
    Returns:
        dict: Dictionary mapping category names to dataset lists
    """
    return DATASET_CATEGORIES.copy()

def validate_dataset_name(dataset_name):
    """
    Validate if a dataset name exists.
    
    Args:
        dataset_name (str): Name of the dataset to validate
    
    Returns:
        bool: True if dataset exists, False otherwise
    """
    return dataset_name in ALL_DATASETS

def get_dataset_category(dataset_name):
    """
    Get the category of a dataset.
    
    Args:
        dataset_name (str): Name of the dataset
    
    Returns:
        str: Category name or None if dataset not found
    """
    for category, datasets in DATASET_CATEGORIES.items():
        if dataset_name in datasets:
            return category
    return None 