#!/usr/bin/env python3
"""
Standard Dataset Loader for sklearn datasets

This module provides a specialized loader for standard sklearn datasets
that inherits from the base loader and implements standard dataset loading.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits, load_diabetes
from .base_loader import BaseDatasetLoader

class StandardDatasetLoader(BaseDatasetLoader):
    """
    Loader for standard sklearn datasets.
    
    This class handles loading and preprocessing of standard sklearn datasets
    including Iris, Breast Cancer, Wine, Digits, and Diabetes datasets.
    """
    
    def __init__(self, data_dir='./data'):
        super().__init__(data_dir)
        self.dataset_loaders = {
            'iris': load_iris,
            'breast_cancer': load_breast_cancer,
            'wine': load_wine,
            'digits': load_digits,
            'diabetes': load_diabetes,
            'breast_cancer_sklearn': load_breast_cancer,
            'digits_sklearn': load_digits,
            'wine_sklearn': load_wine
        }
    
    def load_dataset(self, dataset_name):
        """
        Load a standard sklearn dataset.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            dict: Dictionary containing X, y, metadata
        """
        if dataset_name not in self.dataset_loaders:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        print(f"Loading {dataset_name} dataset...")
        
        # Load the dataset
        dataset = self.dataset_loaders[dataset_name]()
        
        X = dataset.data
        y = dataset.target
        
        # Handle diabetes dataset (regression)
        if dataset_name == 'diabetes':
            # Convert to classification by discretizing
            y = np.digitize(y, bins=[np.percentile(y, 33), np.percentile(y, 66)])
        
        # Apply adaptive smart reshaping
        if X.shape[1] > 2:  # Only reshape if we have more than 2 features
            X_reshaped, feature_groups, _, strategy, _ = self.adaptive_smart_reshape(X, target_rows=2)
        else:
            # For datasets with 2 or fewer features, pad to 2x2
            if X.shape[1] == 1:
                X_reshaped = np.column_stack([X, np.zeros_like(X)])
            else:
                X_reshaped = X
            X_reshaped = X_reshaped.reshape(-1, 1, X_reshaped.shape[1])
            feature_groups = [np.arange(X_reshaped.shape[2])]
            strategy = 'direct'
        
        # Create metadata
        metadata = {
            'description': f'{dataset_name} dataset from sklearn',
            'feature_names': dataset.feature_names if hasattr(dataset, 'feature_names') else [f'feature_{i}' for i in range(X.shape[1])],
            'measurement_names': [f'measurement_{i}' for i in range(X_reshaped.shape[2])],
            'target_names': dataset.target_names if hasattr(dataset, 'target_names') else [f'class_{i}' for i in range(len(np.unique(y)))],
            'data_type': 'standard_sklearn',
            'domain': 'general',
            'original_shape': X.shape,
            'matrix_shape': X_reshaped.shape[1:],
            'source': 'sklearn',
            'reshaping_strategy': strategy
        }
        
        dataset_dict = {
            'X': X_reshaped,
            'y': y,
            'metadata': metadata
        }
        
        # Save the dataset
        self.save_dataset(dataset_name, dataset_dict)
        
        print(f"✓ Loaded {dataset_name}: {X_reshaped.shape[0]} samples, {X_reshaped.shape[1]}x{X_reshaped.shape[2]} matrices")
        
        return dataset_dict
    
    def load_all_datasets(self):
        """Load all standard datasets."""
        results = {}
        for dataset_name in self.dataset_loaders.keys():
            try:
                results[dataset_name] = self.load_dataset(dataset_name)
            except Exception as e:
                print(f"Error loading {dataset_name}: {e}")
                results[dataset_name] = None
        return results 