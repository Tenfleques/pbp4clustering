#!/usr/bin/env python3
"""
Large Dataset Loader for Pseudo-Boolean Polynomial Dimensionality Reduction

This module provides a specialized loader for large datasets that inherits
from the base loader and implements large dataset loading logic.
"""

import pandas as pd
import numpy as np
import requests
import os
from pathlib import Path
from sklearn.datasets import fetch_covtype, fetch_kddcup99
from sklearn.preprocessing import LabelEncoder
from .base_loader import BaseDatasetLoader

class LargeDatasetLoader(BaseDatasetLoader):
    """
    Loader for large datasets.
    
    This class handles loading and preprocessing of large datasets including
    Covertype, KDD Cup 99, Olivetti faces, Linnerrud, and Species distribution.
    """
    
    def __init__(self, data_dir='./data'):
        super().__init__(data_dir)
        self.large_datasets = {
            'covertype': {
                'name': 'Covertype',
                'description': 'Forest cover type prediction',
                'source': 'sklearn'
            },
            'kddcup99': {
                'name': 'KDD Cup 99',
                'description': 'Network intrusion detection',
                'source': 'sklearn'
            },
            'linnerrud': {
                'name': 'Linnerrud',
                'description': 'Physical exercise dataset',
                'source': 'sklearn'
            },
            'species_distribution': {
                'name': 'Species Distribution',
                'description': 'Species distribution modeling',
                'source': 'sklearn'
            }
        }
    
    def load_dataset(self, dataset_name):
        """
        Load a large dataset.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            dict: Dictionary containing X, y, metadata
        """
        if dataset_name not in self.large_datasets:
            raise ValueError(f"Unknown large dataset: {dataset_name}")
        
        print(f"Loading {dataset_name} dataset...")
        
        # Check if dataset is already saved
        saved_dataset = self.load_saved_dataset(dataset_name)
        if saved_dataset is not None:
            print(f"✓ Loaded cached {dataset_name} dataset")
            return saved_dataset
        
        try:
            if dataset_name == 'covertype':
                return self._load_covertype()
            elif dataset_name == 'kddcup99':
                return self._load_kddcup99()
            elif dataset_name == 'linnerrud':
                return self._load_linnerrud()
            elif dataset_name == 'species_distribution':
                return self._load_species_distribution()
            else:
                print(f"Unknown large dataset: {dataset_name}")
                return None
                
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            return None
    
    def _load_covertype(self):
        """Load Covertype dataset."""
        try:
            print("Downloading Covertype dataset...")
            dataset = fetch_covtype()
            
            X = dataset.data
            y = dataset.target
            
            # Apply adaptive smart reshaping
            if X.shape[1] > 3:
                X_reshaped, feature_groups, _, strategy, _ = self.adaptive_smart_reshape(X, target_rows=3)
            else:
                # For datasets with 3 or fewer features, pad to 2x2
                if X.shape[1] == 1:
                    X_reshaped = np.column_stack([X, np.zeros_like(X)])
                elif X.shape[1] == 2:
                    X_reshaped = np.column_stack([X, np.zeros((X.shape[0], 2))])
                else:
                    X_reshaped = np.column_stack([X, np.zeros((X.shape[0], 1))])
                X_reshaped = X_reshaped.reshape(-1, 2, X_reshaped.shape[1] // 2)
                feature_groups = [np.arange(X_reshaped.shape[2])]
                strategy = 'direct'
            
            # Create metadata
            metadata = {
                'description': 'Covertype dataset: Forest cover type prediction',
                'feature_names': [f'feature_{i}' for i in range(X.shape[1])],
                'measurement_names': [f'measurement_{i}' for i in range(X_reshaped.shape[2])],
                'target_names': [f'class_{i}' for i in range(len(np.unique(y)))],
                'data_type': 'large_sklearn',
                'domain': 'environmental',
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
            self.save_dataset('covertype', dataset_dict)
            
            print(f"✓ Loaded covertype: {X_reshaped.shape[0]} samples, {X_reshaped.shape[1]}x{X_reshaped.shape[2]} matrices")
            
            return dataset_dict
            
        except Exception as e:
            print(f"Error loading Covertype dataset: {e}")
            return None
    
    def _load_kddcup99(self):
        """Load KDD Cup 99 dataset."""
        try:
            print("Downloading KDD Cup 99 dataset...")
            dataset = fetch_kddcup99()
            
            X = dataset.data
            y = dataset.target
            
            # Convert string targets to numeric
            le = LabelEncoder()
            y = le.fit_transform(y)
            
            # Apply adaptive smart reshaping
            if X.shape[1] > 3:
                X_reshaped, feature_groups, _, strategy, _ = self.adaptive_smart_reshape(X, target_rows=3)
            else:
                # For datasets with 3 or fewer features, pad to 2x2
                if X.shape[1] == 1:
                    X_reshaped = np.column_stack([X, np.zeros_like(X)])
                elif X.shape[1] == 2:
                    X_reshaped = np.column_stack([X, np.zeros((X.shape[0], 2))])
                else:
                    X_reshaped = np.column_stack([X, np.zeros((X.shape[0], 1))])
                X_reshaped = X_reshaped.reshape(-1, 2, X_reshaped.shape[1] // 2)
                feature_groups = [np.arange(X_reshaped.shape[2])]
                strategy = 'direct'
            
            # Create metadata
            metadata = {
                'description': 'KDD Cup 99 dataset: Network intrusion detection',
                'feature_names': [f'feature_{i}' for i in range(X.shape[1])],
                'measurement_names': [f'measurement_{i}' for i in range(X_reshaped.shape[2])],
                'target_names': list(le.classes_),
                'data_type': 'large_sklearn',
                'domain': 'cybersecurity',
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
            self.save_dataset('kddcup99', dataset_dict)
            
            print(f"✓ Loaded kddcup99: {X_reshaped.shape[0]} samples, {X_reshaped.shape[1]}x{X_reshaped.shape[2]} matrices")
            
            return dataset_dict
            
        except Exception as e:
            print(f"Error loading KDD Cup 99 dataset: {e}")
            return None
    
    def _load_linnerrud(self):
        """Load Linnerrud dataset."""
        try:
            print("Downloading Linnerrud dataset...")
            from sklearn.datasets import load_linnerud
            dataset = load_linnerud()
            
            X = dataset.data
            y = dataset.target
            
            # Convert to classification by discretizing targets
            y_discrete = np.digitize(y[:, 0], bins=[np.percentile(y[:, 0], 33), np.percentile(y[:, 0], 66)])
            
            # Apply adaptive smart reshaping
            if X.shape[1] > 3:
                X_reshaped, feature_groups, _, strategy, _ = self.adaptive_smart_reshape(X, target_rows=3)
            else:
                # For datasets with 3 or fewer features, pad to 2x2
                if X.shape[1] == 1:
                    X_reshaped = np.column_stack([X, np.zeros_like(X)])
                elif X.shape[1] == 2:
                    X_reshaped = np.column_stack([X, np.zeros((X.shape[0], 2))])
                else:
                    X_reshaped = np.column_stack([X, np.zeros((X.shape[0], 1))])
                X_reshaped = X_reshaped.reshape(-1, 2, X_reshaped.shape[1] // 2)
                feature_groups = [np.arange(X_reshaped.shape[2])]
                strategy = 'direct'
            
            # Create metadata
            metadata = {
                'description': 'Linnerrud dataset: Physical exercise measurements',
                'feature_names': dataset.feature_names,
                'measurement_names': [f'measurement_{i}' for i in range(X_reshaped.shape[2])],
                'target_names': ['Low_Exercise', 'Medium_Exercise', 'High_Exercise'],
                'data_type': 'large_sklearn',
                'domain': 'health_fitness',
                'original_shape': X.shape,
                'matrix_shape': X_reshaped.shape[1:],
                'source': 'sklearn',
                'reshaping_strategy': strategy
            }
            
            dataset_dict = {
                'X': X_reshaped,
                'y': y_discrete,
                'metadata': metadata
            }
            
            # Save the dataset
            self.save_dataset('linnerrud', dataset_dict)
            
            print(f"✓ Loaded linnerrud: {X_reshaped.shape[0]} samples, {X_reshaped.shape[1]}x{X_reshaped.shape[2]} matrices")
            
            return dataset_dict
            
        except Exception as e:
            print(f"Error loading Linnerrud dataset: {e}")
            return None
    
    def _load_species_distribution(self):
        """Load Species distribution dataset."""
        try:
            print("Downloading Species distribution dataset...")
            # Create synthetic species distribution data
            n_samples = 1000
            n_features = 20
            
            # Create synthetic environmental features
            np.random.seed(42)
            X = np.random.randn(n_samples, n_features)
            
            # Create synthetic species presence/absence
            y = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
            
            # Apply adaptive smart reshaping
            if X.shape[1] > 3:
                X_reshaped, feature_groups, _, strategy, _ = self.adaptive_smart_reshape(X, target_rows=3)
            else:
                # For datasets with 3 or fewer features, pad to 2x2
                if X.shape[1] == 1:
                    X_reshaped = np.column_stack([X, np.zeros_like(X)])
                elif X.shape[1] == 2:
                    X_reshaped = np.column_stack([X, np.zeros((X.shape[0], 2))])
                else:
                    X_reshaped = np.column_stack([X, np.zeros((X.shape[0], 1))])
                X_reshaped = X_reshaped.reshape(-1, 2, X_reshaped.shape[1] // 2)
                feature_groups = [np.arange(X_reshaped.shape[2])]
                strategy = 'direct'
            
            # Create metadata
            metadata = {
                'description': 'Species distribution dataset: Environmental modeling',
                'feature_names': [f'environmental_feature_{i}' for i in range(X.shape[1])],
                'measurement_names': [f'measurement_{i}' for i in range(X_reshaped.shape[2])],
                'target_names': ['Absent', 'Present'],
                'data_type': 'large_synthetic',
                'domain': 'ecology',
                'original_shape': X.shape,
                'matrix_shape': X_reshaped.shape[1:],
                'source': 'synthetic',
                'reshaping_strategy': strategy
            }
            
            dataset_dict = {
                'X': X_reshaped,
                'y': y,
                'metadata': metadata
            }
            
            # Save the dataset
            self.save_dataset('species_distribution', dataset_dict)
            
            print(f"✓ Loaded species_distribution: {X_reshaped.shape[0]} samples, {X_reshaped.shape[1]}x{X_reshaped.shape[2]} matrices")
            
            return dataset_dict
            
        except Exception as e:
            print(f"Error loading Species distribution dataset: {e}")
            return None
    
    def load_all_datasets(self):
        """Load all large datasets."""
        results = {}
        for dataset_name in self.large_datasets.keys():
            try:
                results[dataset_name] = self.load_dataset(dataset_name)
            except Exception as e:
                print(f"Error loading {dataset_name}: {e}")
                results[dataset_name] = None
        return results 