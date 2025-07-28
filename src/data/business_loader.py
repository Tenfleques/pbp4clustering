#!/usr/bin/env python3
"""
Business Dataset Loader for Pseudo-Boolean Polynomial Dimensionality Reduction

This module provides a specialized loader for business datasets that inherits
from the base loader and implements business-specific processing logic.
"""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from sklearn.impute import SimpleImputer
from .base_loader import BaseDatasetLoader

class BusinessDatasetLoader(BaseDatasetLoader):
    """
    Loader for business datasets.
    
    This class handles loading and preprocessing of business datasets
    including customer churn, credit approval, and other business metrics.
    """
    
    def __init__(self, data_dir='./data'):
        super().__init__(data_dir)
        self.business_data_dir = Path('data/real_business')
    
    def load_dataset(self, dataset_name):
        """
        Load business dataset.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            dict: Dictionary containing X, y, metadata
        """
        print(f"Loading {dataset_name} dataset...")
        
        # Check if dataset is already saved
        saved_dataset = self.load_saved_dataset(dataset_name)
        if saved_dataset is not None:
            print(f"✓ Loaded cached {dataset_name} dataset")
            return saved_dataset
        
        # Check if raw data files exist
        if not self.business_data_dir.exists():
            print(f"Warning: Business data directory {self.business_data_dir} not found")
            print("Please download business data files to the specified directory")
            return None
        
        try:
            if dataset_name == 'churn':
                return self._load_churn_dataset()
            elif dataset_name == 'credit_approval':
                return self._load_credit_approval_dataset()
            else:
                print(f"Unknown business dataset: {dataset_name}")
                return None
                
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            return None
    
    def _load_churn_dataset(self):
        """Load customer churn dataset."""
        try:
            # Try to load from CSV file
            churn_file = self.business_data_dir / 'churn_data.csv'
            
            if not churn_file.exists():
                print(f"Churn data file not found: {churn_file}")
                return None
            
            # Load data
            df = pd.read_csv(churn_file)
            
            # Separate features and target
            feature_cols = [col for col in df.columns if col != 'exited']
            X = df[feature_cols].values
            y = df['exited'].values
            
            # Handle categorical variables
            categorical_cols = ['geography', 'gender']
            for col in categorical_cols:
                if col in df.columns:
                    # One-hot encode categorical variables
                    dummies = pd.get_dummies(df[col], prefix=col)
                    X = np.column_stack([X, dummies.values])
            
            # Reshape to 2D matrices for PBP
            # Reshape to 4x5 matrices (20 features)
            if X.shape[1] >= 20:
                X = X[:, :20]  # Take first 20 features
            else:
                # Pad with zeros if needed
                padding = np.zeros((X.shape[0], 20 - X.shape[1]))
                X = np.column_stack([X, padding])
            
            X = X.reshape(-1, 4, 5)
            
            # Create metadata
            metadata = {
                'feature_names': ['Demographics', 'Financial', 'Behavioral', 'Engagement'],
                'measurement_names': ['Customer_1', 'Customer_2', 'Customer_3', 'Customer_4', 'Customer_5'],
                'description': 'Customer churn prediction dataset',
                'data_type': 'business_real',
                'source': 'Synthetic business dataset',
                'reshaping_strategy': 'manual_reshape'
            }
            
            # Save dataset
            self.save_dataset('churn', X, y, metadata)
            
            return {
                'X': X,
                'y': y,
                'feature_names': metadata['feature_names'],
                'measurement_names': metadata['measurement_names'],
                'description': metadata['description'],
                'data_type': metadata['data_type'],
                'source': metadata['source'],
                'reshaping_strategy': metadata['reshaping_strategy']
            }
            
        except Exception as e:
            print(f"Error loading churn dataset: {e}")
            return None
    
    def _load_credit_approval_dataset(self):
        """Load credit approval dataset."""
        try:
            # Try to load from CSV file
            credit_file = self.business_data_dir / 'credit_approval_data.csv'
            
            if not credit_file.exists():
                print(f"Credit approval data file not found: {credit_file}")
                return None
            
            # Load data
            df = pd.read_csv(credit_file)
            
            # Separate features and target
            feature_cols = [col for col in df.columns if col != 'approval']
            X = df[feature_cols].values
            y = df['approval'].values
            
            # Handle missing values
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)
            
            # Reshape to 2D matrices for PBP
            # Reshape to 2x3 matrices (6 features)
            if X.shape[1] >= 6:
                X = X[:, :6]  # Take first 6 features
            else:
                # Pad with zeros if needed
                padding = np.zeros((X.shape[0], 6 - X.shape[1]))
                X = np.column_stack([X, padding])
            
            X = X.reshape(-1, 2, 3)
            
            # Create metadata
            metadata = {
                'feature_names': ['Personal', 'Financial'],
                'measurement_names': ['Credit_1', 'Credit_2', 'Credit_3'],
                'description': 'Credit card approval prediction dataset',
                'data_type': 'financial_real',
                'source': 'Financial institution dataset',
                'reshaping_strategy': 'manual_reshape'
            }
            
            # Save dataset
            self.save_dataset('credit_approval', X, y, metadata)
            
            return {
                'X': X,
                'y': y,
                'feature_names': metadata['feature_names'],
                'measurement_names': metadata['measurement_names'],
                'description': metadata['description'],
                'data_type': metadata['data_type'],
                'source': metadata['source'],
                'reshaping_strategy': metadata['reshaping_strategy']
            }
            
        except Exception as e:
            print(f"Error loading credit approval dataset: {e}")
            return None
    
    def load_all_datasets(self):
        """Load all business datasets."""
        datasets = {}
        
        business_datasets = ['churn', 'credit_approval']
        
        for dataset_name in business_datasets:
            try:
                dataset = self.load_dataset(dataset_name)
                if dataset is not None:
                    datasets[dataset_name] = dataset
                    print(f"✓ Loaded {dataset_name}: {dataset['X'].shape}")
                else:
                    print(f"✗ Failed to load {dataset_name}")
            except Exception as e:
                print(f"Error loading {dataset_name}: {e}")
        
        return datasets
    
    def get_dataset_info(self, dataset_name):
        """Get information about a specific dataset."""
        dataset = self.load_dataset(dataset_name)
        if dataset is not None:
            return {
                'name': dataset_name,
                'shape': dataset['X'].shape,
                'n_classes': len(np.unique(dataset['y'])),
                'data_type': dataset.get('data_type', 'business_real'),
                'description': dataset.get('description', ''),
                'source': dataset.get('source', 'unknown'),
                'reshaping_strategy': dataset.get('reshaping_strategy', 'unknown')
            }
        return None 