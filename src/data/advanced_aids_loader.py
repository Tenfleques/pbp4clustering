#!/usr/bin/env python3
"""
Advanced AIDS Screen Data Loader for Pseudo-Boolean Polynomial Dimensionality Reduction

This module provides a specialized loader for AIDS antiviral screen data that inherits
from the base loader and implements AIDS-specific processing logic.
"""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from sklearn.impute import SimpleImputer
from .base_loader import BaseDatasetLoader

class AdvancedAIDSLoader(BaseDatasetLoader):
    """
    Loader for AIDS antiviral screen data.
    
    This class handles loading and preprocessing of AIDS antiviral screen data
    including EC50, IC50, and concentration measurements.
    """
    
    def __init__(self, data_dir='./data'):
        super().__init__(data_dir)
        self.aids_data_dir = Path('data/real_medical/downloads/aids_screen')
    
    def load_dataset(self, dataset_name):
        """
        Load AIDS screen dataset.
        
        Args:
            dataset_name: Name of the dataset to load (should be 'aids_screen')
            
        Returns:
            dict: Dictionary containing X, y, metadata
        """
        if dataset_name != 'aids_screen':
            raise ValueError(f"Unknown AIDS dataset: {dataset_name}")
        
        print(f"Loading {dataset_name} dataset...")
        
        # Check if dataset is already saved
        saved_dataset = self.load_saved_dataset(dataset_name)
        if saved_dataset is not None:
            print(f"✓ Loaded cached {dataset_name} dataset")
            return saved_dataset
        
        # Check if raw data files exist
        if not self.aids_data_dir.exists():
            print(f"Warning: AIDS data directory {self.aids_data_dir} not found")
            print("Please download AIDS data files to the specified directory")
            return None
        
        try:
            # Load screening result files
            conc_file = self.aids_data_dir / 'aids_conc_may04.txt'
            ec50_file = self.aids_data_dir / 'aids_ec50_may04.txt'
            ic50_file = self.aids_data_dir / 'aids_ic50_may04.txt'
            
            if not all(f.exists() for f in [conc_file, ec50_file, ic50_file]):
                print("Warning: Some AIDS data files are missing")
                return None
            
            # Read with header row
            conc_df = pd.read_csv(conc_file, header=0)
            ec50_df = pd.read_csv(ec50_file, header=0)
            ic50_df = pd.read_csv(ic50_file, header=0)
            
            # Strip whitespace from all column names
            conc_df.columns = conc_df.columns.str.strip()
            ec50_df.columns = ec50_df.columns.str.strip()
            ic50_df.columns = ic50_df.columns.str.strip()
            
            # Merge all on NSC
            merged = conc_df.merge(ec50_df, on='NSC', how='left', suffixes=('', '_EC50'))
            merged = merged.merge(ic50_df, on='NSC', how='left', suffixes=('', '_IC50'))
            
            # Convert Conclusion to integer label
            result_map = {'CA': 2, 'CM': 1, 'CI': 0}
            merged['Screening_Label'] = merged['Conclusion'].str.strip().map(result_map)
            merged = merged[~merged['Screening_Label'].isna()]
            
            # Select features for matrix
            features = [
                'Log10EC50',
                'Log10IC50',
                'NumExp',  # from EC50
                'NumExp_IC50',  # from IC50
                'StdDev',      # from EC50
                'StdDev_IC50', # from IC50
                'Screening_Label'
            ]
            
            # Coerce all features to numeric
            for f in features:
                merged[f] = pd.to_numeric(merged[f], errors='coerce')
            
            # Create flat feature matrix
            X_flat = merged[features].to_numpy(dtype=np.float32)
            y = merged['Screening_Label'].to_numpy(dtype=np.int32)
            
            # Handle NaN values by imputing with median
            imputer = SimpleImputer(strategy='median')
            X_flat = imputer.fit_transform(X_flat)
            
            # Reshape to matrix format for PBP (samples, 2, 4)
            # Pad to 8 features if needed, then reshape to 2x4
            if X_flat.shape[1] < 8:
                # Pad with zeros to make it 8 features
                padding = 8 - X_flat.shape[1]
                X_flat = np.pad(X_flat, ((0, 0), (0, padding)), mode='constant')
            elif X_flat.shape[1] > 8:
                # Truncate to 8 features
                X_flat = X_flat[:, :8]
            
            # Reshape to (samples, 2, 4) matrix format
            X = X_flat.reshape(-1, 2, 4)
            
            # Create metadata
            metadata = {
                'description': 'AIDS Antiviral Screen: features=[Log10EC50, Log10IC50, NumExp_EC50, NumExp_IC50, StdDev_EC50, StdDev_IC50, Screening_Label] reshaped to 2x4 matrices',
                'feature_names': ['EC50_Features', 'IC50_Features'],
                'measurement_names': ['Measurement_1', 'Measurement_2', 'Measurement_3', 'Measurement_4'],
                'target_names': ['CI', 'CM', 'CA'],
                'data_type': 'antiviral_real',
                'domain': 'drug_discovery',
                'original_shape': X_flat.shape,
                'matrix_shape': X.shape[1:],
                'source': 'AIDS_Antiviral_Screen',
                'reshaping_strategy': 'direct_2x4'
            }
            
            dataset_dict = {
                'X': X,
                'y': y,
                'metadata': metadata
            }
            
            # Save the dataset
            self.save_dataset(dataset_name, dataset_dict)
            
            print(f"✓ Loaded {dataset_name}: {X.shape[0]} samples, {X.shape[1]}x{X.shape[2]} matrices")
            
            return dataset_dict
            
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            return None
    
    def load_all_datasets(self):
        """Load all AIDS datasets."""
        results = {}
        try:
            results['aids_screen'] = self.load_dataset('aids_screen')
        except Exception as e:
            print(f"Error loading AIDS datasets: {e}")
            results['aids_screen'] = None
        return results 