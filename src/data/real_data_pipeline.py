#!/usr/bin/env python3
"""
Real Data Pipeline for Pseudo-Boolean Polynomial Dimensionality Reduction

This module handles downloading and processing real datasets from various public sources.
All datasets are processed into matrix format suitable for PBP analysis.
"""

import pandas as pd
import numpy as np
import requests
import os
import json
import io
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import fetch_openml, load_breast_cancer, load_wine, load_digits
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger().setLevel(logging.ERROR)

class RealDataPipeline:
    """Downloads and processes real datasets for PBP analysis."""
    
    def __init__(self, data_dir='./data/real'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.datasets = {}
        self.download_cache = self.data_dir / 'downloads'
        self.download_cache.mkdir(exist_ok=True)
        
    def download_openml_datasets(self):
        """Download various real datasets from OpenML."""
        print("Downloading real datasets from OpenML...")
        
        # List of well-known OpenML datasets with their IDs and expected characteristics
        openml_datasets = [
            {'id': 31, 'name': 'credit_approval', 'type': 'financial'},
            {'id': 37, 'name': 'diabetes', 'type': 'medical'},
            {'id': 40701, 'name': 'churn', 'type': 'business'},
            {'id': 1049, 'name': 'pc4', 'type': 'software'},
            {'id': 1068, 'name': 'adult', 'type': 'social'}
        ]
        
        downloaded_datasets = {}
        
        for dataset_info in openml_datasets:
            try:
                print(f"\nDownloading {dataset_info['name']} (ID: {dataset_info['id']})...")
                
                # Download dataset
                dataset = fetch_openml(data_id=dataset_info['id'], as_frame=True, parser='auto')
                
                if dataset.data is None or dataset.target is None:
                    print(f"Failed to load dataset {dataset_info['name']}: No data found")
                    continue
                
                X = dataset.data
                y = dataset.target
                
                # Handle different data types
                if hasattr(X, 'values'):
                    X_values = X.values
                else:
                    X_values = X
                
                if hasattr(y, 'values'):
                    y_values = y.values
                else:
                    y_values = y
                
                # Convert target to numeric if needed
                if y_values.dtype == 'object' or isinstance(y_values[0], str):
                    y_numeric = pd.Categorical(y_values).codes
                else:
                    y_numeric = y_values.astype(int)
                
                print(f"Raw data shape: {X_values.shape}, Target classes: {len(np.unique(y_numeric))}")
                
                # Process the dataset
                processed_dataset = self._process_dataset(
                    X_values, y_numeric, 
                    dataset_info['name'], 
                    dataset_info['type']
                )
                
                if processed_dataset:
                    downloaded_datasets[dataset_info['name']] = processed_dataset
                    print(f"Successfully processed {dataset_info['name']}")
                
            except Exception as e:
                print(f"Error downloading {dataset_info['name']}: {e}")
                continue
        
        return downloaded_datasets
    
    def download_sklearn_datasets(self):
        """Download real datasets from scikit-learn."""
        print("\nDownloading real datasets from scikit-learn...")
        
        sklearn_datasets = [
            {'loader': load_breast_cancer, 'name': 'breast_cancer_sklearn', 'type': 'medical'},
            {'loader': load_wine, 'name': 'wine_sklearn', 'type': 'chemical'},
            {'loader': load_digits, 'name': 'digits_sklearn', 'type': 'image'}
        ]
        
        downloaded_datasets = {}
        
        for dataset_info in sklearn_datasets:
            try:
                print(f"Loading {dataset_info['name']}...")
                
                # Load dataset
                dataset = dataset_info['loader']()
                X = dataset.data
                y = dataset.target
                
                print(f"Raw data shape: {X.shape}, Target classes: {len(np.unique(y))}")
                
                # Process the dataset
                processed_dataset = self._process_dataset(
                    X, y, 
                    dataset_info['name'], 
                    dataset_info['type']
                )
                
                if processed_dataset:
                    downloaded_datasets[dataset_info['name']] = processed_dataset
                    print(f"Successfully processed {dataset_info['name']}")
                
            except Exception as e:
                print(f"Error loading {dataset_info['name']}: {e}")
                continue
        
        return downloaded_datasets
    
    def download_uci_datasets(self):
        """Download datasets directly from UCI repository."""
        print("\nDownloading datasets from UCI repository...")
        
        uci_datasets = [
            {
                'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                'name': 'iris_uci',
                'type': 'botanical',
                'columns': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
            },
            {
                'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
                'name': 'wine_quality_red',
                'type': 'chemical',
                'separator': ';'
            }
        ]
        
        downloaded_datasets = {}
        
        for dataset_info in uci_datasets:
            try:
                print(f"Downloading {dataset_info['name']}...")
                
                # Download data
                response = requests.get(dataset_info['url'])
                response.raise_for_status()
                
                # Parse CSV
                if 'separator' in dataset_info:
                    data = pd.read_csv(io.StringIO(response.text), sep=dataset_info['separator'])
                elif 'columns' in dataset_info:
                    data = pd.read_csv(io.StringIO(response.text), header=None, names=dataset_info['columns'])
                else:
                    data = pd.read_csv(io.StringIO(response.text))
                
                # Separate features and target
                if dataset_info['name'] == 'iris_uci':
                    X = data.iloc[:, :-1].values
                    y = pd.Categorical(data.iloc[:, -1]).codes
                elif dataset_info['name'] == 'wine_quality_red':
                    X = data.iloc[:, :-1].values
                    y = data.iloc[:, -1].values
                    # Convert quality scores to classes (e.g., low, medium, high)
                    y = np.digitize(y, bins=[3, 6, 8]) - 1
                
                print(f"Raw data shape: {X.shape}, Target classes: {len(np.unique(y))}")
                
                # Process the dataset
                processed_dataset = self._process_dataset(
                    X, y, 
                    dataset_info['name'], 
                    dataset_info['type']
                )
                
                if processed_dataset:
                    downloaded_datasets[dataset_info['name']] = processed_dataset
                    print(f"Successfully processed {dataset_info['name']}")
                
            except Exception as e:
                print(f"Error downloading {dataset_info['name']}: {e}")
                continue
        
        return downloaded_datasets
    
    def _process_dataset(self, X, y, dataset_name, data_type):
        """Process any dataset into matrix format."""
        try:
            # Handle missing values and ensure numeric data
            X_clean = self._handle_missing_values(X)
            
            if X_clean is None or X_clean.shape[1] == 0:
                print(f"No valid numeric features found in {dataset_name}")
                return None
            
            n_features = X_clean.shape[1]
            print(f"Processing {dataset_name}: {X_clean.shape[0]} samples, {n_features} features")
            
            # Determine optimal matrix structure
            matrix_shape, n_keep, feature_names = self._determine_matrix_structure(n_features, data_type)
            
            if matrix_shape is None:
                print(f"Cannot create matrix structure for {dataset_name} with {n_features} features")
                return None
            
            # Select and reshape features
            X_subset = X_clean[:, :n_keep]
            X_matrices = X_subset.reshape(-1, matrix_shape[0], matrix_shape[1])
            
            measurement_names = [f'Measurement_{i+1}' for i in range(matrix_shape[1])]
            
            # Create meaningful target names
            unique_classes = np.unique(y)
            target_names = [f'{data_type.title()}_Class_{i}' for i in unique_classes]
            
            dataset = {
                'X': X_matrices,
                'y': y,
                'feature_names': feature_names,
                'measurement_names': measurement_names,
                'target_names': target_names,
                'description': f'Real {data_type} data reshaped to {matrix_shape[0]}x{matrix_shape[1]} matrices',
                'data_type': f'{data_type}_real',
                'preprocessing': 'missing_values_handled_standardized',
                'original_shape': X.shape,
                'matrix_shape': matrix_shape
            }
            
            self.datasets[dataset_name] = dataset
            self._save_dataset(dataset_name, dataset)
            
            return dataset
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            return None
    
    def _determine_matrix_structure(self, n_features, data_type):
        """Determine optimal matrix structure based on number of features and data type."""
        
        # Define matrix structures in order of preference
        structures = [
            (5, 6, 30),  # 5x6 = 30 features
            (4, 8, 32),  # 4x8 = 32 features  
            (4, 6, 24),  # 4x6 = 24 features
            (3, 8, 24),  # 3x8 = 24 features
            (4, 5, 20),  # 4x5 = 20 features
            (3, 6, 18),  # 3x6 = 18 features
            (3, 5, 15),  # 3x5 = 15 features
            (3, 4, 12),  # 3x4 = 12 features
            (2, 6, 12),  # 2x6 = 12 features
            (2, 5, 10),  # 2x5 = 10 features
            (2, 4, 8),   # 2x4 = 8 features
            (2, 3, 6),   # 2x3 = 6 features
        ]
        
        # Find the best fitting structure
        for rows, cols, required_features in structures:
            if n_features >= required_features:
                matrix_shape = (rows, cols)
                n_keep = required_features
                feature_names = self._get_feature_names(rows, data_type)
                return matrix_shape, n_keep, feature_names
        
        return None, None, None
    
    def _get_feature_names(self, n_rows, data_type):
        """Generate meaningful feature names based on data type and number of rows."""
        
        feature_name_mapping = {
            'medical': {
                4: ['Vital_Signs', 'Lab_Results', 'Symptoms', 'Demographics'],
                3: ['Clinical_Measures', 'Patient_History', 'Risk_Factors'],
                2: ['Physical_Features', 'Medical_History']
            },
            'financial': {
                4: ['Liquidity_Ratios', 'Profitability_Ratios', 'Leverage_Ratios', 'Efficiency_Ratios'],
                3: ['Financial_Health', 'Performance_Metrics', 'Risk_Indicators'],
                2: ['Income_Metrics', 'Asset_Metrics']
            },
            'chemical': {
                4: ['Organic_Compounds', 'Minerals', 'Acidity_Measures', 'Aromatic_Compounds'],
                3: ['Primary_Components', 'Secondary_Components', 'Trace_Elements'],
                2: ['Major_Constituents', 'Minor_Constituents']
            },
            'botanical': {
                4: ['Morphological', 'Structural', 'Dimensional', 'Textural'],
                3: ['Sepal_Features', 'Petal_Features', 'Overall_Features'],
                2: ['Length_Measures', 'Width_Measures']
            },
            'image': {
                4: ['Intensity_Features', 'Texture_Features', 'Shape_Features', 'Spatial_Features'],
                3: ['Low_Level_Features', 'Mid_Level_Features', 'High_Level_Features'],
                2: ['Pixel_Intensity', 'Spatial_Pattern']
            },
            'business': {
                4: ['Customer_Behavior', 'Service_Usage', 'Demographics', 'Account_Features'],
                3: ['Usage_Patterns', 'Customer_Profile', 'Service_Metrics'],
                2: ['Behavioral_Features', 'Account_Features']
            }
        }
        
        # Default generic names if data type not found
        if data_type not in feature_name_mapping or n_rows not in feature_name_mapping[data_type]:
            return [f'Feature_Group_{i+1}' for i in range(n_rows)]
        
        return feature_name_mapping[data_type][n_rows]
    
    def _handle_missing_values(self, X):
        """Handle missing values and ensure numeric data."""
        try:
            # Convert to DataFrame for easier handling
            if not isinstance(X, pd.DataFrame):
                X_df = pd.DataFrame(X)
            else:
                X_df = X.copy()
            
            # Convert object columns to numeric where possible
            numeric_columns = []
            for col in X_df.columns:
                try:
                    # Try to convert to numeric
                    numeric_col = pd.to_numeric(X_df[col], errors='coerce')
                    if not numeric_col.isna().all():  # If at least some values are numeric
                        numeric_columns.append(numeric_col)
                except:
                    continue
            
            if not numeric_columns:
                return None
            
            # Combine numeric columns
            X_numeric = pd.concat(numeric_columns, axis=1)
            
            # Handle missing values
            if X_numeric.isna().any().any():
                # Use median for missing values
                X_numeric = X_numeric.fillna(X_numeric.median())
            
            # Convert to numpy array
            return X_numeric.values
            
        except Exception as e:
            print(f"Error handling missing values: {e}")
            return None
    
    def _save_dataset(self, dataset_name, dataset):
        """Save dataset to files."""
        try:
            # Save X and y as numpy arrays
            np.save(self.data_dir / f"{dataset_name}_X.npy", dataset['X'])
            np.save(self.data_dir / f"{dataset_name}_y.npy", dataset['y'])
            
            # Save metadata as JSON
            metadata = {
                'description': dataset.get('description', ''),
                'feature_names': dataset.get('feature_names', []),
                'measurement_names': dataset.get('measurement_names', []),
                'target_names': dataset.get('target_names', []),
                'data_type': dataset.get('data_type', ''),
                'preprocessing': dataset.get('preprocessing', ''),
                'original_shape': [int(x) for x in dataset.get('original_shape', [])],
                'matrix_shape': [int(x) for x in dataset.get('matrix_shape', [])],
                'shape': [int(x) for x in dataset['X'].shape],
                'n_classes': int(len(np.unique(dataset['y'])))
            }
            
            with open(self.data_dir / f"{dataset_name}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"  → Saved to {self.data_dir}")
            
        except Exception as e:
            print(f"Error saving dataset {dataset_name}: {e}")
    
    def download_all_real_datasets(self):
        """Download all available real datasets."""
        print("=== Real Data Download Pipeline ===\n")
        
        all_datasets = {}
        
        # Download from multiple sources
        print("Phase 1: Downloading from scikit-learn...")
        sklearn_datasets = self.download_sklearn_datasets()
        all_datasets.update(sklearn_datasets)
        
        print("\nPhase 2: Downloading from UCI...")
        uci_datasets = self.download_uci_datasets()
        all_datasets.update(uci_datasets)
        
        print("\nPhase 3: Downloading from OpenML...")
        openml_datasets = self.download_openml_datasets()
        all_datasets.update(openml_datasets)
        
        print(f"\n=== Successfully downloaded {len(all_datasets)} real datasets ===")
        return all_datasets
    
    def get_dataset_info(self, dataset_name):
        """Get information about a dataset."""
        if dataset_name not in self.datasets:
            return None
        
        dataset = self.datasets[dataset_name]
        info = {
            'name': dataset_name,
            'shape': dataset['X'].shape,
            'n_samples': dataset['X'].shape[0],
            'matrix_shape': dataset.get('matrix_shape', (dataset['X'].shape[1], dataset['X'].shape[2])),
            'original_shape': dataset.get('original_shape', 'Unknown'),
            'n_classes': len(np.unique(dataset['y'])),
            'description': dataset.get('description', 'No description available'),
            'feature_names': dataset.get('feature_names', []),
            'measurement_names': dataset.get('measurement_names', []),
            'target_names': dataset.get('target_names', []),
            'data_type': dataset.get('data_type', 'unknown'),
            'preprocessing': dataset.get('preprocessing', 'none')
        }
        
        return info


def main():
    """Main function to demonstrate real data pipeline."""
    print("=== Real Data Pipeline Demo ===\n")
    
    pipeline = RealDataPipeline()
    
    # Download all real datasets
    datasets = pipeline.download_all_real_datasets()
    
    print("\n" + "="*80)
    print("REAL DATASET SUMMARIES")
    print("="*80)
    
    # Display information about each dataset
    for name, dataset in datasets.items():
        info = pipeline.get_dataset_info(name)
        if info:
            print(f"\n{name.upper().replace('_', ' ')}:")
            print(f"  Description: {info['description']}")
            print(f"  Original shape: {info['original_shape']}")
            print(f"  Processed shape: {info['shape']} (samples × rows × columns)")
            print(f"  Matrix structure: {info['matrix_shape']}")
            print(f"  Number of classes: {info['n_classes']}")
            print(f"  Data type: {info['data_type']}")
            print(f"  Feature names: {info['feature_names']}")
            
            # Show sample matrix
            if dataset['X'].size > 0:
                print(f"\n  Sample matrix (first sample):")
                sample_matrix = dataset['X'][0]
                for i, row_name in enumerate(info['feature_names']):
                    values = [f"{val:.3f}" for val in sample_matrix[i][:3]]  # Show first 3 values
                    print(f"    {row_name:20}: [{', '.join(values)}...]")
            
            # Show class distribution
            print(f"  Class distribution:")
            unique, counts = np.unique(dataset['y'], return_counts=True)
            for cls, count in zip(unique, counts):
                class_name = info['target_names'][cls] if cls < len(info['target_names']) else f"Class_{cls}"
                print(f"    {class_name}: {count} samples")
    
    print(f"\n{'='*80}")
    print(f"Pipeline completed: {len(datasets)} real datasets downloaded and processed")
    print(f"Data saved in: {pipeline.data_dir}")
    print("="*80)


if __name__ == "__main__":
    main() 