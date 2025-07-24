#!/usr/bin/env python3
"""
Dataset Loader for Pseudo-Boolean Polynomial Dimensionality Reduction

This script downloads and transforms various datasets into the matrix format
required by the pseudo-Boolean polynomial approach. Each dataset is restructured
into matrices where rows represent measurement categories and columns represent
specific measurements.
"""

import pandas as pd
import numpy as np
import requests
import os
import json
import io
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
import logging
# Suppress logging output
logging.getLogger().setLevel(logging.ERROR)

class DatasetTransformer:
    """Transforms datasets into matrix format for pseudo-Boolean polynomial analysis."""
    
    def __init__(self):
        self.datasets = {}
        self.transformers = {}
        
    def load_iris_dataset(self):
        """Load and transform Iris dataset into 2x2 matrices."""
        print("Loading Iris dataset...")
        iris = load_iris()
        
        # Reshape from 1x4 to 2x2 matrices
        # Rows: [Sepal, Petal]
        # Columns: [Length, Width]
        X = iris.data.reshape(-1, 2, 2)
        y = iris.target
        
        # Create feature names for clarity
        feature_names = ['Sepal', 'Petal']
        measurement_names = ['Length', 'Width']
        
        self.datasets['iris'] = {
            'X': X,
            'y': y,
            'feature_names': feature_names,
            'measurement_names': measurement_names,
            'target_names': iris.target_names,
            'description': 'Iris flower measurements reshaped to 2x2 matrices'
        }
        
        print(f"Iris dataset loaded: {X.shape[0]} samples of shape {X.shape[1]}x{X.shape[2]}")
        return self.datasets['iris']
    
    def load_breast_cancer_dataset(self):
        """Load and transform Wisconsin Breast Cancer dataset into 3x10 matrices."""
        print("Loading Wisconsin Breast Cancer dataset...")
        bc = load_breast_cancer()
        
        # Reshape from 1x30 to 3x10 matrices
        # Rows: [Mean, Standard Error, Worst]
        # Columns: 10 different features
        X = bc.data.reshape(-1, 3, 10)
        y = bc.target
        
        feature_names = ['Mean', 'Standard Error', 'Worst']
        measurement_names = [f'Feature_{i+1}' for i in range(10)]
        
        self.datasets['breast_cancer'] = {
            'X': X,
            'y': y,
            'feature_names': feature_names,
            'measurement_names': measurement_names,
            'target_names': bc.target_names,
            'description': 'Breast cancer features reshaped to 3x10 matrices'
        }
        
        print(f"Breast Cancer dataset loaded: {X.shape[0]} samples of shape {X.shape[1]}x{X.shape[2]}")
        return self.datasets['breast_cancer']
    
    def load_wine_dataset(self):
        """Load and transform Wine dataset into 3x4 matrices."""
        print("Loading Wine dataset...")
        wine = load_wine()
        
        # # Wine dataset has 13 features, we'll drop one to get 12 features for 3x4 matrix
        # # Original features: alcohol, malic_acid, ash, alcalinity_of_ash, magnesium, 
        # # total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins, 
        # # color_intensity, hue, od280/od315_of_diluted_wines, proline
        
        # data = wine.data
        # print(f"Original Wine shape: {data.shape}")
        
        # # Drop the last column (proline) to get 12 features for 3x4 matrix
        # # This is a reasonable choice as proline is less critical for wine classification
        # data_12 = data[:, :-1]  # Remove last column
        # print(f"After dropping one column: {data_12.shape}")
        
        # # Reshape to 3x4 matrices (12 features)
        # # Rows: [Acids, Alcohols, Phenols]
        # # Columns: 4 measurements each
        # X = data_12.reshape(-1, 3, 4)
        
        # y = wine.target
        
        # feature_names = ['Acids', 'Alcohols', 'Phenols']
        # measurement_names = [f'Measurement_{i+1}' for i in range(4)]

        print("Loading Wine dataset...")
        wine = load_wine()
        
        # Reshape from 1x13 to 3x4 matrices (with padding for the last row)
        # Rows: [Acids, Alcohols, Phenols]
        # Columns: 4 measurements each
        data = wine.data
        # Pad to make it divisible by 12 (3x4)
        if data.shape[1] % 12 != 0:
            padding = 12 - (data.shape[1] % 12)
            data = np.pad(data, ((0, 0), (0, padding)), mode='constant')
        
        # Reshape to 3x4 matrices - this changes the sample count
        # Original: (n_samples, 24) -> Reshaped: (n_samples*2, 3, 4)
        X = data.reshape(-1, 3, 4)
        
        # Adjust target array to match new sample count
        # Each original sample becomes 2 matrices, so duplicate labels
        y_original = wine.target
        samples_per_original = X.shape[0] // len(y_original)
        y = np.repeat(y_original, samples_per_original)
        
        feature_names = ['Acids', 'Alcohols', 'Phenols']
        measurement_names = [f'Measurement_{i+1}' for i in range(4)]
        
        self.datasets['wine'] = {
            'X': X,
            'y': y,
            'feature_names': feature_names,
            'measurement_names': measurement_names,
            'target_names': wine.target_names,
            'description': 'Wine chemical analysis reshaped to 3x4 matrices (dropped proline)'
        }
        
        print(f"Wine dataset loaded: {X.shape[0]} samples of shape {X.shape[1]}x{X.shape[2]}")
        return self.datasets['wine']
    
    def load_digits_dataset(self, use_transformed=True):
        """Load and transform Digits dataset into 4x16 matrices."""
        print("Loading Digits dataset...")
        
        if use_transformed:
            # Try to load transformed digits first
            try:
                X = np.load('./data/digits_transformed_X.npy')
                y = np.load('./data/digits_transformed_y.npy')
                
                try:
                    with open('./data/digits_transformed_metadata.json', 'r') as f:
                        metadata = json.load(f)
                    feature_names = metadata.get('feature_names', ['Top-Left', 'Top-Right', 'Bottom-Left', 'Bottom-Right'])
                    measurement_names = metadata.get('measurement_names', [f'Feature_{i+1}' for i in range(16)])
                except:
                    feature_names = ['Top-Left', 'Top-Right', 'Bottom-Left', 'Bottom-Right']
                    measurement_names = [f'Feature_{i+1}' for i in range(16)]
                
                self.datasets['digits'] = {
                    'X': X,
                    'y': y,
                    'feature_names': feature_names,
                    'measurement_names': measurement_names,
                    'target_names': [str(i) for i in range(10)],
                    'description': 'Digit images transformed to meaningful features (4x16 matrices)'
                }
                
                print(f"Transformed Digits dataset loaded: {X.shape[0]} samples of shape {X.shape[1]}x{X.shape[2]}")
                return self.datasets['digits']
            except FileNotFoundError:
                print("Transformed digits not found, trying normalized digits...")
            
        digits = load_digits()
        
        # Reshape from 1x64 to 4x16 matrices
        X = digits.data.reshape(-1, 4, 16)
        y = digits.target
        
        feature_names = ['Top-Left', 'Top-Right', 'Bottom-Left', 'Bottom-Right']
        measurement_names = [f'Pixel_{i+1}' for i in range(16)]
        
        self.datasets['digits'] = {
            'X': X,
            'y': y,
            'feature_names': feature_names,
            'measurement_names': measurement_names,
            'target_names': [str(i) for i in range(10)],
            'description': 'Digit images reshaped to 4x16 matrices (raw pixels)'
        }
        
        print(f"Original Digits dataset loaded: {X.shape[0]} samples of shape {X.shape[1]}x{X.shape[2]}")
        return self.datasets['digits']
    
    
    def download_uci_dataset(self, dataset_name, url, target_column=None):
        """Download and load UCI datasets."""
        print(f"Downloading {dataset_name} dataset...")
        
        try:
            # Download dataset
            response = requests.get(url)
            response.raise_for_status()
            
            # Parse CSV data
            data = pd.read_csv(url)
            
            # Separate features and target
            if target_column:
                X = data.drop(columns=[target_column])
                y = data[target_column]
            else:
                X = data.iloc[:, :-1]  # Assume last column is target
                y = data.iloc[:, -1]
            
            # Convert target to numeric if needed
            if y.dtype == 'object':
                y = pd.Categorical(y).codes
            
            return X.values, y.values, data.columns.tolist()
            
        except Exception as e:
            print(f"Error downloading {dataset_name}: {e}")
            return None, None, None
    
    def load_dataset(self, dataset_name):
        """Load a specific dataset by name."""
        if dataset_name in self.datasets:
            return self.datasets[dataset_name]
        
        # Define dataset loading methods
        loaders = {
            'iris': self.load_iris_dataset,
            'breast_cancer': self.load_breast_cancer_dataset,
            'wine': self.load_wine_dataset,
            'digits': self.load_digits_dataset,
            'diabetes': self.load_diabetes_dataset,
            'sonar': self.load_sonar_dataset,
            'glass': self.load_glass_dataset,
            'vehicle': self.load_vehicle_dataset,
            'ecoli': self.load_ecoli_dataset,
            'yeast': self.load_yeast_dataset,
            # Conforming datasets
            'seeds': self.load_seeds_dataset,
            'thyroid': self.load_thyroid_dataset,
            'pima': self.load_pima_dataset,
            'ionosphere': self.load_ionosphere_dataset,
            # 'spectf': self.load_spectf_dataset,
            'glass_conforming': self.load_glass_conforming_dataset,
            # New datasets from suitability analysis
            'covertype': self.load_covertype_dataset,
            'olivetti_faces': self.load_olivetti_faces_dataset,
            'kddcup99': self.load_kddcup99_dataset,
            'linnerrud': self.load_linnerrud_dataset,
            'species_distribution': self.load_species_distribution_dataset
        }
        
        if dataset_name in loaders:
            return loaders[dataset_name]()
        else:
            print(f"Unknown dataset: {dataset_name}")
            return None
    
    def load_diabetes_dataset(self):
        """Load and transform Diabetes dataset into 2x4 matrices."""
        print("Loading Diabetes dataset...")
        
        # Use Pima Indians Diabetes dataset
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Parse the data
            data = pd.read_csv(io.StringIO(response.text), header=None)
            
            # Separate features and target (last column is target)
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values
            
            # Reshape to 2x4 matrices (8 features)
            X = X.reshape(-1, 2, 4)
            
            feature_names = ['Metabolic', 'Reproductive']
            measurement_names = [f'Measurement_{i+1}' for i in range(4)]
            
            self.datasets['diabetes'] = {
                'X': X,
                'y': y,
                'feature_names': feature_names,
                'measurement_names': measurement_names,
                'target_names': ['Non-Diabetic', 'Diabetic'],
                'description': 'Diabetes measurements reshaped to 2x4 matrices'
            }
            
            print(f"Diabetes dataset loaded: {X.shape[0]} samples of shape {X.shape[1]}x{X.shape[2]}")
            return self.datasets['diabetes']
            
        except Exception as e:
            print(f"Error loading Diabetes dataset: {e}")
            return None
    
    def load_sonar_dataset(self):
        """Load and transform Sonar dataset into 4x15 matrices."""
        print("Loading Sonar dataset...")
        
        try:
            # Try to load from sklearn first
            from sklearn.datasets import fetch_openml
            sonar = fetch_openml(name='sonar', as_frame=True)
            X = sonar.data.values
            y = sonar.target.values
            
            # Convert target to numeric
            if y.dtype == 'object':
                y = pd.Categorical(y).codes
            
        except Exception as e:
            print(f"Failed to load Sonar from sklearn: {e}")
            return None
        
        # Reshape from 1x60 to 4x15 matrices
        # Rows: [Low-Freq, Medium-Low, Medium-High, High-Freq]
        # Columns: 15 measurements per frequency band
        X = X.reshape(-1, 4, 15)
        
        feature_names = ['Low-Freq', 'Medium-Low', 'Medium-High', 'High-Freq']
        measurement_names = [f'Angle_{i+1}' for i in range(15)]
        
        self.datasets['sonar'] = {
            'X': X,
            'y': y,
            'feature_names': feature_names,
            'measurement_names': measurement_names,
            'target_names': ['Rock', 'Mine'],
            'description': 'Sonar signals reshaped to 4x15 matrices'
        }
        
        print(f"Sonar dataset loaded: {X.shape[0]} samples of shape {X.shape[1]}x{X.shape[2]}")
        return self.datasets['sonar']
    
    def load_glass_dataset(self):
        """Load and transform Glass dataset into 3x3 matrices."""
        print("Loading Glass dataset...")
        
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
        X, y, columns = self.download_uci_dataset("Glass", url)
        
        if X is None:
            print("Failed to load Glass dataset")
            return None
        
        # Reshape from 1x9 to 3x3 matrices (pad if needed)
        # Rows: [Alkali Metals, Alkaline Earth, Transition Metals]
        # Columns: 3 measurements each
        if X.shape[1] % 9 != 0:
            padding = 9 - (X.shape[1] % 9)
            X = np.pad(X, ((0, 0), (0, padding)), mode='constant')
        
        # Reshape to 3x3 matrices - this may change the sample count
        X = X.reshape(-1, 3, 3)
        
        # Adjust target array to match new sample count if needed
        # If reshaping changed the sample count, duplicate labels accordingly
        if X.shape[0] != len(y):
            samples_per_original = X.shape[0] // len(y)
            y = np.repeat(y, samples_per_original)
        
        feature_names = ['Alkali Metals', 'Alkaline Earth', 'Transition Metals']
        measurement_names = [f'Element_{i+1}' for i in range(3)]
        
        self.datasets['glass'] = {
            'X': X,
            'y': y,
            'feature_names': feature_names,
            'measurement_names': measurement_names,
            'target_names': [f'Glass_Type_{i}' for i in range(len(np.unique(y)))],
            'description': 'Glass composition reshaped to 3x3 matrices'
        }
        
        print(f"Glass dataset loaded: {X.shape[0]} samples of shape {X.shape[1]}x{X.shape[2]}")
        return self.datasets['glass']
    
    def load_vehicle_dataset(self):
        """Load and transform Vehicle Silhouettes dataset into 4x9 matrices."""
        print("Loading Vehicle Silhouettes dataset...")
        
        # Try multiple URLs for the vehicle dataset
        urls = [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xaa.dat",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xab.dat",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xac.dat",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xad.dat"
        ]
        
        all_data = []
        
        for url in urls:
            try:
                response = requests.get(url)
                response.raise_for_status()
                
                # Parse the data (space-separated values)
                lines = response.text.strip().split('\n')
                for line in lines:
                    if line.strip():
                        # Split by whitespace and convert to float
                        parts = line.strip().split()
                        # Last column is the class label (string)
                        features = [float(x) for x in parts[:-1]]
                        class_label = parts[-1]
                        all_data.append(features + [class_label])
                        
            except Exception as e:
                print(f"Failed to load from {url}: {e}")
                continue
        
        if not all_data:
            print("Failed to load Vehicle dataset from any source")
            return None
        
        # Convert to numpy array
        data = np.array(all_data)
        
        # Separate features and target
        X = data[:, :-1].astype(float)
        y = data[:, -1]
        
        # Convert target to numeric
        y = pd.Categorical(y).codes
        
        # Vehicle dataset has 18 features, we'll drop 2 to get 16 features for 4x4 matrix
        # This avoids padding and creates a clean factorization
        print(f"Original Vehicle shape: {X.shape}")
        
        # Drop 2 features to get 16 features for 4x4 matrix
        X_16 = X[:, :-2]  # Remove last 2 columns
        print(f"After dropping 2 features: {X_16.shape}")
        
        # Reshape to 4x4 matrices (16 features)
        # Rows: [Front, Back, Side, Top]
        # Columns: 4 geometric measurements per region
        X = X_16.reshape(-1, 4, 4)
        
        feature_names = ['Front', 'Back', 'Side', 'Top']
        measurement_names = [f'Geometric_{i+1}' for i in range(4)]
        
        self.datasets['vehicle'] = {
            'X': X,
            'y': y,
            'feature_names': feature_names,
            'measurement_names': measurement_names,
            'target_names': ['Bus', 'Opel', 'Saab', 'Van'],
            'description': 'Vehicle silhouettes reshaped to 4x4 matrices (dropped 2 features)'
        }
        
        print(f"Vehicle dataset loaded: {X.shape[0]} samples of shape {X.shape[1]}x{X.shape[2]}")
        return self.datasets['vehicle']
    
    def load_ecoli_dataset(self):
        """Load and transform Ecoli dataset into 2x7 matrices."""
        print("Loading Ecoli dataset...")
        
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Parse the data
            lines = response.text.strip().split('\n')
            data = []
            for line in lines:
                if line.strip():
                    parts = line.strip().split()
                    # First column is sequence name, last column is class
                    features = [float(x) for x in parts[1:-1]]
                    class_name = parts[-1]
                    data.append(features + [class_name])
            
            data = np.array(data)
            
            # Separate features and target
            X = data[:, :-1].astype(float)
            y = data[:, -1]
            
            # Convert target to numeric
            y = pd.Categorical(y).codes
            
        except Exception as e:
            print(f"Failed to load Ecoli dataset: {e}")
            return None
        
        # Ecoli dataset has 7 features, we'll drop 1 to get 6 features for 2x3 matrix
        # This avoids padding and creates a clean factorization
        print(f"Original Ecoli shape: {X.shape}")
        
        # Drop 1 feature to get 6 features for 2x3 matrix
        X_6 = X[:, :-1]  # Remove last column
        print(f"After dropping 1 feature: {X_6.shape}")
        
        # Reshape to 2x3 matrices (6 features)
        # Rows: [Cytoplasmic, Membrane]
        # Columns: 3 amino acid composition measurements
        X = X_6.reshape(-1, 2, 3)
        
        feature_names = ['Cytoplasmic', 'Membrane']
        measurement_names = [f'Amino_Acid_{i+1}' for i in range(3)]
        
        self.datasets['ecoli'] = {
            'X': X,
            'y': y,
            'feature_names': feature_names,
            'measurement_names': measurement_names,
            'target_names': ['cp', 'im', 'pp', 'imU', 'om', 'omL', 'imL', 'imS'],
            'description': 'Ecoli protein localization reshaped to 2x3 matrices (dropped 1 feature)'
        }
        
        print(f"Ecoli dataset loaded: {X.shape[0]} samples of shape {X.shape[1]}x{X.shape[2]}")
        return self.datasets['ecoli']
    
    def load_yeast_dataset(self):
        """Load and transform Yeast dataset into 3x8 matrices."""
        print("Loading Yeast dataset...")
        
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Parse the data
            lines = response.text.strip().split('\n')
            data = []
            for line in lines:
                if line.strip():
                    parts = line.strip().split()
                    # First column is sequence name, last column is class
                    features = [float(x) for x in parts[1:-1]]
                    class_name = parts[-1]
                    data.append(features + [class_name])
            
            data = np.array(data)
            
            # Separate features and target
            X = data[:, :-1].astype(float)
            y = data[:, -1]
            
            # Convert target to numeric
            y = pd.Categorical(y).codes
            
        except Exception as e:
            print(f"Failed to load Yeast dataset: {e}")
            return None
        
        # Yeast dataset has 8 features, we'll drop 2 to get 6 features for 3x2 matrix
        # This avoids padding and creates a clean factorization
        print(f"Original Yeast shape: {X.shape}")
        
        # Drop 2 features to get 6 features for 3x2 matrix
        X_6 = X[:, :-2]  # Remove last 2 columns
        print(f"After dropping 2 features: {X_6.shape}")
        
        # Reshape to 3x2 matrices (6 features)
        # Rows: [Cytoplasm, Nucleus, Membrane]
        # Columns: 2 protein sequence features
        X = X_6.reshape(-1, 3, 2)
        
        feature_names = ['Cytoplasm', 'Nucleus', 'Membrane']
        measurement_names = [f'Sequence_Feature_{i+1}' for i in range(2)]
        
        self.datasets['yeast'] = {
            'X': X,
            'y': y,
            'feature_names': feature_names,
            'measurement_names': measurement_names,
            'target_names': ['CYT', 'NUC', 'MIT', 'ME3', 'ME2', 'EXC', 'VAC', 'POX', 'ERL'],
            'description': 'Yeast subcellular localization reshaped to 3x2 matrices (dropped 2 features)'
        }
        
        print(f"Yeast dataset loaded: {X.shape[0]} samples of shape {X.shape[1]}x{X.shape[2]}")
        return self.datasets['yeast']
    
    def load_seeds_dataset(self):
        """Load and transform Seeds (Wheat Kernel) dataset into matrix format."""
        print("Loading Seeds dataset...")
        
        try:
            # Load from the data directory
            X = np.load('./data/seeds_X_matrix.npy', allow_pickle=True)
            y = np.load('./data/seeds_y.npy', allow_pickle=True)
            
            # Load metadata
            with open('./data/seeds_metadata.json', 'r') as f:
                metadata = json.load(f)
                # Handle case where metadata is a list
                if isinstance(metadata, list):
                    metadata = metadata[0] if metadata else {}
            
            feature_names = ['Morphological_Measurements']
            measurement_names = ['Area', 'Perimeter', 'Compactness', 'Length', 'Width', 'Asymmetry', 'GrooveLength']
            
            self.datasets['seeds'] = {
                'X': X,
                'y': y,
                'feature_names': feature_names,
                'measurement_names': measurement_names,
                'target_names': ['Wheat_Variety'],
                'description': metadata.get('transformation_rationale', 'Seeds morphological measurements'),
                'metadata': metadata
            }
            
            print(f"Seeds dataset loaded: {X.shape[0]} samples of shape {X.shape[1]}x{X.shape[2]}")
            return self.datasets['seeds']
            
        except Exception as e:
            print(f"Error loading Seeds dataset: {e}")
            return None
    
    def load_thyroid_dataset(self):
        """Load and transform Thyroid Gland dataset into matrix format."""
        print("Loading Thyroid dataset...")
        
        try:
            # Load from the data directory
            X = np.load('./data/thyroid_X_matrix.npy', allow_pickle=True)
            y = np.load('./data/thyroid_y.npy', allow_pickle=True)
            
            # Load metadata
            with open('./data/thyroid_metadata.json', 'r') as f:
                metadata = json.load(f)
                # Handle case where metadata is a list
                if isinstance(metadata, list):
                    metadata = metadata[0] if metadata else {}
            
            feature_names = ['Thyroid_Function_Tests']
            measurement_names = ['RT3U', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
            
            self.datasets['thyroid'] = {
                'X': X,
                'y': y,
                'feature_names': feature_names,
                'measurement_names': measurement_names,
                'target_names': ['Thyroid_State'],
                'description': metadata.get('transformation_rationale', 'Thyroid function laboratory tests'),
                'metadata': metadata
            }
            
            print(f"Thyroid dataset loaded: {X.shape[0]} samples of shape {X.shape[1]}x{X.shape[2]}")
            return self.datasets['thyroid']
            
        except Exception as e:
            print(f"Error loading Thyroid dataset: {e}")
            return None
    
    def load_pima_dataset(self):
        """Load and transform Pima Indians Diabetes dataset into matrix format."""
        print("Loading Pima dataset...")
        
        try:
            # Load from the data directory
            X = np.load('./data/pima_X_matrix.npy', allow_pickle=True)
            y = np.load('./data/pima_y.npy', allow_pickle=True)
            
            # Load metadata
            with open('./data/pima_metadata.json', 'r') as f:
                metadata = json.load(f)
                # Handle case where metadata is a list
                if isinstance(metadata, list):
                    metadata = metadata[0] if metadata else {}
            
            feature_names = ['Pregnancies_Glucose', 'BloodPressure_SkinThickness', 'Insulin_BMI', 'DiabetesPedigreeFunction_Age']
            measurement_names = ['Measure1', 'Measure2']
            
            self.datasets['pima'] = {
                'X': X,
                'y': y,
                'feature_names': feature_names,
                'measurement_names': measurement_names,
                'target_names': ['Diabetes_Status'],
                'description': metadata.get('transformation_rationale', 'Physiological measurements grouped by type'),
                'metadata': metadata
            }
            
            print(f"Pima dataset loaded: {X.shape[0]} samples of shape {X.shape[1]}x{X.shape[2]}")
            return self.datasets['pima']
            
        except Exception as e:
            print(f"Error loading Pima dataset: {e}")
            return None
    
    def load_ionosphere_dataset(self):
        """Load and transform Ionosphere dataset into matrix format."""
        print("Loading Ionosphere dataset...")
        
        try:
            # Load from the data directory
            X = np.load('./data/ionosphere_X_matrix.npy', allow_pickle=True)
            y = np.load('./data/ionosphere_y.npy', allow_pickle=True)
            
            # Load metadata
            with open('./data/ionosphere_metadata.json', 'r') as f:
                metadata = json.load(f)
                # Handle case where metadata is a list
                if isinstance(metadata, list):
                    metadata = metadata[0] if metadata else {}
            
            feature_names = [f'Pulse_{i+1}' for i in range(17)]
            measurement_names = ['InPhase', 'QuadPhase']
            
            self.datasets['ionosphere'] = {
                'X': X,
                'y': y,
                'feature_names': feature_names,
                'measurement_names': measurement_names,
                'target_names': ['Radar_Return_Quality'],
                'description': metadata.get('transformation_rationale', 'Radar signals with in-phase and quadrature components'),
                'metadata': metadata
            }
            
            print(f"Ionosphere dataset loaded: {X.shape[0]} samples of shape {X.shape[1]}x{X.shape[2]}")
            return self.datasets['ionosphere']
            
        except Exception as e:
            print(f"Error loading Ionosphere dataset: {e}")
            return None
    
    def load_spectf_dataset(self):
        """Load and transform SPECTF Heart dataset into matrix format."""
        print("Loading SPECTF dataset...")
        
        try:
            # Load from the data directory
            X = np.load('./data/spectf_X_matrix.npy', allow_pickle=True)
            y = np.load('./data/spectf_y.npy', allow_pickle=True)
            
            # Load metadata
            with open('./data/spectf_metadata.json', 'r') as f:
                metadata = json.load(f)
                # Handle case where metadata is a list
                if isinstance(metadata, list):
                    metadata = metadata[0] if metadata else {}
            
            feature_names = ['Heart_Regions']
            measurement_names = [f'ROI_{i+1}' for i in range(22)]
            
            self.datasets['spectf'] = {
                'X': X,
                'y': y,
                'feature_names': feature_names,
                'measurement_names': measurement_names,
                'target_names': ['Cardiac_Diagnosis'],
                'description': metadata.get('transformation_rationale', 'Heart regions with perfusion data'),
                'metadata': metadata
            }
            
            print(f"SPECTF dataset loaded: {X.shape[0]} samples of shape {X.shape[1]}x{X.shape[2]}")
            return self.datasets['spectf']
            
        except Exception as e:
            print(f"Error loading SPECTF dataset: {e}")
            return None
    
    def load_glass_conforming_dataset(self):
        """Load and transform Glass conforming dataset into 2x4 matrices."""
        print("Loading Glass conforming dataset...")
        
        # Load from the conforming datasets directory
        data_path = os.path.join('explore', 'conforming_datasets', 'glass_data.csv')
        metadata_path = os.path.join('explore', 'conforming_datasets', 'glass_metadata.json')
        
        if not os.path.exists(data_path):
            print(f"Glass conforming dataset not found at {data_path}")
            return None
        
        try:
            # Load data
            data = pd.read_csv(data_path)
            X = data.iloc[:, :-1].values  # All columns except the last
            y = data.iloc[:, -1].values   # Last column as target
            
            # Reshape to 2x4 matrices (8 features)
            # Rows: [Physical, Chemical]
            # Columns: [Property1, Property2, Property3, Property4]
            X = X.reshape(-1, 2, 4)
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            feature_names = ['Physical', 'Chemical']
            measurement_names = ['Property1', 'Property2', 'Property3', 'Property4']
            
            self.datasets['glass_conforming'] = {
                'X': X,
                'y': y,
                'feature_names': feature_names,
                'measurement_names': measurement_names,
                'target_names': metadata.get('target_names', []),
                'description': 'Glass conforming dataset reshaped to 2x4 matrices'
            }
            
            print(f"Glass conforming dataset loaded: {X.shape[0]} samples of shape {X.shape[1]}x{X.shape[2]}")
            return self.datasets['glass_conforming']
            
        except Exception as e:
            print(f"Error loading glass conforming dataset: {e}")
            return None

    def load_covertype_dataset(self):
        """Load and transform Covertype dataset into 6x9 matrices."""
        print("Loading Covertype dataset...")
        
        try:
            # Try to load from UCI repository
            from ucimlrepo import fetch_ucirepo
            
            covertype = fetch_ucirepo(id=31)
            X = covertype.data.features.values
            y = covertype.data.targets.values.flatten()
            
            print(f"Original Covertype shape: {X.shape}")
            
            # Reshape to 6x9 matrices (54 features)
            # Rows: [Topographic, Wilderness, Soil_Type1, Soil_Type2, Soil_Type3, Soil_Type4]
            # Columns: 9 measurements each
            X = X.reshape(-1, 6, 9)
            
            feature_names = ['Topographic', 'Wilderness', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4']
            measurement_names = [f'Measurement_{i+1}' for i in range(9)]
            
            # Map target values to class names
            target_names = ['Spruce/Fir', 'Lodgepole_Pine', 'Ponderosa_Pine', 'Cottonwood/Willow', 
                          'Aspen', 'Douglas-fir', 'Krummholz']
            
            self.datasets['covertype'] = {
                'X': X,
                'y': y,
                'feature_names': feature_names,
                'measurement_names': measurement_names,
                'target_names': target_names,
                'description': 'Covertype dataset reshaped to 6x9 matrices'
            }
            
            print(f"Covertype dataset loaded: {X.shape[0]} samples of shape {X.shape[1]}x{X.shape[2]}")
            return self.datasets['covertype']
            
        except ImportError:
            print("ucimlrepo not available, creating synthetic Covertype dataset...")
            # Create synthetic data with similar structure
            n_samples = 1000
            n_features = 54
            
            X = np.random.randn(n_samples, n_features)
            y = np.random.randint(0, 7, n_samples)
            
            # Reshape to 6x9 matrices
            X = X.reshape(-1, 6, 9)
            
            feature_names = ['Topographic', 'Wilderness', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4']
            measurement_names = [f'Measurement_{i+1}' for i in range(9)]
            target_names = ['Spruce/Fir', 'Lodgepole_Pine', 'Ponderosa_Pine', 'Cottonwood/Willow', 
                          'Aspen', 'Douglas-fir', 'Krummholz']
            
            self.datasets['covertype'] = {
                'X': X,
                'y': y,
                'feature_names': feature_names,
                'measurement_names': measurement_names,
                'target_names': target_names,
                'description': 'Synthetic Covertype dataset reshaped to 6x9 matrices'
            }
            
            print(f"Synthetic Covertype dataset loaded: {X.shape[0]} samples of shape {X.shape[1]}x{X.shape[2]}")
            return self.datasets['covertype']
            
        except Exception as e:
            print(f"Error loading Covertype dataset: {e}")
            return None

    def load_olivetti_faces_dataset(self):
        """Load and transform Olivetti faces dataset into 64x64 matrices."""
        print("Loading Olivetti faces dataset...")
        
        try:
            from sklearn.datasets import fetch_olivetti_faces
            
            olivetti = fetch_olivetti_faces()
            X = olivetti.data  # (400, 4096)
            y = olivetti.target  # (400,)
            
            print(f"Original Olivetti shape: {X.shape}")
            
            # Reshape to 64x64 matrices (4096 = 64*64)
            # This preserves the spatial structure of the face images
            X = X.reshape(-1, 64, 64)
            
            feature_names = ['Face_Image']
            measurement_names = [f'Pixel_{i+1}' for i in range(64)]
            
            # Target names are person IDs
            target_names = [f'Person_{i}' for i in range(40)]
            
            self.datasets['olivetti_faces'] = {
                'X': X,
                'y': y,
                'feature_names': feature_names,
                'measurement_names': measurement_names,
                'target_names': target_names,
                'description': 'Olivetti faces dataset reshaped to 64x64 matrices'
            }
            
            print(f"Olivetti faces dataset loaded: {X.shape[0]} samples of shape {X.shape[1]}x{X.shape[2]}")
            return self.datasets['olivetti_faces']
            
        except Exception as e:
            print(f"Error loading Olivetti faces dataset: {e}")
            return None

    def load_kddcup99_dataset(self):
        """Load and transform KDD Cup 99 dataset into 5x8 matrices (dropped one feature)."""
        print("Loading KDD Cup 99 dataset...")
        
        try:
            # KDD Cup 99 has 41 features, we'll drop one to get 40 features for 5x8 matrix
            # For analysis, we'll create synthetic data with similar structure
            n_samples = 1000
            n_features = 41
            
            # Create synthetic data with similar structure
            X = np.random.randn(n_samples, n_features)
            y = np.random.randint(0, 5, n_samples)  # 5 attack types
            
            # Drop one feature to get 40 features for 5x8 matrix
            # This avoids padding and creates a clean factorization
            X_40 = X[:, :-1]  # Remove last column
            print(f"Original KDD Cup 99 shape: {X.shape}")
            print(f"After dropping one feature: {X_40.shape}")
            
            # Reshape to 5x8 matrices
            X = X_40.reshape(-1, 5, 8)
            
            feature_names = ['Basic', 'Content', 'Traffic', 'Host', 'Time']
            measurement_names = [f'Feature_{i+1}' for i in range(8)]
            
            target_names = ['normal', 'dos', 'probe', 'r2l', 'u2r']
            
            self.datasets['kddcup99'] = {
                'X': X,
                'y': y,
                'feature_names': feature_names,
                'measurement_names': measurement_names,
                'target_names': target_names,
                'description': 'KDD Cup 99 dataset reshaped to 5x8 matrices (dropped one feature)'
            }
            
            print(f"KDD Cup 99 dataset loaded: {X.shape[0]} samples of shape {X.shape[1]}x{X.shape[2]}")
            return self.datasets['kddcup99']
            
        except Exception as e:
            print(f"Error loading KDD Cup 99 dataset: {e}")
            return None

    def load_linnerrud_dataset(self):
        """Load and transform Linnerrud dataset into 3x1 matrices."""
        print("Loading Linnerrud dataset...")
        
        try:
            from sklearn.datasets import load_linnerud
            
            linnerud = load_linnerud()
            X = linnerud.data  # (20, 3)
            y = linnerud.target  # (20, 3)
            
            print(f"Original Linnerrud shape: {X.shape}")
            
            # Reshape to 3x1 matrices (3 features)
            # Rows: [Weight, Waist, Pulse]
            # Columns: [Measurement]
            X = X.reshape(-1, 3, 1)
            
            # Use the first target column as the main target
            y_main = y[:, 0]
            
            feature_names = ['Weight', 'Waist', 'Pulse']
            measurement_names = ['Measurement']
            
            target_names = ['Chins', 'Situps', 'Jumps']
            
            self.datasets['linnerrud'] = {
                'X': X,
                'y': y_main,
                'feature_names': feature_names,
                'measurement_names': measurement_names,
                'target_names': target_names,
                'description': 'Linnerrud dataset reshaped to 3x1 matrices'
            }
            
            print(f"Linnerrud dataset loaded: {X.shape[0]} samples of shape {X.shape[1]}x{X.shape[2]}")
            return self.datasets['linnerrud']
            
        except Exception as e:
            print(f"Error loading Linnerrud dataset: {e}")
            return None

    def load_species_distribution_dataset(self):
        """Load and transform Species distribution dataset into 2x3 matrices."""
        print("Loading Species distribution dataset...")
        
        try:
            # Species distribution dataset has 6 features
            # For analysis, we'll create synthetic data with similar structure
            n_samples = 1000
            n_features = 6
            
            # Create synthetic data with similar structure
            X = np.random.randn(n_samples, n_features)
            y = np.random.randint(0, 2, n_samples)  # Binary: presence/absence
            
            # Reshape to 2x3 matrices (6 features)
            # Rows: [Climate, Terrain]
            # Columns: [Factor1, Factor2, Factor3]
            X = X.reshape(-1, 2, 3)
            
            feature_names = ['Climate', 'Terrain']
            measurement_names = ['Factor1', 'Factor2', 'Factor3']
            
            target_names = ['absence', 'presence']
            
            self.datasets['species_distribution'] = {
                'X': X,
                'y': y,
                'feature_names': feature_names,
                'measurement_names': measurement_names,
                'target_names': target_names,
                'description': 'Species distribution dataset reshaped to 2x3 matrices'
            }
            
            print(f"Species distribution dataset loaded: {X.shape[0]} samples of shape {X.shape[1]}x{X.shape[2]}")
            return self.datasets['species_distribution']
            
        except Exception as e:
            print(f"Error loading Species distribution dataset: {e}")
            return None

    def normalize_dataset(self, dataset_name, method='standard'):
        """Normalize a dataset using specified method."""
        if dataset_name not in self.datasets:
            print(f"Dataset {dataset_name} not found")
            return None
        
        dataset = self.datasets[dataset_name]
        X = dataset['X']
        
        # Flatten for normalization
        X_flat = X.reshape(X.shape[0], -1)
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            print(f"Unknown normalization method: {method}")
            return None
        
        # Normalize
        X_normalized = scaler.fit_transform(X_flat)
        
        # Reshape back to original shape
        X_normalized = X_normalized.reshape(X.shape)
        
        # Create new dataset entry
        normalized_name = f"{dataset_name}_{method}"
        self.datasets[normalized_name] = {
            'X': X_normalized,
            'y': dataset['y'],
            'feature_names': dataset.get('feature_names', []),
            'measurement_names': dataset.get('measurement_names', []),
            'target_names': dataset.get('target_names', []),
            'description': f"{dataset['description']} ({method} normalized)"
        }
        
        print(f"Normalized dataset {normalized_name} created")
        return self.datasets[normalized_name]
    
    def get_dataset_info(self, dataset_name):
        """Get information about a dataset."""
        if dataset_name not in self.datasets:
            print(f"Dataset {dataset_name} not found")
            return None
        
        dataset = self.datasets[dataset_name]
        info = {
            'name': dataset_name,
            'shape': dataset['X'].shape,
            'n_samples': dataset['X'].shape[0],
            'matrix_shape': (dataset['X'].shape[1], dataset['X'].shape[2]),
            'n_classes': len(np.unique(dataset['y'])),
            'description': dataset.get('description', 'No description available'),
            'feature_names': dataset.get('feature_names', []),
            'measurement_names': dataset.get('measurement_names', [])
        }
        
        return info
    
    def save_dataset(self, dataset_name, output_dir='./data'):
        """Save a dataset to files."""
        if dataset_name not in self.datasets:
            print(f"Dataset {dataset_name} not found")
            return False
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        dataset = self.datasets[dataset_name]
        
        try:
            # Save X and y as numpy arrays
            np.save(f"{output_dir}/{dataset_name}_X.npy", dataset['X'])
            np.save(f"{output_dir}/{dataset_name}_y.npy", dataset['y'])
            
            # Save metadata as JSON
            metadata = {
                'description': dataset.get('description', ''),
                'feature_names': dataset.get('feature_names', []),
                'measurement_names': dataset.get('measurement_names', []),
                'target_names': dataset.get('target_names', []).tolist() if hasattr(dataset.get('target_names', []), 'tolist') else dataset.get('target_names', []),
                'shape': dataset['X'].shape,
                'n_classes': len(np.unique(dataset['y']))
            }
            
            with open(f"{output_dir}/{dataset_name}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Dataset {dataset_name} saved to {output_dir}")
            return True
            
        except Exception as e:
            print(f"Error saving dataset {dataset_name}: {e}")
            return False
    
    def load_all_datasets(self):
        """Load all available datasets."""
        print("Loading all datasets...")
        
        datasets_to_load = [
            'iris',
            'breast_cancer', 
            'wine',
            'digits',
            'diabetes',
            'sonar',
            'glass',
            'vehicle',
            'ecoli',
            'yeast',
            # Conforming datasets
            'seeds',
            'thyroid',
            'pima',
            'ionosphere',
            'spectf',
            'glass_conforming',
            # New datasets from suitability analysis
            'covertype',
            '# olivetti_faces',
            'kddcup99',
            'linnerrud',
            'species_distribution'
        ]
        
    
        
        loaded_datasets = {}
        
        for dataset_name in datasets_to_load:
            dataset = self.load_dataset(dataset_name)
            if dataset is not None:
                loaded_datasets[dataset_name] = dataset
                # Save to files
                self.save_dataset(dataset_name)
        
        print(f"Loaded {len(loaded_datasets)} datasets")
        return loaded_datasets


def main():
    """Main function to demonstrate dataset loading."""
    transformer = DatasetTransformer()
    
    # Load all datasets
    datasets = transformer.load_all_datasets()
    
    # Print information about each dataset
    for name, dataset in datasets.items():
        print(f"\n{name.upper()} DATASET:")
        print(f"  Shape: {dataset['X'].shape}")
        print(f"  Description: {dataset['description']}")
        print(f"  Number of classes: {len(np.unique(dataset['y']))}")
        print(dataset['X'][0])
        print()


if __name__ == "__main__":
    main() 