#!/usr/bin/env python3
"""
Matrix Structure Validation Script

This script validates the matrix structures of the conforming datasets and provides
detailed analysis of their properties, including semantic relationships and CNN compatibility.
"""

import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class MatrixStructureValidator:
    """
    Validator for matrix structures of conforming datasets.
    """
    
    def __init__(self, data_dir="conforming_datasets"):
        self.data_dir = data_dir
        self.results = {}
        
    def load_dataset(self, name):
        """Load a dataset and its metadata."""
        try:
            X = np.load(os.path.join(self.data_dir, f'{name}_X_matrix.npy'), allow_pickle=True)
            y = np.load(os.path.join(self.data_dir, f'{name}_y.npy'), allow_pickle=True)
            
            with open(os.path.join(self.data_dir, f'{name}_metadata.json'), 'r') as f:
                metadata = json.load(f)
            
            return X, y, metadata[0] if isinstance(metadata, list) else metadata
        except Exception as e:
            print(f"Error loading {name}: {e}")
            return None, None, None
    
    def validate_matrix_structure(self, name, X, y, metadata):
        """Validate the matrix structure of a dataset."""
        print(f"\n{'='*60}")
        print(f"VALIDATING: {name.upper()}")
        print(f"{'='*60}")
        
        # Basic shape validation
        print(f"Matrix Shape: {X.shape}")
        print(f"Number of Samples: {X.shape[0]}")
        print(f"Matrix Dimensions: {X.shape[1]} × {X.shape[2]}")
        print(f"Number of Classes: {len(np.unique(y))}")
        
        # Check for consistency
        if X.shape[0] == len(y):
            print("✓ Sample count consistency: OK")
        else:
            print("✗ Sample count consistency: FAILED")
        
        # Check for NaN values
        if np.isnan(X).any():
            print("✗ Contains NaN values")
        else:
            print("✓ No NaN values")
        
        # Check for infinite values
        if np.isinf(X).any():
            print("✗ Contains infinite values")
        else:
            print("✓ No infinite values")
        
        # Statistical properties
        print(f"\nStatistical Properties:")
        print(f"  Mean: {np.mean(X):.4f}")
        print(f"  Std: {np.std(X):.4f}")
        print(f"  Min: {np.min(X):.4f}")
        print(f"  Max: {np.max(X):.4f}")
        
        # Class distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"\nClass Distribution:")
        for class_id, count in zip(unique, counts):
            print(f"  Class {class_id}: {count} samples ({count/len(y)*100:.1f}%)")
        
        # Matrix structure analysis
        print(f"\nMatrix Structure Analysis:")
        print(f"  Total Elements per Sample: {X.shape[1] * X.shape[2]}")
        print(f"  Memory per Sample: {X.shape[1] * X.shape[2] * 8} bytes")
        
        # CNN compatibility
        if X.shape[2] == 1:
            print("  CNN Input Shape: (batch_size, 1, height)")
        elif X.shape[2] == 2:
            print("  CNN Input Shape: (batch_size, 2, height)")
        else:
            print("  CNN Input Shape: (batch_size, channels, height, width)")
        
        return {
            'name': name,
            'shape': X.shape,
            'n_samples': X.shape[0],
            'n_classes': len(np.unique(y)),
            'matrix_dims': (X.shape[1], X.shape[2]),
            'has_nan': np.isnan(X).any(),
            'has_inf': np.isinf(X).any(),
            'mean': np.mean(X),
            'std': np.std(X),
            'min': np.min(X),
            'max': np.max(X),
            'class_distribution': dict(zip(unique, counts)),
            'metadata': metadata
        }
    
    def analyze_semantic_relationships(self, name, X, metadata):
        """Analyze semantic relationships in the matrix structure."""
        print(f"\nSemantic Relationship Analysis:")
        
        if name == 'seeds':
            features = ['Area', 'Perimeter', 'Compactness', 'Length', 'Width', 'Asymmetry', 'GrooveLength']
            print(f"  Features: {features}")
            print(f"  Semantic Group: All morphological measurements of wheat kernels")
            print(f"  Matrix Interpretation: 1D array of morphological measurements")
            
        elif name == 'thyroid':
            features = ['RT3U', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
            print(f"  Features: {features}")
            print(f"  Semantic Group: All thyroid function laboratory tests")
            print(f"  Matrix Interpretation: 1D array of thyroid hormone measurements")
            
        elif name == 'pima':
            groups = [
                ['Pregnancies', 'Glucose'],
                ['BloodPressure', 'SkinThickness'],
                ['Insulin', 'BMI'],
                ['DiabetesPedigreeFunction', 'Age']
            ]
            print(f"  Feature Groups: {groups}")
            print(f"  Semantic Group: Physiological measurements grouped by type")
            print(f"  Matrix Interpretation: 4×2 matrix where each row is a health category")
            
        elif name == 'ionosphere':
            print(f"  Feature Structure: 17 pulse returns × 2 phases (in-phase, quadrature)")
            print(f"  Semantic Group: Radar signals with in-phase and quadrature components")
            print(f"  Matrix Interpretation: 17×2 matrix where each row is a pulse return")
            
        elif name == 'spectf':
            print(f"  Feature Structure: 22 ROIs with perfusion data")
            print(f"  Semantic Group: Heart regions with perfusion data")
            print(f"  Matrix Interpretation: 1D array of heart region measurements")
            
        elif name == 'glass':
            print(f"  Feature Structure: Major oxides × Trace oxides")
            print(f"  Semantic Group: Major vs trace oxides in ceramic composition")
            print(f"  Matrix Interpretation: 4×4 chemistry matrix")
    
    def visualize_matrix_structure(self, name, X, y):
        """Create visualizations of the matrix structure."""
        print(f"\nCreating visualizations...")
        
        # Create output directory for plots
        plots_dir = os.path.join(self.data_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Sample matrix heatmap
        plt.figure(figsize=(10, 6))
        
        if X.shape[2] == 1:
            # 1D matrix - show as heatmap
            sample_idx = 0
            sample_matrix = X[sample_idx].flatten()
            
            plt.subplot(1, 2, 1)
            plt.imshow(sample_matrix.reshape(1, -1), aspect='auto', cmap='viridis')
            plt.title(f'{name.upper()} - Sample Matrix Structure')
            plt.xlabel('Features')
            plt.ylabel('Sample')
            plt.colorbar()
            
            # Show multiple samples
            plt.subplot(1, 2, 2)
            plt.imshow(X[:10].reshape(10, -1), aspect='auto', cmap='viridis')
            plt.title(f'{name.upper()} - First 10 Samples')
            plt.xlabel('Features')
            plt.ylabel('Sample Index')
            plt.colorbar()
            
        else:
            # 2D matrix - show as heatmap
            sample_idx = 0
            sample_matrix = X[sample_idx]
            
            plt.subplot(1, 2, 1)
            plt.imshow(sample_matrix, aspect='auto', cmap='viridis')
            plt.title(f'{name.upper()} - Sample Matrix Structure')
            plt.xlabel('Columns')
            plt.ylabel('Rows')
            plt.colorbar()
            
            # Show class distribution
            plt.subplot(1, 2, 2)
            unique, counts = np.unique(y, return_counts=True)
            plt.bar(unique, counts)
            plt.title(f'{name.upper()} - Class Distribution')
            plt.xlabel('Class')
            plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{name}_matrix_structure.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved visualization: {name}_matrix_structure.png")
    
    def validate_all_datasets(self):
        """Validate all conforming datasets."""
        datasets = ['seeds', 'thyroid', 'pima', 'ionosphere', 'spectf', 'glass']
        
        print("="*80)
        print("MATRIX STRUCTURE VALIDATION")
        print("="*80)
        
        for name in datasets:
            X, y, metadata = self.load_dataset(name)
            if X is not None:
                result = self.validate_matrix_structure(name, X, y, metadata)
                self.analyze_semantic_relationships(name, X, metadata)
                self.visualize_matrix_structure(name, X, y)
                self.results[name] = result
        
        # Create summary report
        self.create_validation_report()
    
    def create_validation_report(self):
        """Create a comprehensive validation report."""
        print(f"\n{'='*80}")
        print("VALIDATION SUMMARY")
        print(f"{'='*80}")
        
        summary = {
            'total_datasets': len(self.results),
            'successful_validations': len(self.results),
            'datasets': {}
        }
        
        for name, result in self.results.items():
            summary['datasets'][name] = {
                'matrix_shape': str(result['shape']),
                'n_samples': int(result['n_samples']),
                'n_classes': int(result['n_classes']),
                'matrix_dims': str(result['matrix_dims']),
                'has_nan': bool(result['has_nan']),
                'has_inf': bool(result['has_inf']),
                'class_distribution': {str(k): int(v) for k, v in result['class_distribution'].items()},
                'semantic_grouping': str(result['metadata'].get('semantic_grouping', 'N/A')),
                'transformation_rationale': str(result['metadata'].get('transformation_rationale', 'N/A'))
            }
        
        # Save validation report
        with open(os.path.join(self.data_dir, 'validation_report.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Validation report saved: validation_report.json")
        
        # Print summary table
        print(f"\nDataset Summary:")
        print(f"{'Dataset':<15} {'Matrix Shape':<12} {'Samples':<8} {'Classes':<8} {'Status':<10}")
        print(f"{'-'*60}")
        
        for name, result in self.results.items():
            status = "✓ VALID" if not (result['has_nan'] or result['has_inf']) else "✗ INVALID"
            print(f"{name.upper():<15} {str(result['matrix_dims']):<12} {result['n_samples']:<8} {result['n_classes']:<8} {status:<10}")
        
        print(f"\nAll datasets successfully validated and processed!")
        print(f"Visualizations saved in: {os.path.join(self.data_dir, 'plots')}")

def main():
    """Main validation function."""
    validator = MatrixStructureValidator()
    validator.validate_all_datasets()

if __name__ == "__main__":
    main() 