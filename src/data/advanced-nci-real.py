#!/usr/bin/env python3
"""
NCI Chemical Data Processor for Pseudo-Boolean Polynomial Analysis

This script processes NCI chemical data (SMILES, molecular weight, CAS numbers)
and converts them into matrix format suitable for PBP dimensionality reduction.
"""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("RDKit not available. Install with: pip install rdkit-pypi")

class NCIChemicalDataProcessor:
    """Process NCI chemical data into PBP-compatible matrices."""
    
    def __init__(self, data_dir="data/real_medical/downloads/nci_chemical"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path("data/real_medical")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.smiles_file = self.data_dir / "nsc_smiles.csv"
        self.mw_file = self.data_dir / "nsc_mw_mf.csv"
        self.cas_file = self.data_dir / "nsc_cas.csv"
        
        # Check if files exist
        self._check_files()
        
    def _check_files(self):
        """Check if required files exist."""
        files = [self.smiles_file, self.mw_file, self.cas_file]
        missing_files = [f for f in files if not f.exists()]
        
        if missing_files:
            print(f"Missing files: {missing_files}")
            print("Please download NCI chemical data files to the specified directory.")
            return False
        
        print("✓ All NCI chemical data files found")
        return True
    
    def load_data(self):
        """Load all NCI chemical data files."""
        print("Loading NCI chemical data...")
        
        # Load SMILES data
        self.smiles_df = pd.read_csv(self.smiles_file)
        print(f"SMILES data: {len(self.smiles_df)} compounds")
        
        # Load molecular weight data
        self.mw_df = pd.read_csv(self.mw_file)
        print(f"Molecular weight data: {len(self.mw_df)} compounds")
        
        # Load CAS data
        self.cas_df = pd.read_csv(self.cas_file)
        print(f"CAS data: {len(self.cas_df)} compounds")
        
        # Merge data on NSC
        self.merged_df = self.smiles_df.merge(self.mw_df, on='NSC', how='inner')
        self.merged_df = self.merged_df.merge(self.cas_df, on='NSC', how='left')
        
        print(f"Merged data: {len(self.merged_df)} compounds")
        return self.merged_df
    
    def create_molecular_fingerprints(self, smiles_list, radius=2, nBits=2048):
        """Convert SMILES to Morgan fingerprints."""
        if not RDKIT_AVAILABLE:
            print("RDKit not available. Using simplified molecular descriptors.")
            return self._create_simple_descriptors(smiles_list)
        
        fingerprints = []
        valid_indices = []
        
        for i, smiles in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
                    fingerprints.append(np.array(fp))
                    valid_indices.append(i)
            except:
                continue
        
        if fingerprints:
            return np.array(fingerprints), valid_indices
        else:
            print("No valid fingerprints generated. Using simplified descriptors.")
            return self._create_simple_descriptors(smiles_list)
    
    def _create_simple_descriptors(self, smiles_list):
        """Create simple molecular descriptors when RDKit is not available."""
        print("Creating simplified molecular descriptors...")
        
        descriptors = []
        valid_indices = []
        
        for i, smiles in enumerate(smiles_list):
            try:
                # Simple character-based features
                features = []
                
                # Length of SMILES
                features.append(len(smiles))
                
                # Count of different atom types
                features.append(smiles.count('C'))
                features.append(smiles.count('N'))
                features.append(smiles.count('O'))
                features.append(smiles.count('S'))
                features.append(smiles.count('P'))
                features.append(smiles.count('F'))
                features.append(smiles.count('Cl'))
                features.append(smiles.count('Br'))
                features.append(smiles.count('I'))
                
                # Count of bonds
                features.append(smiles.count('='))
                features.append(smiles.count('#'))
                features.append(smiles.count('('))
                features.append(smiles.count(')'))
                
                # Count of special groups
                features.append(smiles.count('OH'))
                features.append(smiles.count('NH'))
                features.append(smiles.count('COOH'))
                features.append(smiles.count('COO'))
                
                # Ring count (approximate)
                features.append(smiles.count('c'))
                features.append(smiles.count('n'))
                
                descriptors.append(features)
                valid_indices.append(i)
                
            except:
                continue
        
        if descriptors:
            return np.array(descriptors), valid_indices
        else:
            print("Failed to create descriptors")
            return None, []
    
    def create_molecular_descriptors(self, smiles_list):
        """Create comprehensive molecular descriptors."""
        if not RDKIT_AVAILABLE:
            return self._create_simple_descriptors(smiles_list)
        
        descriptors = []
        valid_indices = []
        
        for i, smiles in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Calculate various molecular descriptors
                    desc = []
                    
                    # Basic descriptors
                    desc.append(Descriptors.MolWt(mol))
                    desc.append(Descriptors.MolLogP(mol))
                    desc.append(Descriptors.MolMR(mol))
                    desc.append(Descriptors.NumHDonors(mol))
                    desc.append(Descriptors.NumHAcceptors(mol))
                    desc.append(Descriptors.NumRotatableBonds(mol))
                    desc.append(Descriptors.NumAromaticRings(mol))
                    desc.append(Descriptors.NumAliphaticRings(mol))
                    desc.append(Descriptors.NumSaturatedRings(mol))
                    desc.append(Descriptors.NumHeteroatoms(mol))
                    
                    # Additional descriptors
                    desc.append(Descriptors.FractionCsp3(mol))
                    desc.append(Descriptors.HeavyAtomCount(mol))
                    desc.append(Descriptors.RingCount(mol))
                    desc.append(Descriptors.AromaticRings(mol))
                    desc.append(Descriptors.SaturatedRings(mol))
                    
                    descriptors.append(desc)
                    valid_indices.append(i)
                    
            except:
                continue
        
        if descriptors:
            return np.array(descriptors), valid_indices
        else:
            print("Failed to create molecular descriptors")
            return None, []
    
    def create_matrix_structure(self, data_matrix, n_rows=4, n_cols=8):
        """Reshape data into matrix structure for PBP."""
        # Flatten the data
        flat_data = data_matrix.reshape(data_matrix.shape[0], -1)
        
        # Pad if necessary to make it divisible by n_rows * n_cols
        total_features = flat_data.shape[1]
        target_features = n_rows * n_cols
        
        if total_features < target_features:
            # Pad with zeros
            padding = target_features - total_features
            flat_data = np.pad(flat_data, ((0, 0), (0, padding)), mode='constant')
        elif total_features > target_features:
            # Truncate to target size
            flat_data = flat_data[:, :target_features]
        
        # Reshape to matrix structure
        matrix_data = flat_data.reshape(-1, n_rows, n_cols)
        
        return matrix_data
    
    def create_target_variables(self, valid_indices):
        """Create target variables for classification."""
        # Use molecular weight as a proxy for target classification
        mw_values = self.merged_df.iloc[valid_indices]['MW'].values
        
        # Handle NaN values
        valid_mw = ~np.isnan(mw_values)
        mw_values_clean = mw_values[valid_mw]
        
        if len(mw_values_clean) == 0:
            print("Warning: No valid molecular weight values found")
            # Create random targets as fallback
            targets = np.random.randint(0, 2, size=len(valid_indices))
            multi_targets = np.random.randint(0, 4, size=len(valid_indices))
            return targets, multi_targets, mw_values
        
        # Create binary classification based on molecular weight
        # Class 0: Low MW (< median), Class 1: High MW (>= median)
        median_mw = np.median(mw_values_clean)
        targets = (mw_values >= median_mw).astype(int)
        
        # Alternative: Create multiple classes based on MW ranges
        mw_ranges = np.percentile(mw_values_clean, [25, 50, 75])
        multi_targets = np.digitize(mw_values, mw_ranges)
        
        return targets, multi_targets, mw_values
    
    def process_nci_chemical_data(self, max_compounds=10000):
        """Main processing function for NCI chemical data."""
        print("Processing NCI chemical data for PBP analysis...")
        
        # Load data
        self.load_data()
        
        # Limit to max_compounds for processing speed
        if len(self.merged_df) > max_compounds:
            self.merged_df = self.merged_df.sample(n=max_compounds, random_state=42)
            print(f"Sampled {max_compounds} compounds for processing")
        
        # Get SMILES strings
        smiles_list = self.merged_df['SMILES'].tolist()
        
        # Create molecular fingerprints
        print("Creating molecular fingerprints...")
        fingerprints, valid_indices = self.create_molecular_fingerprints(smiles_list)
        
        if fingerprints is None:
            print("Failed to create fingerprints")
            return None
        
        print(f"Created fingerprints for {len(valid_indices)} compounds")
        
        # Create target variables
        targets, multi_targets, mw_values = self.create_target_variables(valid_indices)
        
        # Create matrix structure for PBP
        print("Creating matrix structure for PBP...")
        matrix_data = self.create_matrix_structure(fingerprints, n_rows=4, n_cols=8)
        
        # Create dataset dictionary
        dataset = {
            'X': matrix_data,
            'y': targets,
            'y_multi': multi_targets,
            'mw_values': mw_values,
            'feature_names': ['Molecular_Descriptor_1', 'Molecular_Descriptor_2', 
                            'Molecular_Descriptor_3', 'Molecular_Descriptor_4'],
            'measurement_names': [f'Feature_{i+1}' for i in range(8)],
            'target_names': ['Low_MW', 'High_MW'],
            'description': f'NCI Chemical Data: {len(matrix_data)} compounds, {matrix_data.shape[1]}x{matrix_data.shape[2]} matrices'
        }
        
        # Save processed data
        self.save_dataset(dataset, 'nci_chemical')
        
        print(f"✓ NCI Chemical Data processed successfully!")
        print(f"  Shape: {matrix_data.shape}")
        print(f"  Target distribution: {np.bincount(targets)}")
        
        return dataset
    
    def save_dataset(self, dataset, name):
        """Save processed dataset."""
        # Save numpy arrays
        np.save(self.output_dir / f"{name}_X.npy", dataset['X'])
        np.save(self.output_dir / f"{name}_y.npy", dataset['y'])
        np.save(self.output_dir / f"{name}_y_multi.npy", dataset['y_multi'])
        np.save(self.output_dir / f"{name}_mw_values.npy", dataset['mw_values'])
        
        # Save metadata
        metadata = {
            'description': dataset['description'],
            'feature_names': dataset['feature_names'],
            'measurement_names': dataset['measurement_names'],
            'target_names': dataset['target_names'],
            'shape': dataset['X'].shape,
            'n_classes': len(np.unique(dataset['y'])),
            'n_classes_multi': len(np.unique(dataset['y_multi'])),
            'data_type': 'chemical_real',
            'domain': 'molecular_pharmaceutical',
            'sample_count': dataset['X'].shape[0]
        }
        
        with open(self.output_dir / f"{name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset saved to {self.output_dir}")

def main():
    """Main function to process NCI chemical data."""
    processor = NCIChemicalDataProcessor()
    
    if not processor._check_files():
        return
    
    # Process the data
    dataset = processor.process_nci_chemical_data(max_compounds=5000)
    
    if dataset is not None:
        print("\nNCI Chemical Data Summary:")
        print(f"  Samples: {dataset['X'].shape[0]}")
        print(f"  Matrix shape: {dataset['X'].shape[1]}x{dataset['X'].shape[2]}")
        print(f"  Classes: {len(np.unique(dataset['y']))}")
        print(f"  Multi-classes: {len(np.unique(dataset['y_multi']))}")
        print(f"  MW range: {dataset['mw_values'].min():.1f} - {dataset['mw_values'].max():.1f}")

if __name__ == "__main__":
    main() 