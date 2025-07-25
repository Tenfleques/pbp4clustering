#!/usr/bin/env python3
"""
Advanced NCI Chemical Data Loader for Pseudo-Boolean Polynomial Dimensionality Reduction

This module provides a specialized loader for NCI chemical data that inherits
from the base loader and implements NCI-specific processing logic.
"""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from .base_loader import BaseDatasetLoader

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("RDKit not available. Install with: pip install rdkit-pypi")

class AdvancedNCILoader(BaseDatasetLoader):
    """
    Loader for NCI chemical data.
    
    This class handles loading and preprocessing of NCI chemical data including
    SMILES strings, molecular weights, CAS numbers, and molecular descriptors.
    """
    
    def __init__(self, data_dir='./data'):
        super().__init__(data_dir)
        self.nci_data_dir = Path("data/real_medical/downloads/nci_chemical")
        
        # File paths
        self.smiles_file = self.nci_data_dir / "nsc_smiles.csv"
        self.mw_file = self.nci_data_dir / "nsc_mw_mf.csv"
        self.cas_file = self.nci_data_dir / "nsc_cas.csv"
    
    def load_dataset(self, dataset_name):
        """
        Load NCI chemical dataset.
        
        Args:
            dataset_name: Name of the dataset to load (should be 'nci_chemical')
            
        Returns:
            dict: Dictionary containing X, y, metadata
        """
        if dataset_name != 'nci_chemical':
            raise ValueError(f"Unknown NCI dataset: {dataset_name}")
        
        print(f"Loading {dataset_name} dataset...")
        
        # Check if dataset is already saved
        saved_dataset = self.load_saved_dataset(dataset_name)
        if saved_dataset is not None:
            print(f"✓ Loaded cached {dataset_name} dataset")
            return saved_dataset
        
        # Check if required files exist
        if not self._check_files():
            print("Warning: NCI chemical data files not found")
            print("Please download NCI chemical data files to the specified directory")
            return None
        
        try:
            # Load and process the data
            dataset = self._process_nci_chemical_data(max_compounds=5000)
            if dataset is not None:
                return dataset
                
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            return None
        
        return None
    
    def _check_files(self):
        """Check if required NCI files exist."""
        files = [self.smiles_file, self.mw_file, self.cas_file]
        missing_files = [f for f in files if not f.exists()]
        
        if missing_files:
            print(f"Missing files: {missing_files}")
            return False
        
        print("✓ All NCI chemical data files found")
        return True
    
    def _process_nci_chemical_data(self, max_compounds=10000):
        """Process NCI chemical data into PBP format."""
        print("Processing NCI chemical data for PBP analysis...")
        
        try:
            # Load data
            self._load_data()
            
            # Limit to max_compounds for processing speed
            if len(self.merged_df) > max_compounds:
                self.merged_df = self.merged_df.sample(n=max_compounds, random_state=42)
                print(f"Sampled {max_compounds} compounds for processing")
            
            # Get SMILES strings
            smiles_list = self.merged_df['SMILES'].tolist()
            
            # Create molecular fingerprints
            print("Creating molecular fingerprints...")
            fingerprints, valid_indices = self._create_molecular_fingerprints(smiles_list)
            
            if fingerprints is None:
                print("Failed to create fingerprints")
                return None
            
            print(f"Created fingerprints for {len(valid_indices)} compounds")
            
            # Create target variables
            targets, multi_targets, mw_values = self._create_target_variables(valid_indices)
            
            # Create matrix structure for PBP
            print("Creating matrix structure for PBP...")
            matrix_data = self._create_matrix_structure(fingerprints, n_rows=4, n_cols=8)
            
            # Create metadata
            metadata = {
                'description': f'NCI Chemical Data: {len(matrix_data)} compounds, {matrix_data.shape[1]}x{matrix_data.shape[2]} matrices',
                'feature_names': ['Molecular_Descriptor_1', 'Molecular_Descriptor_2', 
                                'Molecular_Descriptor_3', 'Molecular_Descriptor_4'],
                'measurement_names': [f'Feature_{i+1}' for i in range(8)],
                'target_names': ['Low_MW', 'High_MW'],
                'data_type': 'chemical_real',
                'domain': 'molecular_pharmaceutical',
                'original_shape': fingerprints.shape,
                'matrix_shape': matrix_data.shape[1:],
                'source': 'NCI_Chemical',
                'reshaping_strategy': 'molecular_descriptor_grouping'
            }
            
            dataset = {
                'X': matrix_data,
                'y': targets,
                'y_multi': multi_targets,
                'mw_values': mw_values,
                'metadata': metadata
            }
            
            # Save the dataset
            self.save_dataset('nci_chemical', dataset)
            
            print(f"✓ NCI Chemical Data processed successfully!")
            print(f"  Shape: {matrix_data.shape}")
            print(f"  Target distribution: {np.bincount(targets)}")
            
            return dataset
            
        except Exception as e:
            print(f"Error processing NCI chemical data: {e}")
            return None
    
    def _load_data(self):
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
    
    def _preprocess_smiles(self, smiles):
        """Preprocess SMILES string to handle common issues."""
        if not smiles or pd.isna(smiles):
            return None
        
        # Convert to string if needed
        smiles = str(smiles).strip()
        
        # Handle common problematic patterns
        # Remove [R] placeholders (common in NCI data)
        smiles = smiles.replace('[R]', '')
        
        # Remove other problematic placeholders
        smiles = smiles.replace('[*]', '')
        smiles = smiles.replace('[X]', '')
        smiles = smiles.replace('[Y]', '')
        smiles = smiles.replace('[Z]', '')
        
        # Remove charge indicators that might cause issues
        smiles = smiles.replace('[+]', '')
        smiles = smiles.replace('[-]', '')
        
        # Handle multiple dots (separators for mixtures)
        if smiles.count('.') > 2:
            # Take the longest part (likely the main compound)
            parts = smiles.split('.')
            smiles = max(parts, key=len)
        
        # Remove leading/trailing dots
        smiles = smiles.strip('.')
        
        # Handle explicit valence errors by removing problematic atoms
        # This is a simple approach - in practice, you might want more sophisticated handling
        if 'Cl' in smiles and smiles.count('Cl') > 10:  # Too many Cl atoms might indicate issues
            return None
        
        # Basic validation
        if len(smiles) < 3:
            return None
        
        # Check for basic SMILES syntax
        if not any(char in smiles for char in 'CNOSPFIBrCl'):
            return None
        
        # Remove any remaining problematic characters
        problematic_chars = ['[', ']', '\\', '/', '|', ';', ':', '"', "'"]
        for char in problematic_chars:
            smiles = smiles.replace(char, '')
        
        # Final validation
        if len(smiles) < 3:
            return None
        
        return smiles
    
    def _create_molecular_fingerprints(self, smiles_list, radius=2, nBits=2048):
        """Convert SMILES to Morgan fingerprints with improved error handling."""
        if not RDKIT_AVAILABLE:
            print("RDKit not available. Using simplified molecular descriptors.")
            return self._create_simple_descriptors(smiles_list)
        
        fingerprints = []
        valid_indices = []
        processed_count = 0
        error_count = 0
        
        print(f"Processing {len(smiles_list)} SMILES strings...")
        
        for i, smiles in enumerate(smiles_list):
            try:
                # Preprocess SMILES
                processed_smiles = self._preprocess_smiles(smiles)
                if processed_smiles is None:
                    error_count += 1
                    continue
                
                # Try to create molecule
                mol = Chem.MolFromSmiles(processed_smiles)
                if mol is not None:
                    try:
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
                        fingerprints.append(np.array(fp))
                        valid_indices.append(i)
                        processed_count += 1
                    except Exception as e:
                        error_count += 1
                        continue
                else:
                    error_count += 1
                    
            except Exception as e:
                error_count += 1
                continue
        
        print(f"SMILES processing complete: {processed_count} successful, {error_count} failed")
        
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
        processed_count = 0
        error_count = 0
        
        for i, smiles in enumerate(smiles_list):
            try:
                # Preprocess SMILES
                processed_smiles = self._preprocess_smiles(smiles)
                if processed_smiles is None:
                    error_count += 1
                    continue
                
                # Simple character-based features
                features = []
                
                # Length of SMILES
                features.append(len(processed_smiles))
                
                # Count of different atom types
                features.append(processed_smiles.count('C'))
                features.append(processed_smiles.count('N'))
                features.append(processed_smiles.count('O'))
                features.append(processed_smiles.count('S'))
                features.append(processed_smiles.count('P'))
                features.append(processed_smiles.count('F'))
                features.append(processed_smiles.count('Cl'))
                features.append(processed_smiles.count('Br'))
                features.append(processed_smiles.count('I'))
                
                # Count of bonds
                features.append(processed_smiles.count('='))
                features.append(processed_smiles.count('#'))
                features.append(processed_smiles.count('('))
                features.append(processed_smiles.count(')'))
                
                # Count of special groups
                features.append(processed_smiles.count('OH'))
                features.append(processed_smiles.count('NH'))
                features.append(processed_smiles.count('COOH'))
                features.append(processed_smiles.count('COO'))
                
                # Ring count (approximate)
                features.append(processed_smiles.count('c'))
                features.append(processed_smiles.count('n'))
                
                descriptors.append(features)
                valid_indices.append(i)
                processed_count += 1
                
            except Exception as e:
                error_count += 1
                continue
        
        print(f"Simple descriptors processing complete: {processed_count} successful, {error_count} failed")
        
        if descriptors:
            return np.array(descriptors), valid_indices
        else:
            print("Failed to create descriptors")
            return None, []
    
    def _create_molecular_descriptors(self, smiles_list):
        """Create comprehensive molecular descriptors."""
        if not RDKIT_AVAILABLE:
            return self._create_simple_descriptors(smiles_list)
        
        descriptors = []
        valid_indices = []
        processed_count = 0
        error_count = 0
        
        for i, smiles in enumerate(smiles_list):
            try:
                # Preprocess SMILES
                processed_smiles = self._preprocess_smiles(smiles)
                if processed_smiles is None:
                    error_count += 1
                    continue
                
                mol = Chem.MolFromSmiles(processed_smiles)
                if mol is not None:
                    try:
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
                        processed_count += 1
                    except Exception as e:
                        error_count += 1
                        continue
                else:
                    error_count += 1
                    
            except Exception as e:
                error_count += 1
                continue
        
        print(f"Molecular descriptors processing complete: {processed_count} successful, {error_count} failed")
        
        if descriptors:
            return np.array(descriptors), valid_indices
        else:
            print("Failed to create molecular descriptors")
            return None, []
    
    def _create_matrix_structure(self, data_matrix, n_rows=4, n_cols=8):
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
    
    def _create_target_variables(self, valid_indices):
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
    
    def load_all_datasets(self):
        """Load all NCI datasets."""
        results = {}
        try:
            results['nci_chemical'] = self.load_dataset('nci_chemical')
        except Exception as e:
            print(f"Error loading NCI datasets: {e}")
            results['nci_chemical'] = None
        return results 