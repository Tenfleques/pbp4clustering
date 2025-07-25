#!/usr/bin/env python3
"""
Real Medical Data Loader for Pseudo-Boolean Polynomial Dimensionality Reduction

This module handles downloading and processing real medical datasets from public sources:
- MetaboLights for metabolomics data
- MIMIC-III for electronic health records
- PhysioNet for physiological signals
- The Cancer Imaging Archive (TCIA) for medical imaging features
- DrugBank for pharmacogenomics data

Each dataset includes specific URLs and processing instructions.
"""

import pandas as pd
import numpy as np
import requests
import os
import json
import io
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger().setLevel(logging.ERROR)

class RealMedicalDataLoader:
    """Loads and processes real medical datasets for PBP analysis."""
    
    def __init__(self, data_dir='./data/real_medical'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.datasets = {}
        self.download_cache = self.data_dir / 'downloads'
        self.download_cache.mkdir(exist_ok=True)
        
    def download_metabolights_dataset(self):
        """
        Download metabolomics data from MetaboLights.
        
        Example Study: MTBLS1 - A metabolomics study of urinary changes in type 2 diabetes
        URL: https://www.ebi.ac.uk/metabolights/MTBLS1
        """
        dataset_name = 'metabolights_mtbls1'
        
        # Check if dataset already exists
        if self._dataset_exists(dataset_name):
            print(f"✓ Dataset {dataset_name} already exists, skipping download")
            return self._load_existing_dataset(dataset_name)
        
        print("Setting up MetaboLights metabolomics dataset download...")
        
        dataset_info = {
            'study_id': 'MTBLS1',
            'title': 'A metabolomics study of urinary changes in type 2 diabetes',
            'url': 'https://www.ebi.ac.uk/metabolights/MTBLS1',
            'ftp_base': 'https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/',
            'sample_study': 'MTBLS1',
            'key_files': [
                'm_MTBLS1_metabolite_profiling_NMR_spectroscopy.txt',
                's_MTBLS1.txt',
                'a_MTBLS1_metabolite_profiling_NMR_spectroscopy.txt'
            ],
            'download_instructions': [
                '1. Visit: https://www.ebi.ac.uk/metabolights/',
                '2. Browse studies by disease area (diabetes, cancer, etc.)',
                '3. Select study with metabolite concentration data',
                '4. Download: Sample file (s_), Assay file (a_), and Metabolite file (m_)',
                '5. Process concentration values into pathway-based matrices'
            ],
            'direct_urls': {
                'study_page': 'https://www.ebi.ac.uk/metabolights/MTBLS1',
                'ftp_data': 'https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS1/'
            }
        }
        
        try:
            # Try to download study metadata
            study_url = f"{dataset_info['ftp_base']}{dataset_info['sample_study']}/s_{dataset_info['sample_study']}.txt"
            print(f"Attempting MetaboLights download: {study_url}")
            
            response = requests.get(study_url, timeout=30)
            if response.status_code == 200:
                study_file = self.download_cache / f"s_{dataset_info['sample_study']}.txt"
                with open(study_file, 'wb') as f:
                    f.write(response.content)
                print(f"✓ Downloaded study metadata to {study_file}")
                
                # Try to download metabolite data
                metabolite_url = f"{dataset_info['ftp_base']}{dataset_info['sample_study']}/m_{dataset_info['sample_study']}_metabolite_profiling_NMR_spectroscopy_v2_maf.tsv"
                response = requests.get(metabolite_url, timeout=30)
                if response.status_code == 200:
                    metabolite_file = self.download_cache / f"m_{dataset_info['sample_study']}_metabolite_profiling_NMR_spectroscopy_v2_maf.tsv"
                    with open(metabolite_file, 'wb') as f:
                        f.write(response.content)
                    print(f"✓ Downloaded metabolite data to {metabolite_file}")
                    
                    # Process the data
                    processed_data = self._process_metabolights_data(study_file, metabolite_file, dataset_name)
                    if processed_data:
                        return processed_data
                        
        except Exception as e:
            print(f"MetaboLights download error: {e}")
        
        print("\nMETABOLIGHTS DOWNLOAD INSTRUCTIONS")
        print("="*80)
        print(f"Study Example: {dataset_info['title']}")
        print(f"URL: {dataset_info['url']}")
        print()
        print("Manual Download Steps:")
        for step in dataset_info['download_instructions']:
            print(f"  {step}")
        print()
        print("Key Files to Download:")
        for file in dataset_info['key_files']:
            print(f"  - {file}")
        print()
        print("Recommended Studies:")
        print("  - MTBLS1: Type 2 diabetes urine metabolomics")
        print("  - MTBLS2: Gut microbiome metabolomics")
        print("  - MTBLS90: Breast cancer tissue metabolomics")
        print("  - MTBLS374: COVID-19 serum metabolomics")
        
        return dataset_info
    
    def download_physionet_dataset(self):
        """
        Download physiological data from PhysioNet.
        
        Example: MIT-BIH Arrhythmia Database
        URL: https://physionet.org/content/mitdb/1.0.0/
        """
        print("Setting up PhysioNet physiological dataset download...")
        
        dataset_info = {
            'database': 'MIT-BIH Arrhythmia Database',
            'url': 'https://physionet.org/content/mitdb/1.0.0/',
            'data_type': 'ECG signals with arrhythmia annotations',
            'base_url': 'https://physionet.org/files/mitdb/1.0.0/',
            'sample_files': [
                '100.atr',  # Annotations
                '100.dat',  # Signal data
                '100.hea'   # Header
            ],
            'manual_steps': [
                '1. Visit: https://physionet.org/',
                '2. Browse databases by signal type (ECG, EEG, EMG, etc.)',
                '3. Select database with physiological measurements',
                '4. Download signal files (.dat) and annotation files (.atr)',
                '5. Use WFDB Python package for signal processing',
                '6. Extract features: time-domain, frequency-domain, nonlinear'
            ],
            'alternative_datasets': [
                'MIMIC-III Waveform Database: https://physionet.org/content/mimic3wdb/',
                'PTB Diagnostic ECG Database: https://physionet.org/content/ptbdb/',
                'Sleep-EDF Database: https://physionet.org/content/sleep-edfx/',
                'BIDMC Congestive Heart Failure: https://physionet.org/content/chfdb/'
            ]
        }
        
        try:
            # Try to download a small header file
            header_url = f"{dataset_info['base_url']}100.hea"
            print(f"Testing PhysioNet access: {header_url}")
            
            response = requests.get(header_url, timeout=30)
            if response.status_code == 200:
                header_file = self.download_cache / '100.hea'
                with open(header_file, 'wb') as f:
                    f.write(response.content)
                print(f"✓ Downloaded header file to {header_file}")
                
                # Parse header info
                with open(header_file, 'r') as f:
                    header_content = f.read()
                print("Header content preview:")
                print(header_content[:200])
                
        except Exception as e:
            print(f"PhysioNet access error: {e}")
        
        print("\nPHYSIONET DOWNLOAD INSTRUCTIONS")
        print("="*80)
        print(f"Database: {dataset_info['database']}")
        print(f"URL: {dataset_info['url']}")
        print()
        print("Download Steps:")
        for step in dataset_info['manual_steps']:
            print(f"  {step}")
        print()
        print("Alternative Databases:")
        for db in dataset_info['alternative_datasets']:
            print(f"  - {db}")
        print()
        print("Processing Strategy:")
        print("  - Extract physiological features from signals")
        print("  - Group by: time-domain, frequency-domain, nonlinear features")
        print("  - Create matrices: rows = feature types, columns = measurement windows")
        print("  - Use clinical annotations as targets (normal, abnormal, disease type)")
        
        return dataset_info
    
    def download_mimic_demo_dataset(self):
        """
        Download MIMIC-III demo dataset.
        
        URL: https://physionet.org/content/mimiciii-demo/1.4/
        """
        print("Setting up MIMIC-III demo dataset download...")
        
        dataset_info = {
            'database': 'MIMIC-III Clinical Database Demo',
            'url': 'https://physionet.org/content/mimiciii-demo/1.4/',
            'description': 'Demo version of MIMIC-III electronic health records',
            'base_url': 'https://physionet.org/files/mimiciii-demo/1.4/',
            'key_tables': [
                'PATIENTS.csv',
                'ADMISSIONS.csv',
                'CHARTEVENTS.csv',
                'LABEVENTS.csv'
            ],
            'access_requirements': [
                'Free registration required at PhysioNet',
                'Complete CITI Data or Specimens Only Research training',
                'Sign data use agreement'
            ]
        }
        
        try:
            # The MIMIC data requires authentication, but we can check if it's accessible
            patients_url = f"{dataset_info['base_url']}PATIENTS.csv"
            print(f"Checking MIMIC demo access: {patients_url}")
            
            response = requests.get(patients_url, timeout=10)
            if response.status_code == 200:
                patients_file = self.download_cache / 'PATIENTS.csv'
                with open(patients_file, 'wb') as f:
                    f.write(response.content)
                print(f"✓ Downloaded PATIENTS table to {patients_file}")
                
                # Process basic patient data
                patients_df = pd.read_csv(patients_file)
                print(f"Patient data: {patients_df.shape}")
                print(f"Columns: {list(patients_df.columns)}")
                
        except Exception as e:
            print(f"MIMIC access requires authentication: {e}")
        
        print("\nMIMIC-III DEMO DOWNLOAD INSTRUCTIONS")
        print("="*80)
        print(f"Database: {dataset_info['database']}")
        print(f"URL: {dataset_info['url']}")
        print()
        print("Access Requirements:")
        for req in dataset_info['access_requirements']:
            print(f"  - {req}")
        print()
        print("Key Tables:")
        for table in dataset_info['key_tables']:
            print(f"  - {table}")
        print()
        print("Processing Strategy:")
        print("  - Combine lab values, vital signs, and medications")
        print("  - Group by: laboratory tests, vital signs, treatments, demographics")
        print("  - Create matrices: rows = measurement types, columns = time windows")
        print("  - Use outcomes as targets (mortality, length of stay, readmission)")
        
        return dataset_info
    
    def download_gdsc_gene_expression_dataset(self):
        """
        Download GDSC gene expression data from EBI ArrayExpress.
        
        Dataset: E-MTAB-3610 - Transcriptional Profiling of 1,000 human cancer cell lines
        URL: https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-3610
        """
        dataset_name = 'gdsc_expression'
        
        # Check if dataset already exists
        if self._dataset_exists(dataset_name):
            print(f"✓ Dataset {dataset_name} already exists, skipping download")
            return self._load_existing_dataset(dataset_name)
        
        print("Downloading GDSC gene expression dataset (E-MTAB-3610)...")
        
        dataset_info = {
            'study_id': 'E-MTAB-3610',
            'title': 'Transcriptional Profiling of 1,000 human cancer cell lines',
            'url': 'https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-3610',
            'description': 'Basal expression profiles of 1,000 human cancer cell lines in GDSC panel',
            'processed_data_github': 'https://github.com/mdozmorov/E-MTAB-3610',
            'direct_download_urls': {
                'expression_matrix': 'https://github.com/mdozmorov/E-MTAB-3610/raw/main/E-MTAB-3610_matrix.csv.gz',
                'cell_annotations': 'https://github.com/mdozmorov/E-MTAB-3610/raw/main/E-MTAB-3610_cell_annotations.csv.gz'
            }
        }
        
        try:
            # Try to download expression matrix
            matrix_url = dataset_info['direct_download_urls']['expression_matrix']
            print(f"Downloading expression matrix from: {matrix_url}")
            
            response = requests.get(matrix_url, timeout=60)
            if response.status_code == 200:
                matrix_file = self.download_cache / 'E-MTAB-3610_matrix.csv.gz'
                with open(matrix_file, 'wb') as f:
                    f.write(response.content)
                print(f"✓ Downloaded expression matrix to {matrix_file}")
                
                # Try to download cell annotations
                annotations_url = dataset_info['direct_download_urls']['cell_annotations']
                print(f"Downloading cell annotations from: {annotations_url}")
                
                response = requests.get(annotations_url, timeout=30)
                if response.status_code == 200:
                    annotations_file = self.download_cache / 'E-MTAB-3610_cell_annotations.csv.gz'
                    with open(annotations_file, 'wb') as f:
                        f.write(response.content)
                    print(f"✓ Downloaded cell annotations to {annotations_file}")
                    
                    # Process the data
                    processed_data = self._process_gdsc_expression_data(matrix_file, annotations_file, dataset_name)
                    if processed_data:
                        return processed_data
                        
        except Exception as e:
            print(f"GDSC download error: {e}")
        
        print("\nGDSC GENE EXPRESSION DOWNLOAD INSTRUCTIONS")
        print("="*80)
        print(f"Study: {dataset_info['title']}")
        print(f"Study ID: {dataset_info['study_id']}")
        print(f"URL: {dataset_info['url']}")
        print()
        print("Automatic Download Available:")
        print(f"  Expression Matrix: {dataset_info['direct_download_urls']['expression_matrix']}")
        print(f"  Cell Annotations: {dataset_info['direct_download_urls']['cell_annotations']}")
        print()
        print("Alternative Sources:")
        print(f"  GitHub Repository: {dataset_info['processed_data_github']}")
        print(f"  ArrayExpress: https://www.ebi.ac.uk/arrayexpress/experiments/E-MTAB-3610/")
        print()
        print("Data Description:")
        print("  - 1,000 human cancer cell lines")
        print("  - ~20,000 genes per cell line")
        print("  - Log2-transformed gene expression values")
        print("  - Cell line annotations with tissue types")
        
        return dataset_info
    
    def download_drug_response_dataset(self):
        """
        Download drug response/pharmacogenomics dataset.
        
        Example: GDSC (Genomics of Drug Sensitivity in Cancer)
        URL: https://www.cancerrxgene.org/
        """
        print("Setting up GDSC drug response dataset download...")
        
        dataset_info = {
            'database': 'GDSC - Genomics of Drug Sensitivity in Cancer',
            'url': 'https://www.cancerrxgene.org/',
            'description': 'Drug sensitivity data for cancer cell lines',
            'downloads_page': 'https://www.cancerrxgene.org/downloads/bulk_download',
            'key_files': [
                'GDSC2_fitted_dose_response_27Oct23.xlsx',
                'GDSC2_public_raw_data_27Oct23.csv.zip',
                'Cell_Lines_Details.xlsx'
            ],
            'direct_urls': {
                'gdsc2_fitted': 'https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC2_fitted_dose_response_27Oct23.xlsx',
                'gdsc2_raw': 'https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC2_public_raw_data_27Oct23.csv.zip',
                'cell_lines': 'https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/Cell_Lines_Details.xlsx'
            },
            'manual_steps': [
                '1. Visit: https://www.cancerrxgene.org/downloads/bulk_download',
                '2. Download GDSC2 fitted dose response data (IC50 values)',
                '3. Download cell line details',
                '4. Process IC50 values into dose-response matrices'
            ],
            'alternative_sources': [
                'NCI-60 Drug Data: https://dtp.cancer.gov/discovery_development/nci-60/',
                'CCLE: https://portals.broadinstitute.org/ccle',
                'DrugBank: https://go.drugbank.com/',
                'ChEMBL: https://www.ebi.ac.uk/chembl/'
            ]
        }
        
        print("GDSC DRUG RESPONSE DATA INSTRUCTIONS")
        print("="*80)
        print(f"Database: {dataset_info['database']}")
        print(f"Downloads Page: {dataset_info['downloads_page']}")
        print()
        print("Direct Download URLs:")
        for name, url in dataset_info['direct_urls'].items():
            print(f"  {name}: {url}")
        print()
        print("Download Steps:")
        for step in dataset_info['manual_steps']:
            print(f"  {step}")
        print()
        print("Key Files:")
        for file in dataset_info['key_files']:
            print(f"  - {file}")
        print()
        print("Alternative Sources:")
        for source in dataset_info['alternative_sources']:
            print(f"  - {source}")
        print()
        print("Processing Strategy:")
        print("  - Group drugs by mechanism of action or target pathway")
        print("  - Create matrices: rows = response types (viability, apoptosis, etc.)")
        print("  - Columns = concentration levels or time points")
        print("  - Use cell line sensitivity as targets (sensitive, resistant)")
        
        return dataset_info
    
    def download_medical_imaging_features(self):
        """
        Download medical imaging feature datasets.
        
        Example: Radiomics features from TCIA
        URL: https://www.cancerimagingarchive.net/
        """
        print("Setting up medical imaging features download...")
        
        dataset_info = {
            'archive': 'The Cancer Imaging Archive (TCIA)',
            'url': 'https://www.cancerimagingarchive.net/',
            'example_collection': 'TCGA-BRCA',
            'data_types': ['CT', 'MRI', 'PET', 'Digital Pathology'],
            'feature_types': [
                'Shape features (volume, surface area, compactness)',
                'Intensity features (mean, std, skewness, kurtosis)',
                'Texture features (GLCM, GLRLM, GLSZM)',
                'Wavelet features (decomposition coefficients)'
            ],
            'tools_required': [
                'PyRadiomics: https://pyradiomics.readthedocs.io/',
                'SimpleITK: https://simpleitk.org/',
                'DICOM readers: pydicom, nibabel'
            ],
            'manual_steps': [
                '1. Visit: https://www.cancerimagingarchive.net/',
                '2. Browse collections by cancer type',
                '3. Download DICOM images',
                '4. Use PyRadiomics to extract features',
                '5. Group features by type (shape, intensity, texture)',
                '6. Create matrices for PBP analysis'
            ]
        }
        
        print("MEDICAL IMAGING FEATURES INSTRUCTIONS")
        print("="*80)
        print(f"Archive: {dataset_info['archive']}")
        print(f"URL: {dataset_info['url']}")
        print()
        print("Data Types Available:")
        for dtype in dataset_info['data_types']:
            print(f"  - {dtype}")
        print()
        print("Feature Types:")
        for ftype in dataset_info['feature_types']:
            print(f"  - {ftype}")
        print()
        print("Required Tools:")
        for tool in dataset_info['tools_required']:
            print(f"  - {tool}")
        print()
        print("Processing Steps:")
        for step in dataset_info['manual_steps']:
            print(f"  {step}")
        print()
        print("Matrix Structure Example:")
        print("  - Rows: [Shape_Features, Intensity_Features, Texture_Features]")
        print("  - Columns: [Feature_1, Feature_2, ..., Feature_N]")
        print("  - Targets: [Benign, Malignant] or [Grade_1, Grade_2, Grade_3]")
        
        return dataset_info
    
    def _process_gdsc_expression_data(self, matrix_file, annotations_file, dataset_name):
        """Process GDSC gene expression data into PBP format."""
        try:
            print(f"Processing GDSC expression data...")
            
            # Read expression matrix (genes × cell lines)
            print("Reading expression matrix...")
            expression_df = pd.read_csv(matrix_file, compression='gzip', index_col=0)
            print(f"Expression data shape: {expression_df.shape} (genes × cell lines)")
            
            # Read cell annotations
            print("Reading cell annotations...")
            annotations_df = pd.read_csv(annotations_file, compression='gzip')
            print(f"Annotations shape: {annotations_df.shape}")
            
            # Transpose to get (cell lines × genes)
            X_expression = expression_df.T.values
            print(f"Transposed expression shape: {X_expression.shape} (cell lines × genes)")
            
            # Extract tissue types for targets
            if 'Characteristics[organism part]' in annotations_df.columns:
                tissue_col = 'Characteristics[organism part]'
            elif 'Source Name' in annotations_df.columns:
                # Extract tissue from source name (format: "cell_line_tissue_id")
                tissues = annotations_df['Source Name'].str.split('_').str[2]
                annotations_df['tissue_type'] = tissues
                tissue_col = 'tissue_type'
            else:
                # Use cell line names as proxy
                tissue_col = annotations_df.columns[1]  # Usually cell line column
            
            # Create targets from tissue types
            y = pd.Categorical(annotations_df[tissue_col]).codes
            target_names = list(annotations_df[tissue_col].unique())
            
            print(f"Found {len(target_names)} tissue types: {target_names[:5]}...")
            
            # Create matrix structure for gene expression
            n_genes = X_expression.shape[1]
            
            # Group genes into functional categories
            if n_genes >= 200:  # Use a subset for manageable matrix size
                matrix_shape = (5, 8)  # 5 pathway types × 8 gene groups
                n_keep = 40
                feature_names = ['Oncogenes', 'Tumor_Suppressors', 'Cell_Cycle', 'Apoptosis', 'Metastasis']
            elif n_genes >= 32:
                matrix_shape = (4, 8)
                n_keep = 32
                feature_names = ['Oncogenes', 'Tumor_Suppressors', 'Cell_Cycle', 'DNA_Repair']
            elif n_genes >= 24:
                matrix_shape = (3, 8)
                n_keep = 24
                feature_names = ['Proliferation', 'Differentiation', 'Apoptosis']
            else:
                matrix_shape = (2, n_genes // 2)
                n_keep = (n_genes // 2) * 2
                feature_names = ['High_Expression', 'Low_Expression']
            
            # Select top variable genes (by variance)
            gene_vars = np.var(X_expression, axis=0)
            top_gene_indices = np.argsort(gene_vars)[-n_keep:]
            
            X_subset = X_expression[:, top_gene_indices]
            X_matrices = X_subset.reshape(-1, matrix_shape[0], matrix_shape[1])
            
            measurement_names = [f'Gene_Group_{i+1}' for i in range(matrix_shape[1])]
            
            dataset = {
                'X': X_matrices,
                'y': y,
                'feature_names': feature_names,
                'measurement_names': measurement_names,
                'target_names': target_names,
                'description': f'Real GDSC gene expression data reshaped to {matrix_shape[0]}x{matrix_shape[1]} matrices',
                'data_type': 'gene_expression_real',
                'preprocessing': 'log2_transformed_variance_selected',
                'original_shape': X_expression.shape,
                'matrix_shape': matrix_shape,
                'source': 'GDSC_E-MTAB-3610',
                'n_genes_selected': n_keep,
                'n_total_genes': n_genes
            }
            
            self.datasets[dataset_name] = dataset
            self._save_dataset(dataset_name, dataset)
            
            print(f"✓ Processed {dataset_name}: {X_matrices.shape[0]} cell lines, {matrix_shape} matrix structure")
            print(f"  Selected {n_keep} most variable genes from {n_genes} total genes")
            return dataset
            
        except Exception as e:
            print(f"Error processing GDSC expression data: {e}")
            return None
    
    def _process_metabolights_data(self, study_file, metabolite_file, dataset_name):
        """Process MetaboLights data into PBP format."""
        try:
            print(f"Processing MetaboLights data...")
            
            # Read study metadata
            study_df = pd.read_csv(study_file, sep='\t')
            print(f"Study data shape: {study_df.shape}")
            
            # Read metabolite data
            metabolite_df = pd.read_csv(metabolite_file, sep='\t')
            print(f"Metabolite data shape: {metabolite_df.shape}")
            
            # Extract sample IDs and target labels from study file
            sample_ids = study_df['Sample Name'].values
            target_labels = study_df['Factor Value[Metabolic syndrome]'].values
            
            # Create target mapping
            target_mapping = {'diabetes mellitus': 1, 'Control Group': 0}
            y = np.array([target_mapping.get(label, 0) for label in target_labels])
            
            # Find sample columns in metabolite data
            sample_cols = [col for col in metabolite_df.columns if col in sample_ids]
            
            if len(sample_cols) == 0:
                print("No matching sample columns found in metabolite data")
                return None
            
            print(f"Found {len(sample_cols)} matching sample columns")
            
            # Extract concentration data
            concentration_data = metabolite_df[sample_cols].values.T  # Transpose to get samples as rows
            
            # Apply log transformation to handle concentration values
            concentration_data = np.log1p(np.abs(concentration_data))
            
            # Create matrix structure based on chemical shift ranges
            n_samples = concentration_data.shape[0]
            n_features = concentration_data.shape[1]
            
            # Group features into matrix structure
            matrix_rows = 4
            matrix_cols = 4
            
            # Reshape data into matrix format
            features_per_matrix = max(1, n_features // (matrix_rows * matrix_cols))
            total_matrix_features = matrix_rows * matrix_cols * features_per_matrix
            
            if n_features > total_matrix_features:
                concentration_data = concentration_data[:, :total_matrix_features]
            elif n_features < total_matrix_features:
                # Pad with zeros
                padding = np.zeros((n_samples, total_matrix_features - n_features))
                concentration_data = np.hstack([concentration_data, padding])
            
            # Reshape to matrix format: (samples, matrix_rows, matrix_cols, features_per_matrix)
            X = concentration_data.reshape(n_samples, matrix_rows, matrix_cols, features_per_matrix)
            
            # Take mean across features_per_matrix to get final matrix
            X = np.mean(X, axis=3)  # Shape: (samples, matrix_rows, matrix_cols)
            
            # Create feature and measurement names based on chemical shift ranges
            feature_names = ['Aliphatic_Region', 'Aromatic_Region', 'Carbohydrate_Region', 'Amino_Acid_Region']
            measurement_names = [f'Metabolite_Group_{i+1}' for i in range(matrix_cols)]
            target_names = ['Control', 'Diabetes']
            
            dataset = {
                'X': X,
                'y': y,
                'feature_names': feature_names,
                'measurement_names': measurement_names,
                'target_names': target_names,
                'description': 'MetaboLights MTBLS1 diabetes metabolomics data',
                'data_type': 'metabolomics_real',
                'preprocessing': 'log1p_normalization',
                'original_shape': concentration_data.shape,
                'matrix_shape': (matrix_rows, matrix_cols),
                'source': 'MetaboLights_MTBLS1'
            }
            
            self.datasets[dataset_name] = dataset
            self._save_dataset(dataset_name, dataset)
            
            print(f"✓ Processed {dataset_name}: {X.shape[0]} samples, {dataset['matrix_shape']} matrix structure")
            return dataset
            
        except Exception as e:
            print(f"Error processing MetaboLights data: {e}")
            return None
    
    def _dataset_exists(self, dataset_name):
        """Check if dataset already exists."""
        x_file = self.data_dir / f"{dataset_name}_X.npy"
        y_file = self.data_dir / f"{dataset_name}_y.npy"
        metadata_file = self.data_dir / f"{dataset_name}_metadata.json"
        
        return x_file.exists() and y_file.exists() and metadata_file.exists()
    
    def _load_existing_dataset(self, dataset_name):
        """Load existing dataset from files."""
        try:
            x_file = self.data_dir / f"{dataset_name}_X.npy"
            y_file = self.data_dir / f"{dataset_name}_y.npy"
            metadata_file = self.data_dir / f"{dataset_name}_metadata.json"
            
            X = np.load(x_file, allow_pickle=True)
            y = np.load(y_file, allow_pickle=True)
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            dataset = {
                'X': X,
                'y': y,
                'feature_names': metadata.get('feature_names', []),
                'measurement_names': metadata.get('measurement_names', []),
                'target_names': metadata.get('target_names', []),
                'description': metadata.get('description', ''),
                'data_type': metadata.get('data_type', 'unknown'),
                'preprocessing': metadata.get('preprocessing', 'none'),
                'original_shape': tuple(metadata.get('original_shape', X.shape)),
                'matrix_shape': tuple(metadata.get('matrix_shape', (X.shape[1], X.shape[2]))),
                'source': metadata.get('source', 'unknown')
            }
            
            self.datasets[dataset_name] = dataset
            print(f"  → Loaded existing dataset: {X.shape[0]} samples, {dataset['matrix_shape']} matrix structure")
            return dataset
            
        except Exception as e:
            print(f"Error loading existing dataset {dataset_name}: {e}")
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
                'data_type': dataset.get('data_type', 'medical_real'),
                'domain': dataset.get('domain', 'medical_research'),
                'sample_count': dataset['X'].shape[0],
                'preprocessing': dataset.get('preprocessing', ''),
                'original_shape': [int(x) for x in dataset.get('original_shape', [])],
                'matrix_shape': [int(x) for x in dataset.get('matrix_shape', [])],
                'shape': [int(x) for x in dataset['X'].shape],
                'n_classes': int(len(np.unique(dataset['y']))),
                'source': dataset.get('source', 'unknown')
            }
            
            with open(self.data_dir / f"{dataset_name}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"  → Saved to {self.data_dir}")
            
        except Exception as e:
            print(f"Error saving dataset {dataset_name}: {e}")
    
    def download_all_medical_datasets(self):
        """Attempt to download all medical datasets."""
        print("=== Real Medical Data Download Pipeline ===\n")
        
        downloaded_datasets = {}
        
        # 1. GDSC gene expression data (high priority - should work automatically)
        print("1. GDSC Gene Expression Data (E-MTAB-3610)")
        print("-" * 50)
        gdsc_dataset = self.download_gdsc_gene_expression_dataset()
        if gdsc_dataset and isinstance(gdsc_dataset, dict) and 'X' in gdsc_dataset:
            downloaded_datasets['gdsc_expression'] = gdsc_dataset
        
        # 2. MetaboLights metabolomics data
        print("\n2. MetaboLights Metabolomics Data")
        print("-" * 50)
        metabolights_dataset = self.download_metabolights_dataset()
        if metabolights_dataset and isinstance(metabolights_dataset, dict) and 'X' in metabolights_dataset:
            downloaded_datasets['metabolights'] = metabolights_dataset
        
        # 3. PhysioNet physiological data
        print("\n3. PhysioNet Physiological Data")
        print("-" * 50)
        physionet_info = self.download_physionet_dataset()
        
        # 4. MIMIC-III demo data
        print("\n4. MIMIC-III Demo Dataset")
        print("-" * 50)
        mimic_info = self.download_mimic_demo_dataset()
        
        # 5. Drug response data
        print("\n5. GDSC Drug Response Data")
        print("-" * 50)
        drug_info = self.download_drug_response_dataset()
        
        # 6. Medical imaging features
        print("\n6. Medical Imaging Features")
        print("-" * 50)
        imaging_info = self.download_medical_imaging_features()
        
        print(f"\n=== Downloaded {len(downloaded_datasets)} medical datasets ===")
        
        if not downloaded_datasets:
            print("\nNOTE: Most medical datasets require:")
            print("- Registration and data use agreements")
            print("- Manual download due to large file sizes")
            print("- Specialized processing tools")
            print("\nPlease follow the instructions above to manually download the datasets.")
        else:
            print("\nSuccessfully downloaded:")
            for name in downloaded_datasets.keys():
                print(f"  ✓ {name}")
        
        return downloaded_datasets


def main():
    """Main function to demonstrate medical data loading."""
    print("=== Real Medical Data Loader Demo ===\n")
    
    loader = RealMedicalDataLoader()
    
    # Attempt to download datasets
    datasets = loader.download_all_medical_datasets()
    
    print("\n" + "="*80)
    print("MEDICAL DATASETS SUMMARY")
    print("="*80)
    
    if datasets:
        for name, dataset in datasets.items():
            if isinstance(dataset, dict) and 'X' in dataset:
                print(f"\n{name.upper()}:")
                print(f"  Shape: {dataset['X'].shape}")
                print(f"  Matrix structure: {dataset.get('matrix_shape', 'unknown')}")
                print(f"  Data type: {dataset.get('data_type', 'unknown')}")
                print(f"  Source: {dataset.get('source', 'unknown')}")
                print(f"  Description: {dataset.get('description', 'No description')}")
    else:
        print("No datasets were automatically downloaded.")
        print("Please follow the manual download instructions provided above.")
    
    print(f"\nDownload cache location: {loader.download_cache}")
    print(f"Processed data location: {loader.data_dir}")


if __name__ == "__main__":
    main() 