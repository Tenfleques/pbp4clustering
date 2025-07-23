#!/usr/bin/env python3
"""
AIDS Antiviral Screen Data Processor for PBP Analysis

Processes the AIDS antiviral screen data into a matrix suitable for PBP clustering/classification.
"""
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path

def load_aids_screen_data(data_dir):
    data_dir = Path(data_dir)
    # Load screening result (class label)
    conc_file = data_dir / 'aids_conc_may04.txt'
    ec50_file = data_dir / 'aids_ec50_may04.txt'
    ic50_file = data_dir / 'aids_ic50_may04.txt'

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
    from sklearn.impute import SimpleImputer
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

    # Save processed data
    out_dir = Path('data/real_medical')
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / 'aids_screen_X.npy', X)
    np.save(out_dir / 'aids_screen_y.npy', y)
    metadata = {
        'description': 'AIDS Antiviral Screen: features=[Log10EC50, Log10IC50, NumExp_EC50, NumExp_IC50, StdDev_EC50, StdDev_IC50, Screening_Label] reshaped to 2x4 matrices',
        'feature_names': ['Feature_1', 'Feature_2'],
        'measurement_names': ['Measurement_1', 'Measurement_2', 'Measurement_3', 'Measurement_4'],
        'target_names': ['CI', 'CM', 'CA'],
        'shape': X.shape,
        'n_classes': 3,
        'data_type': 'antiviral_real',
        'domain': 'drug_discovery',
        'sample_count': X.shape[0]
    }
    with open(out_dir / 'aids_screen_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    return X, y, metadata, merged

def main():
    X, y, metadata, merged = load_aids_screen_data('data/real_medical/downloads/aids_screen')
    print('AIDS Antiviral Screen Data:')
    print('  Shape:', X.shape)
    print('  Features:', metadata['feature_names'])
    print('  Target names:', metadata['target_names'])
    print('  Sample row:')
    print(merged.head(3))

if __name__ == '__main__':
    main() 