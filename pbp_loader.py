
import pandas as pd
import json
import os

def load_pbp_dataset(dataset_name):
    """
    Load a PBP dataset.
    
    Args:
        dataset_name (str): Name of the dataset ('spectroscopy', 'protein', 'dna', 'crystal')
    
    Returns:
        tuple: (data, metadata, target_labels)
    """
    base_path = 'pbp_datasets'
    
    # Load data
    data_path = os.path.join(base_path, f'{dataset_name}_data.csv')
    data = pd.read_csv(data_path, index_col=0)
    
    # Load metadata
    metadata_path = os.path.join(base_path, f'{dataset_name}_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Create target labels (for clustering)
    target_labels = list(data.index)
    
    return data, metadata, target_labels

def get_available_datasets():
    """Get list of available PBP datasets."""
    base_path = 'pbp_datasets'
    datasets = []
    
    if os.path.exists(base_path):
        for file in os.listdir(base_path):
            if file.endswith('_data.csv'):
                dataset_name = file.replace('_data.csv', '')
                datasets.append(dataset_name)
    
    return datasets

def print_dataset_info(dataset_name):
    """Print information about a specific dataset."""
    try:
        data, metadata, targets = load_pbp_dataset(dataset_name)
        
        print(f"Dataset: {dataset_name}")
        print(f"Shape: {data.shape}")
        print(f"Structure: {metadata['structure']}")
        print(f"Source: {metadata['source']}")
        print(f"Natural Relationships: {metadata['natural_relationships']}")
        print(f"Features: {metadata['features']}")
        print(f"Samples: {metadata.get('compounds', metadata.get('proteins', metadata.get('sequences', metadata.get('materials', []))))}")
        print()
        
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
