
import pandas as pd
import json
import os

def load_expanded_pbp_dataset(dataset_name):
    """
    Load an expanded PBP dataset.
    
    Args:
        dataset_name (str): Name of the dataset ('spectroscopy', 'protein', 'dna', 'crystal')
    
    Returns:
        tuple: (data, metadata, target_labels)
    """
    base_path = 'expanded_pbp_datasets'
    
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

def get_available_expanded_datasets():
    """Get list of available expanded PBP datasets."""
    base_path = 'expanded_pbp_datasets'
    datasets = []
    
    if os.path.exists(base_path):
        for file in os.listdir(base_path):
            if file.endswith('_data.csv'):
                dataset_name = file.replace('_data.csv', '')
                datasets.append(dataset_name)
    
    return datasets

def print_expanded_dataset_info(dataset_name):
    """Print information about a specific expanded dataset."""
    try:
        data, metadata, targets = load_expanded_pbp_dataset(dataset_name)
        
        print(f"Expanded Dataset: {dataset_name}")
        print(f"Shape: {data.shape}")
        print(f"Structure: {metadata['structure']}")
        print(f"Source: {metadata['source']}")
        print(f"Natural Relationships: {metadata['natural_relationships']}")
        print(f"Features: {metadata['features']}")
        print(f"Samples: {metadata.get('compounds', metadata.get('proteins', metadata.get('sequences', metadata.get('materials', []))))}")
        print()
        
    except Exception as e:
        print(f"Error loading expanded dataset {dataset_name}: {e}")

def compare_dataset_sizes():
    """Compare original vs expanded dataset sizes."""
    print("DATASET SIZE COMPARISON")
    print("=" * 50)
    
    # Original datasets
    original_sizes = {
        'spectroscopy': (3, 4),
        'protein': (2, 4),
        'dna': (3, 4),
        'crystal': (3, 6)
    }
    
    # Expanded datasets
    expanded_sizes = {}
    for dataset in ['spectroscopy', 'protein', 'dna', 'crystal']:
        try:
            data, _, _ = load_expanded_pbp_dataset(dataset)
            expanded_sizes[dataset] = data.shape
        except:
            expanded_sizes[dataset] = (0, 0)
    
    print(f"{'Dataset':<15} {'Original':<12} {'Expanded':<12} {'Growth':<10}")
    print("-" * 50)
    
    for dataset in ['spectroscopy', 'protein', 'dna', 'crystal']:
        orig = original_sizes[dataset]
        exp = expanded_sizes[dataset]
        growth = f"{exp[0]/orig[0]:.1f}x" if orig[0] > 0 else "N/A"
        
        print(f"{dataset:<15} {orig[0]:>2}x{orig[1]:<9} {exp[0]:>2}x{exp[1]:<9} {growth:<10}")
