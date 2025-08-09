import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
import sys
import os
from sklearn.preprocessing import LabelEncoder

from pbp_transform import pbp_vector
from visualize import scatter_pca_2d

def preprocess_dataset(ds):
    """
    Preprocess dataset to ensure:
    1. y is numeric
    2. target_names are properly formatted (l_1, l_2, etc. if needed)
    
    Args:
        ds: Dataset dictionary with 'y' and 'target_names' keys
        
    Returns:
        dict: Preprocessed dataset
    """
    if not isinstance(ds, dict) or 'y' not in ds or 'target_names' not in ds:
        return ds
    
    y = ds['y']
    target_names = ds['target_names']
    
    # Convert y to numeric if it's not already
    if not np.issubdtype(y.dtype, np.number):
        le = LabelEncoder()
        y_numeric = le.fit_transform(y)
        unique_targets = le.classes_
        
        # Update target_names if there's a mismatch
        if len(target_names) == 1 and len(unique_targets) > 1:
            # Create labels l_1, l_2, ..., l_n
            target_names = [f'{target_names[0]}_{i+1}' for i in range(len(unique_targets))]
        elif len(target_names) != len(unique_targets):
            # Use the encoded class names or create generic labels
            target_names = [f'{target_names[0]}_{i+1}' for i in range(len(unique_targets))]
    else:
        # y is already numeric
        unique_targets = np.unique(y)
        
        # Check if target_names need updating
        if len(target_names) == 1 and len(unique_targets) > 1:
            # Create labels l_1, l_2, ..., l_n
            target_names = [f'{target_names[0]}_{i+1}' for i in range(len(unique_targets))]
        elif len(target_names) != len(unique_targets):
            # Create generic labels
            target_names = [f'{target_names[0]}_{i+1}' for i in range(len(unique_targets))]
    
    # Create updated dataset
    updated_ds = ds.copy()
    updated_ds['y'] = y_numeric if 'y_numeric' in locals() else y
    updated_ds['target_names'] = target_names
    
    return updated_ds


def vis(pbp_features, y, labels, title):
    plt.figure(figsize=(16, 8), dpi=120)

    colors = ['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'brown', 'pink', 'gray', 'black']
    markers = ['o', 's', 'd', 'v', '^', 'p', 'h', 'x', 'D', '1']
    
    # Check if we have multiple columns for 2D or 3D plot
    if pbp_features.shape[1] >= 3:
        # 3D plot
        ax = plt.axes(projection='3d')
        for i in range(len(labels)):
            ax.scatter(pbp_features[y==i, 2], pbp_features[y==i, 1], pbp_features[y==i, 0], 
                      s=5, c=colors[i], label=labels[i], marker=markers[i])
        ax.set_zlabel("aggregated (min)")
        ax.set_ylabel("aggregated y_2")
        ax.set_xlabel("aggregated y_3")
    elif pbp_features.shape[1] >= 2:
        # 2D plot
        for i in range(len(labels)):
            plt.scatter(pbp_features[y==i, 1], pbp_features[y==i, 0], s=5, c=colors[i], label=labels[i], marker=markers[i])
        plt.ylabel("aggregated (min)")
        plt.xlabel("aggregated y_2")
    else:
        # 1D plot - use the single column for y-axis and sample index for x-axis
        for i in range(len(labels)):
            mask = y == i
            plt.scatter(np.arange(len(pbp_features[mask])), pbp_features[mask, 0], s=5, c=colors[i], label=labels[i], marker=markers[i])
        plt.ylabel("PBP Feature Value")
        plt.xlabel("Sample Index")

    plt.title(f"Scatter plot of pseudo-Boolean polynomials of {title}")
    plt.legend()
    plt.show()
    plt.close()


def process_dataset(X, y, labels, title, verbose=False):
    pbp_features = []
    for i, x in enumerate(X):
        v = pbp_vector(x)
        pbp_features.append(v)

    pbp_features = np.array(pbp_features)
    non_zero_cols = ~(np.all(pbp_features == 0, axis=0))
    pbp_features = pbp_features[:, non_zero_cols]
    if verbose:
        print(pbp_features[:5])

    vis(pbp_features, y, labels, title)

if __name__ == "__main__":
    data = datasets.load_iris()
    X = data.data
    X = X.reshape(-1, 2, 2)
    y = data.target
    labels = data.target_names

    process_dataset(X, y, labels, "Iris")