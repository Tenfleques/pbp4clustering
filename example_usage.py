#!/usr/bin/env python3
"""
Example Usage of PBP Codebase

This script demonstrates how to use the PBP codebase
for dimensionality reduction and clustering analysis.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


from src.pbp.core import pbp_vector, create_pbp, truncate_pBp
from src.data.loader import DatasetTransformer


def basic_example():
    """Basic PBP usage example."""
    print("=== Basic PBP Example ===")
    
    # Load dataset using the DatasetTransformer
    transformer = DatasetTransformer()
    iris_data = transformer.load_iris_dataset()
    
    if iris_data is None:
        print("Failed to load iris dataset")
        return None, None, None
    
    X = iris_data['X']
    y = iris_data['y']
    metadata = iris_data
    
    print(f"Loaded {metadata['description']}")
    print(f"Dataset shape: {X.shape}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Matrix shape per sample: {X.shape[1]}x{X.shape[2]}")
    
    # Apply PBP transformation to a few samples
    print("\nApplying PBP transformation to first 3 samples...")
    
    for i in range(min(3, X.shape[0])):
        sample_matrix = X[i]  # Shape: (2, 2) for iris
        print(f"\nSample {i+1} matrix:")
        print(sample_matrix)
        
        # Apply PBP transformation
        pbp_vector_result = pbp_vector(sample_matrix)
        print(f"PBP vector length: {len(pbp_vector_result)}")
        print(f"PBP vector: {pbp_vector_result}")
        
        # Create full PBP representation
        pbp_df = create_pbp(sample_matrix)
        print(f"PBP DataFrame:")
        print(pbp_df)
    
    return X, y, metadata


def comparison_example():
    """Compare PBP with other dimensionality reduction methods."""
    print("\n=== Method Comparison Example ===")
    
    # Load dataset
    transformer = DatasetTransformer()
    iris_data = transformer.load_iris_dataset()
    
    if iris_data is None:
        print("Failed to load iris dataset")
        return None
    
    X = iris_data['X']
    y = iris_data['y']
    
    print(f"Dataset shape: {X.shape}")
    
    # Compare different samples
    n_samples_to_compare = min(5, X.shape[0])
    results = {}
    
    for i in range(n_samples_to_compare):
        sample_matrix = X[i]
        
        # PBP transformation
        pbp_vector_result = pbp_vector(sample_matrix)
        
        # Store results
        results[f'sample_{i+1}'] = {
            'original_shape': sample_matrix.shape,
            'pbp_vector_length': len(pbp_vector_result),
            'pbp_vector': pbp_vector_result
        }
    
    # Print comparison
    print("\nPBP Transformation Results:")
    print("-" * 50)
    for sample_name, result in results.items():
        print(f"{sample_name}:")
        print(f"  Original shape: {result['original_shape']}")
        print(f"  PBP vector length: {result['pbp_vector_length']}")
        print(f"  PBP vector: {result['pbp_vector']}")
    
    return results


def clustering_example():
    """Demonstrate clustering with PBP vectors."""
    print("\n=== Clustering Example ===")
    
    # Load dataset
    transformer = DatasetTransformer()
    iris_data = transformer.load_iris_dataset()
    
    if iris_data is None:
        print("Failed to load iris dataset")
        return None
    
    X = iris_data['X']
    y = iris_data['y']
    
    print(f"Dataset shape: {X.shape}")
    
    # Apply PBP transformation to all samples
    print("Applying PBP transformation to all samples...")
    pbp_vectors = []
    
    for i in range(X.shape[0]):
        sample_matrix = X[i]
        pbp_vector_result = pbp_vector(sample_matrix)
        pbp_vectors.append(pbp_vector_result)
    
    pbp_vectors = np.array(pbp_vectors)
    print(f"PBP vectors shape: {pbp_vectors.shape}")
    
    # Apply K-means clustering
    n_clusters = len(np.unique(y))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(pbp_vectors)
    
    # Evaluate clustering
    silhouette = silhouette_score(pbp_vectors, y_pred)
    print(f"Clustering Results:")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Silhouette Score: {silhouette:.4f}")
    
    return {
        'pbp_vectors': pbp_vectors,
        'y_true': y,
        'y_pred': y_pred,
        'silhouette_score': silhouette
    }


def truncation_example():
    """Demonstrate PBP truncation."""
    print("\n=== PBP Truncation Example ===")
    
    # Create a sample matrix
    c = np.array([
        [7, 8, 2, 10, 3],
        [4, 12, 1, 8, 4],
        [5, 3, 0, 6, 9],
        [9, 6, 7, 1, 5]
    ])
    
    print("Original matrix:")
    print(c)
    
    # Create full PBP
    pbp_full = create_pbp(c)
    print(f"\nFull PBP:")
    print(pbp_full)
    
    # Apply truncation with different p values
    p_values = [2, 3, 4]
    
    for p in p_values:
        pbp_truncated = truncate_pBp(pbp_full, c, p)
        print(f"\nPBP truncated with p={p}:")
        print(pbp_truncated)


def visualization_example():
    """Demonstrate visualization of PBP results."""
    print("\n=== Visualization Example ===")
    
    # Load dataset
    transformer = DatasetTransformer()
    iris_data = transformer.load_iris_dataset()
    
    if iris_data is None:
        print("Failed to load iris dataset")
        return None
    
    X = iris_data['X']
    y = iris_data['y']
    
    # Apply PBP transformation
    pbp_vectors = []
    for i in range(X.shape[0]):
        sample_matrix = X[i]
        pbp_vector_result = pbp_vector(sample_matrix)
        pbp_vectors.append(pbp_vector_result)
    
    pbp_vectors = np.array(pbp_vectors)
    
    # Apply clustering
    n_clusters = len(np.unique(y))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(pbp_vectors)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: True labels
    scatter1 = axes[0].scatter(pbp_vectors[:, 0], pbp_vectors[:, 1], c=y, cmap='viridis', alpha=0.7)
    axes[0].set_title('Iris Dataset - True Labels (PBP)')
    axes[0].set_xlabel('PBP Component 1')
    axes[0].set_ylabel('PBP Component 2')
    plt.colorbar(scatter1, ax=axes[0])
    
    # Plot 2: Predicted labels
    scatter2 = axes[1].scatter(pbp_vectors[:, 0], pbp_vectors[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
    axes[1].set_title('Iris Dataset - Predicted Labels (PBP)')
    axes[1].set_xlabel('PBP Component 1')
    axes[1].set_ylabel('PBP Component 2')
    plt.colorbar(scatter2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('results/figures/example_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved as 'results/figures/example_visualization.png'")


def main():
    """Run all examples."""
    print("PBP Codebase Examples")
    print("=" * 50)
    
    # Run basic example
    basic_example()
    
    # Run comparison example
    comparison_example()
    
    # Run clustering example
    clustering_example()
    
    # Run truncation example
    # truncation_example()
    
    # Run visualization example
    visualization_example()
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!")


if __name__ == "__main__":
    main() 