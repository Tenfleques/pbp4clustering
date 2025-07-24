#!/usr/bin/env python3
"""
Feature Analysis Script for PBP Features

This script analyzes cached PBP features to identify the most significant columns
for clustering based on target variables. It provides multiple analysis metrics
including standard deviation, correlation with targets, and clustering impact.
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import argparse
from typing import Dict, List, Tuple, Any
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_dataset_data(dataset_name: str, data_dir: str = "data") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load PBP features and target data for a given dataset.
    
    Args:
        dataset_name: Name of the dataset
        data_dir: Directory containing the data files
        
    Returns:
        Tuple of (features, targets)
    """
    # Valid datasets for PBP analysis
    # valid_datasets = ['iris', 'breast_cancer', 'seeds', 'thyroid', 'pima', 'ionosphere']
    
    # if dataset_name not in valid_datasets:
    #     raise ValueError(f"Dataset '{dataset_name}' is not valid for PBP analysis. "
    #                    f"Valid datasets are: {valid_datasets}. "
    #                    f"Other datasets don't have natural combinatorial relationships.")
    
    features_path = os.path.join(data_dir, f"{dataset_name}_pbp_features.npy")
    targets_path = os.path.join(data_dir, f"{dataset_name}_y.npy")
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not os.path.exists(targets_path):
        raise FileNotFoundError(f"Targets file not found: {targets_path}")
    
    features = np.load(features_path)
    targets = np.load(targets_path)

    print(features[:5])
    zero_columns = np.all(features == 0, axis=0)
    print(f"Has zero columns: {zero_columns}")

    features = features[:, ~zero_columns]
    
    # Ensure targets is 1D
    if targets.ndim > 1:
        targets = targets.flatten()
    
    # Validate dimensions
    if features.shape[0] != targets.shape[0]:
        # If there's a mismatch, try to fix it
        if features.shape[0] == targets.shape[0] * 2:
            # Duplicate targets to match features
            targets = np.repeat(targets, 2)
        elif targets.shape[0] == features.shape[0] * 2:
            # Take every other target
            targets = targets[::2]
        else:
            raise ValueError(f"Dimension mismatch: features has {features.shape[0]} samples, targets has {targets.shape[0]} samples")
    
    return features, targets

def calculate_standard_deviation_importance(features: np.ndarray) -> Dict[str, Any]:
    """
    Calculate standard deviation for each feature column.
    
    Args:
        features: Feature matrix
        
    Returns:
        Dictionary with std values and rankings
    """
    std_values = np.std(features, axis=0)
    std_ranking = np.argsort(std_values)[::-1]  # Descending order
    
    return {
        'std_values': std_values,
        'std_ranking': std_ranking,
        'std_scores': std_values / np.max(std_values)  # Normalized scores
    }

def calculate_correlation_importance(features: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
    """
    Calculate correlation between features and targets.
    
    Args:
        features: Feature matrix
        targets: Target values
        
    Returns:
        Dictionary with correlation values and rankings
    """
    correlations = []
    for i in range(features.shape[1]):
        corr = np.corrcoef(features[:, i], targets)[0, 1]
        correlations.append(abs(corr) if not np.isnan(corr) else 0)
    
    correlations = np.array(correlations)
    corr_ranking = np.argsort(correlations)[::-1]
    
    return {
        'correlation_values': correlations,
        'correlation_ranking': corr_ranking,
        'correlation_scores': correlations / np.max(correlations) if np.max(correlations) > 0 else correlations
    }

def calculate_mutual_information_importance(features: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
    """
    Calculate mutual information between features and targets.
    
    Args:
        features: Feature matrix
        targets: Target values
        
    Returns:
        Dictionary with MI values and rankings
    """
    mi_scores = mutual_info_classif(features, targets, random_state=42)
    mi_ranking = np.argsort(mi_scores)[::-1]
    
    return {
        'mi_values': mi_scores,
        'mi_ranking': mi_ranking,
        'mi_scores': mi_scores / np.max(mi_scores) if np.max(mi_scores) > 0 else mi_scores
    }

def calculate_f_statistic_importance(features: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
    """
    Calculate F-statistic for feature selection.
    
    Args:
        features: Feature matrix
        targets: Target values
        
    Returns:
        Dictionary with F-statistic values and rankings
    """
    f_scores, _ = f_classif(features, targets)
    f_ranking = np.argsort(f_scores)[::-1]
    
    return {
        'f_values': f_scores,
        'f_ranking': f_ranking,
        'f_scores': f_scores / np.max(f_scores) if np.max(f_scores) > 0 else f_scores
    }

def calculate_clustering_impact(features: np.ndarray, targets: np.ndarray, n_clusters: int = None) -> Dict[str, Any]:
    """
    Calculate clustering impact by measuring silhouette scores with and without each feature.
    
    Args:
        features: Feature matrix
        targets: Target values
        n_clusters: Number of clusters (defaults to number of unique targets)
        
    Returns:
        Dictionary with clustering impact scores
    """
    if n_clusters is None:
        n_clusters = len(np.unique(targets))
    
    try:
        # Baseline clustering with all features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Check if we have enough unique clusters for silhouette score
        unique_clusters = len(np.unique(cluster_labels))
        if unique_clusters < 2:
            # Fallback: use all features as baseline
            baseline_silhouette = 0.0
        else:
            baseline_silhouette = silhouette_score(features_scaled, cluster_labels)
        
        # Calculate impact of removing each feature
        impact_scores = []
        for i in range(features.shape[1]):
            try:
                # Remove feature i
                features_reduced = np.delete(features_scaled, i, axis=1)
                
                # Re-cluster
                kmeans_reduced = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels_reduced = kmeans_reduced.fit_predict(features_reduced)
                
                # Check if we have enough unique clusters
                unique_clusters_reduced = len(np.unique(cluster_labels_reduced))
                if unique_clusters_reduced < 2:
                    reduced_silhouette = 0.0
                else:
                    reduced_silhouette = silhouette_score(features_reduced, cluster_labels_reduced)
                
                # Impact is the difference (higher is better)
                impact = baseline_silhouette - reduced_silhouette
                impact_scores.append(impact)
                
            except Exception as e:
                # If clustering fails for this feature, use 0 impact
                impact_scores.append(0.0)
        
        impact_scores = np.array(impact_scores)
        impact_ranking = np.argsort(impact_scores)[::-1]
        
        return {
            'impact_values': impact_scores,
            'impact_ranking': impact_ranking,
            'impact_scores': impact_scores / np.max(np.abs(impact_scores)) if np.max(np.abs(impact_scores)) > 0 else impact_scores,
            'baseline_silhouette': baseline_silhouette
        }
        
    except Exception as e:
        # If clustering fails entirely, return zeros
        n_features = features.shape[1]
        impact_scores = np.zeros(n_features)
        impact_ranking = np.arange(n_features)
        
        return {
            'impact_values': impact_scores,
            'impact_ranking': impact_ranking,
            'impact_scores': impact_scores,
            'baseline_silhouette': 0.0
        }

def calculate_combined_importance(metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate combined importance score from multiple metrics.
    
    Args:
        metrics: Dictionary containing all metric results
        
    Returns:
        Dictionary with combined scores and rankings
    """
    # Combine all normalized scores
    combined_scores = (
        metrics['std']['std_scores'] +
        metrics['correlation']['correlation_scores'] +
        metrics['mutual_info']['mi_scores'] +
        metrics['f_statistic']['f_scores'] +
        metrics['clustering_impact']['impact_scores']
    ) / 5.0  # Average of all metrics
    
    combined_ranking = np.argsort(combined_scores)[::-1]
    
    return {
        'combined_scores': combined_scores,
        'combined_ranking': combined_ranking
    }

def extract_top_features(features: np.ndarray, results: Dict[str, Any], k: int = 3) -> Tuple[np.ndarray, List[int]]:
    """
    Extract top-k features from the dataset, dropping constant features.
    
    Args:
        features: Original feature matrix
        results: Analysis results containing combined rankings
        k: Maximum number of features to extract (default: 3)
        
    Returns:
        Tuple of (extracted_features, selected_indices)
    """
    # Get combined ranking
    if 'combined' not in results['top_features']:
        raise ValueError("Combined ranking not available in results")
    
    combined_ranking = results['top_features']['combined']['indices']
    
    # Check for constant features (zero variance)
    feature_vars = np.var(features, axis=0)
    non_constant_indices = np.where(feature_vars > 0)[0]
    
    if len(non_constant_indices) == 0:
        raise ValueError("All features are constant")
    
    # Filter ranking to only include non-constant features
    valid_ranking = [idx for idx in combined_ranking if idx in non_constant_indices]
    
    # Take top-k features (or all if less than k)
    k = min(k, len(valid_ranking))
    selected_indices = valid_ranking[:k]
    
    # Extract the selected features
    extracted_features = features[:, selected_indices]
    
    return extracted_features, selected_indices

def analyze_features(dataset_name: str, top_k: int = 10, data_dir: str = "data") -> Dict[str, Any]:
    """
    Perform comprehensive feature analysis for a dataset.
    
    Args:
        dataset_name: Name of the dataset to analyze
        top_k: Number of top features to return
        data_dir: Directory containing the data files
        
    Returns:
        Dictionary containing all analysis results
    """
    print(f"Loading data for dataset: {dataset_name}")
    features, targets = load_dataset_data(dataset_name, data_dir)
    
    print(f"Feature matrix shape: {features.shape}")
    print(f"Target vector shape: {targets.shape}")
    print(f"Number of unique targets: {len(np.unique(targets))}")
    
    # Calculate all metrics
    print("Calculating feature importance metrics...")
    
    metrics = {
        'std': calculate_standard_deviation_importance(features),
        'correlation': calculate_correlation_importance(features, targets),
        'mutual_info': calculate_mutual_information_importance(features, targets),
        'f_statistic': calculate_f_statistic_importance(features, targets),
        'clustering_impact': calculate_clustering_impact(features, targets)
    }
    
    # Calculate combined importance
    metrics['combined'] = calculate_combined_importance(metrics)
    
    # Prepare results
    results = {
        'dataset_name': dataset_name,
        'feature_count': features.shape[1],
        'sample_count': features.shape[0],
        'target_count': len(np.unique(targets)),
        'metrics': metrics,
        'top_features': {}
    }
    
    # Get top features for each metric
    for metric_name, metric_data in metrics.items():
        # Check for ranking key (could be named differently)
        ranking_key = None
        for key in metric_data.keys():
            if 'ranking' in key:
                ranking_key = key
                break
        
        if ranking_key:
            top_indices = metric_data[ranking_key][:top_k]
            
            # Handle different metric naming patterns
            if metric_name == 'combined':
                scores_key = 'combined_scores'
                values_key = 'combined_scores'
            elif metric_name == 'clustering_impact':
                scores_key = 'impact_scores'
                values_key = 'impact_values'
            elif metric_name == 'std':
                scores_key = 'std_scores'
                values_key = 'std_values'
            elif metric_name == 'correlation':
                scores_key = 'correlation_scores'
                values_key = 'correlation_values'
            elif metric_name == 'mutual_info':
                scores_key = 'mi_scores'
                values_key = 'mi_values'
            elif metric_name == 'f_statistic':
                scores_key = 'f_scores'
                values_key = 'f_values'
            else:
                scores_key = f'{metric_name.split("_")[0]}_scores'
                values_key = f'{metric_name.split("_")[0]}_values'
            
            # Check if keys exist
            if scores_key not in metric_data:
                continue
            if values_key not in metric_data:
                continue
            
            top_features = {
                'indices': top_indices.tolist(),
                'scores': metric_data[scores_key][top_indices].tolist(),
                'values': metric_data[values_key][top_indices].tolist()
            }
            results['top_features'][metric_name] = top_features
    
    return results

def print_analysis_results(results: Dict[str, Any], top_k: int = 10):
    """
    Print formatted analysis results.
    
    Args:
        results: Analysis results dictionary
        top_k: Number of top features to display
    """
    print(f"\n{'='*60}")
    print(f"FEATURE ANALYSIS RESULTS FOR: {results['dataset_name'].upper()}")
    print(f"{'='*60}")
    print(f"Dataset Info:")
    print(f"  - Features: {results['feature_count']}")
    print(f"  - Samples: {results['sample_count']}")
    print(f"  - Target classes: {results['target_count']}")
    
    print(f"\nTop {top_k} Features by Metric:")
    print(f"{'='*60}")
    
    for metric_name, top_data in results['top_features'].items():
        print(f"\n{metric_name.upper().replace('_', ' ')}:")
        print(f"{'Index':<8} {'Score':<12} {'Value':<15}")
        print("-" * 35)
        
        for i in range(min(top_k, len(top_data['indices']))):
            idx = top_data['indices'][i]
            score = top_data['scores'][i]
            value = top_data['values'][i]
            print(f"{idx:<8} {score:<12.4f} {value:<15.4f}")
    
    # Show combined ranking
    if 'combined' in results['top_features']:
        combined_data = results['top_features']['combined']
        print(f"\nCOMBINED IMPORTANCE RANKING:")
        print(f"{'Index':<8} {'Combined Score':<15}")
        print("-" * 25)
        
        for i in range(min(top_k, len(combined_data['indices']))):
            idx = combined_data['indices'][i]
            score = combined_data['scores'][i]
            print(f"{idx:<8} {score:<15.4f}")
    else:
        print(f"\nCOMBINED IMPORTANCE RANKING: Not available")

def visualize_features(extracted_features: np.ndarray, targets: np.ndarray, 
                      selected_indices: List[int], dataset_name: str, 
                      save_plot: bool = False, output_file: str = None):
    """
    Visualize samples using the top features in a scatter plot.
    
    Args:
        extracted_features: Feature matrix with selected features
        targets: Target values
        selected_indices: Indices of selected features
        dataset_name: Name of the dataset
        save_plot: Whether to save the plot
        output_file: Output file path for the plot
    """
    n_features = extracted_features.shape[1]
    
    if n_features == 0:
        print("No features to visualize")
        return
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    if n_features == 1:
        # 1D plot
        plt.scatter(extracted_features[:, 0], np.zeros_like(extracted_features[:, 0]), 
                   c=targets, cmap='viridis', alpha=0.7)
        plt.xlabel(f'Feature {selected_indices[0]}')
        plt.ylabel('Position')
        plt.title(f'{dataset_name.upper()} - Top Feature Visualization')
        
    elif n_features == 2:
        # 2D plot
        plt.scatter(extracted_features[:, 0], extracted_features[:, 1], 
                   c=targets, cmap='viridis', alpha=0.7)
        plt.xlabel(f'Feature {selected_indices[0]}')
        plt.ylabel(f'Feature {selected_indices[1]}')
        plt.title(f'{dataset_name.upper()} - Top 2 Features Visualization')
        
    elif n_features == 3:
        # 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(extracted_features[:, 0], extracted_features[:, 1], extracted_features[:, 2], 
                           c=targets, cmap='viridis', alpha=0.7)
        ax.set_xlabel(f'Feature {selected_indices[0]}')
        ax.set_ylabel(f'Feature {selected_indices[1]}')
        ax.set_zlabel(f'Feature {selected_indices[2]}')
        ax.set_title(f'{dataset_name.upper()} - Top 3 Features Visualization')
        
        # Add colorbar
        plt.colorbar(scatter)
    
    # Add legend for target classes
    unique_targets = np.unique(targets)
    if len(unique_targets) <= 10:  # Only add legend if reasonable number of classes
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=plt.cm.viridis(i/len(unique_targets)), 
                                     markersize=8, label=f'Class {i}') 
                          for i in unique_targets]
        plt.legend(handles=legend_elements, title='Target Classes')
    
    plt.tight_layout()
    
    if save_plot:
        if output_file is None:
            output_file = f"{dataset_name}_top_features_visualization.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    
    plt.show()
    plt.close()

def save_results(results: Dict[str, Any], output_file: str = None):
    """
    Save analysis results to JSON file.
    
    Args:
        results: Analysis results dictionary
        output_file: Output file path (optional)
    """
    if output_file is None:
        output_file = f"feature_analysis_{results['dataset_name']}.json"
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return obj
    
    # Deep copy and convert
    import copy
    results_copy = copy.deepcopy(results)
    
    def convert_dict(d):
        for key, value in d.items():
            if isinstance(value, dict):
                convert_dict(value)
            else:
                d[key] = convert_numpy(value)
    
    convert_dict(results_copy)
    
    with open(output_file, 'w') as f:
        json.dump(results_copy, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


def analyze_dataset(dataset_name: str, args: argparse.Namespace):
    """
    Analyze a dataset and return the results.
    """
    # Perform analysis
    results = analyze_features(dataset_name, args.top_k, args.data_dir)
    
    # Print results
    print_analysis_results(results, args.top_k)
    
    # Save results if requested
    if args.save or args.output:
        save_results(results, args.output)
    
    # Extract and visualize top features if requested
    if args.visualize:
        print(f"\nExtracting top {args.k} features for visualization...")
        
        # Load original data for extraction
        features, targets = load_dataset_data(dataset_name, args.data_dir)
        
        # Extract top features
        extracted_features, selected_indices = extract_top_features(features, results, args.k)
        
        print(f"Selected features: {selected_indices}")
        print(f"Extracted features shape: {extracted_features.shape}")
        
        # Visualize
        visualize_features(extracted_features, targets, selected_indices, 
                        dataset_name, args.save_plot, args.plot_output)

def main():
    """Main function to run feature analysis."""
    parser = argparse.ArgumentParser(description="Analyze PBP features for clustering significance")
    parser.add_argument("-d", "--dataset_name", default='all', help="Name of the dataset to analyze")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top features to return (default: 10)")
    parser.add_argument("--k", type=int, default=3, help="Maximum number of features to extract for visualization (default: 3)")
    parser.add_argument("--data-dir", default="data", help="Directory containing data files (default: data)")
    parser.add_argument("--output", help="Output JSON file path (optional)")
    parser.add_argument("--save", action="store_true", help="Save results to JSON file")
    parser.add_argument("--visualize", action="store_true", help="Create visualization of top features")
    parser.add_argument("--save-plot", action="store_true", help="Save visualization plot")
    parser.add_argument("--plot-output", help="Output file path for the plot (optional)")
    
    args = parser.parse_args()
    
    try:
        if args.dataset_name == 'all':
            datasets = ['iris', 'breast_cancer', 'wine', 'digits', 'diabetes',
                        'sonar', 'glass', 'vehicle', 'ecoli', 'yeast',
                        'seeds', 'thyroid', 'pima', 'ionosphere', 'glass_conforming']
            for dataset in datasets:
                print(f"Analyzing dataset: {dataset}")
                analyze_dataset(dataset, args)
        else:
            analyze_dataset(args.dataset_name, args)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 