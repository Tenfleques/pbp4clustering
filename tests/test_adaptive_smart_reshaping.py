#!/usr/bin/env python3
"""
Test script for adaptive smart reshaping with automatic strategy selection.

This script demonstrates the adaptive approach that tries multiple rearrangement
strategies and automatically selects the best one based on comprehensive evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from src.data.consolidated_loader import ConsolidatedDatasetLoader

def test_adaptive_approach(dataset_name):
    """
    Test the adaptive approach on a specific dataset.
    
    Args:
        dataset_name: Name of the dataset to test
        
    Returns:
        Dictionary with test results
    """
    print(f"\n{'='*80}")
    print(f"ADAPTIVE SMART RESHAPING TEST: {dataset_name.upper()}")
    print(f"{'='*80}")
    
    # Load dataset
    loader = ConsolidatedDatasetLoader()
    dataset = loader.load_dataset(dataset_name)
    
    if dataset is None:
        print(f"Failed to load {dataset_name} dataset")
        return None
    
    X = dataset['X']
    y = dataset['y']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Check if adaptive reshaping was used
    if 'reshaping_method' in dataset and dataset['reshaping_method'] == 'adaptive_smart_reshape':
        print(f"\n✅ Adaptive smart reshaping applied!")
        
        # Analyze results
        selected_strategy = dataset.get('selected_strategy', 'unknown')
        evaluation_metrics = dataset.get('evaluation_metrics', {})
        feature_groups = dataset.get('feature_groups', [])
        
        print(f"\n🎯 Selected Strategy: {selected_strategy}")
        
        print(f"\n📊 Evaluation Metrics:")
        for metric, value in evaluation_metrics.items():
            if metric != 'clustering_metrics':
                print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
        
        print(f"\n🔢 Feature Groups:")
        for i, group in enumerate(feature_groups):
            print(f"  Row {i+1}: {len(group)} features - {group}")
        
        print(f"\n🎯 Clustering Metrics:")
        clustering_metrics = evaluation_metrics.get('clustering_metrics', {})
        for metric, value in clustering_metrics.items():
            print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
        
        return {
            'dataset_name': dataset_name,
            'shape': X.shape,
            'selected_strategy': selected_strategy,
            'evaluation_metrics': evaluation_metrics,
            'feature_groups': feature_groups
        }
    else:
        print(f"❌ Adaptive smart reshaping not applied to {dataset_name}")
        return None

def compare_strategies_manually(dataset_name):
    """
    Manually compare different strategies for a dataset.
    
    Args:
        dataset_name: Name of the dataset to test
    """
    print(f"\n{'='*80}")
    print(f"MANUAL STRATEGY COMPARISON: {dataset_name.upper()}")
    print(f"{'='*80}")
    
    loader = ConsolidatedDatasetLoader()
    
    # Load original data
    if dataset_name == 'glass':
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
        X, y, columns = loader.download_uci_dataset("Glass", url)
    elif dataset_name == 'vehicle':
        # For vehicle, we'll use synthetic data for comparison
        X = np.random.randn(100, 18)
        y = np.random.randint(0, 4, 100)
    else:
        print(f"Dataset {dataset_name} not supported for manual comparison")
        return
    
    if X is None:
        print(f"Failed to load {dataset_name} data")
        return
    
    print(f"Original data shape: {X.shape}")
    
    # Test different strategies manually
    strategies = {
        'homogeneity_only': {
            'balance_weights': {'homogeneity': 1.0, 'importance': 0.0, 'diversity': 0.0, 'balance': 0.0},
            'use_feature_importance': False
        },
        'importance_focused': {
            'balance_weights': {'homogeneity': 0.2, 'importance': 0.6, 'diversity': 0.1, 'balance': 0.1},
            'use_feature_importance': True
        },
        'diversity_focused': {
            'balance_weights': {'homogeneity': 0.2, 'importance': 0.2, 'diversity': 0.5, 'balance': 0.1},
            'use_feature_importance': True
        },
        'balanced': {
            'balance_weights': {'homogeneity': 0.35, 'importance': 0.35, 'diversity': 0.2, 'balance': 0.1},
            'use_feature_importance': True
        },
        'homogeneity_focused': {
            'balance_weights': {'homogeneity': 0.6, 'importance': 0.2, 'diversity': 0.1, 'balance': 0.1},
            'use_feature_importance': True
        }
    }
    
    results = {}
    
    for strategy_name, config in strategies.items():
        print(f"\n🔧 Testing {strategy_name}...")
        
        try:
            reshaped_X, feature_groups, col_indices, optimization_metrics = dt.smart_reshape_with_homogeneity(
                X, target_rows=2, balance_weights=config['balance_weights'], 
                use_feature_importance=config['use_feature_importance']
            )
            
            # Evaluate clustering quality
            X_2d = reshaped_X.reshape(reshaped_X.shape[0], -1)
            kmeans = KMeans(n_clusters=min(5, len(np.unique(y))), random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_2d)
            
            clustering_metrics = {
                'silhouette_score': silhouette_score(X_2d, cluster_labels),
                'calinski_harabasz_score': calinski_harabasz_score(X_2d, cluster_labels),
                'davies_bouldin_score': davies_bouldin_score(X_2d, cluster_labels),
                'inertia': kmeans.inertia_
            }
            
            # Calculate feature distribution score
            group_sizes = [len(group) for group in feature_groups]
            feature_distribution_score = np.var(group_sizes)
            
            results[strategy_name] = {
                'shape': reshaped_X.shape,
                'clustering_metrics': clustering_metrics,
                'feature_distribution': feature_distribution_score,
                'optimization_metrics': optimization_metrics,
                'feature_groups': feature_groups
            }
            
            print(f"  ✅ Shape: {reshaped_X.shape}")
            print(f"  📊 Silhouette: {clustering_metrics['silhouette_score']:.4f}")
            print(f"  📊 Feature Distribution: {feature_distribution_score:.4f}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    # Print comparison summary
    print(f"\n{'='*80}")
    print("STRATEGY COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    for strategy_name, result in results.items():
        print(f"\n📊 {strategy_name.upper()}:")
        print(f"  Shape: {result['shape']}")
        print(f"  Silhouette Score: {result['clustering_metrics']['silhouette_score']:.4f}")
        print(f"  Calinski-Harabasz: {result['clustering_metrics']['calinski_harabasz_score']:.4f}")
        print(f"  Davies-Bouldin: {result['clustering_metrics']['davies_bouldin_score']:.4f}")
        print(f"  Feature Distribution: {result['feature_distribution']:.4f}")
        print(f"  Feature Groups: {[len(g) for g in result['feature_groups']]}")

def visualize_adaptive_results(dataset_name):
    """
    Visualize the results of adaptive reshaping.
    
    Args:
        dataset_name: Name of the dataset to visualize
    """
    loader = ConsolidatedDatasetLoader()
    dataset = loader.load_dataset(dataset_name)
    
    if dataset is None or 'evaluation_metrics' not in dataset:
        print(f"Cannot visualize adaptive results for {dataset_name}")
        return
    
    evaluation_metrics = dataset['evaluation_metrics']
    selected_strategy = dataset.get('selected_strategy', 'unknown')
    feature_groups = dataset.get('feature_groups', [])
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Strategy selection
    strategies = ['homogeneity_only', 'importance_focused', 'diversity_focused', 'balanced', 'homogeneity_focused']
    strategy_scores = [0.6835, 0.5877, 0.5877, 0.5877, 0.5877]  # Example scores
    
    axes[0, 0].bar(strategies, strategy_scores)
    axes[0, 0].set_title('Strategy Comparison Scores')
    axes[0, 0].set_ylabel('Combined Score (lower is better)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].axhline(y=min(strategy_scores), color='r', linestyle='--', label=f'Best: {selected_strategy}')
    axes[0, 0].legend()
    
    # Plot 2: Feature group sizes
    group_sizes = [len(group) for group in feature_groups]
    axes[0, 1].bar(range(len(group_sizes)), group_sizes)
    axes[0, 1].set_title('Feature Group Sizes')
    axes[0, 1].set_xlabel('Group Index')
    axes[0, 1].set_ylabel('Number of Features')
    
    # Plot 3: Evaluation metrics
    metric_names = ['clustering_quality', 'homogeneity', 'feature_distribution', 'importance', 'diversity', 'balance']
    metric_values = [evaluation_metrics.get(name, 0) for name in metric_names]
    
    # Normalize values for better visualization
    normalized_values = [v / max(metric_values) if max(metric_values) > 0 else 0 for v in metric_values]
    
    axes[0, 2].bar(metric_names, normalized_values)
    axes[0, 2].set_title('Normalized Evaluation Metrics')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # Plot 4: Clustering metrics
    clustering_metrics = evaluation_metrics.get('clustering_metrics', {})
    if clustering_metrics:
        cluster_names = list(clustering_metrics.keys())
        cluster_values = list(clustering_metrics.values())
        
        # Normalize clustering metrics
        max_cluster = max(cluster_values) if cluster_values else 1
        normalized_cluster = [v / max_cluster for v in cluster_values]
        
        axes[1, 0].bar(cluster_names, normalized_cluster)
        axes[1, 0].set_title('Normalized Clustering Metrics')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 5: Strategy comparison pie chart
    strategy_counts = {'homogeneity_only': 1, 'importance_focused': 2, 'diversity_focused': 0, 
                      'balanced': 0, 'homogeneity_focused': 0}
    strategy_names = list(strategy_counts.keys())
    strategy_values = list(strategy_counts.values())
    
    # Only show strategies with counts > 0
    non_zero_indices = [i for i, v in enumerate(strategy_values) if v > 0]
    non_zero_names = [strategy_names[i] for i in non_zero_indices]
    non_zero_values = [strategy_values[i] for i in non_zero_indices]
    
    if non_zero_values:
        axes[1, 1].pie(non_zero_values, labels=non_zero_names, autopct='%1.1f%%')
        axes[1, 1].set_title('Strategy Distribution')
    
    # Plot 6: Combined score breakdown
    score_components = ['clustering_quality', 'homogeneity', 'feature_distribution']
    component_values = [evaluation_metrics.get(name, 0) for name in score_components]
    
    axes[1, 2].bar(score_components, component_values)
    axes[1, 2].set_title('Score Component Breakdown')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.suptitle(f'Adaptive Smart Reshaping Analysis: {dataset_name.upper()}\nSelected Strategy: {selected_strategy}')
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_adaptive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to test adaptive smart reshaping."""
    print("Adaptive Smart Reshaping Test Suite")
    print("=" * 60)
    
    # Test datasets that use adaptive smart reshaping
    test_datasets = ['glass', 'vehicle', 'ecoli', 'yeast']
    
    results = []
    
    # Test each dataset
    for dataset_name in test_datasets:
        result = test_adaptive_approach(dataset_name)
        if result:
            results.append(result)
    
    # Print comprehensive summary
    print(f"\n{'='*80}")
    print("ADAPTIVE SMART RESHAPING SUMMARY")
    print(f"{'='*80}")
    
    for result in results:
        print(f"\n📊 {result['dataset_name'].upper()}:")
        print(f"  Shape: {result['shape']}")
        print(f"  Selected Strategy: {result['selected_strategy']}")
        
        evaluation_metrics = result['evaluation_metrics']
        print(f"  🎯 Combined Score: {evaluation_metrics.get('combined_score', 0):.4f}")
        print(f"  🎯 Clustering Quality: {evaluation_metrics.get('clustering_quality', 0):.4f}")
        print(f"  🎯 Feature Distribution: {evaluation_metrics.get('feature_distribution', 0):.4f}")
        
        clustering_metrics = evaluation_metrics.get('clustering_metrics', {})
        print(f"  🎯 Silhouette Score: {clustering_metrics.get('silhouette_score', 0):.4f}")
        print(f"  🎯 Calinski-Harabasz: {clustering_metrics.get('calinski_harabasz_score', 0):.4f}")
        print(f"  🎯 Davies-Bouldin: {clustering_metrics.get('davies_bouldin_score', 0):.4f}")
        
        print(f"  🔢 Feature Groups: {[len(g) for g in result['feature_groups']]}")
    
    # Manual strategy comparison for one dataset
    print(f"\n{'='*80}")
    print("MANUAL STRATEGY COMPARISON")
    print(f"{'='*80}")
    
    compare_strategies_manually('glass')
    
    # Visualize results for one dataset
    print(f"\n📊 Visualizing adaptive results for glass dataset...")
    visualize_adaptive_results('glass')

if __name__ == "__main__":
    main()