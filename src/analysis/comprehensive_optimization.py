#!/usr/bin/env python3
"""
Comprehensive Optimization for PBP Feature Generation

This module implements a comprehensive optimization approach that tries all adaptations
to get the best cluster scores before saving PBP features. It tests multiple strategies
and selects the optimal approach based on clustering performance.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, List, Tuple, Optional, Any
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveOptimizer:
    """
    Comprehensive optimizer that tries all adaptations to get the best cluster scores.
    
    This class implements a multi-strategy approach that tests various adaptations:
    1. Different reshaping strategies (homogeneity, importance, diversity, etc.)
    2. Transpose optimization
    3. Different target row counts
    4. Feature selection approaches
    5. Preprocessing methods
    """
    
    def __init__(self, max_strategies=10, random_state=42):
        """
        Initialize the comprehensive optimizer.
        
        Args:
            max_strategies: Maximum number of strategies to test
            random_state: Random state for reproducibility
        """
        self.max_strategies = max_strategies
        self.random_state = random_state
        self.results = {}
        self.best_result = None
        self.best_score = -np.inf
        
    def optimize_dataset(self, dataset_name: str, X: np.ndarray, y: np.ndarray, 
                        dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize a dataset using all available strategies.
        
        Args:
            dataset_name: Name of the dataset
            X: Feature matrix
            y: Target labels
            dataset_info: Dataset information
            
        Returns:
            Dictionary containing the best result and all tested strategies
        """
        logger.info(f"Starting comprehensive optimization for dataset: {dataset_name}")
        
        # Initialize results
        self.results = {}
        self.best_result = None
        self.best_score = -np.inf
        
        # Test different strategies
        strategies = self._generate_strategies(X, y, dataset_info)
        
        for i, strategy in enumerate(strategies):
            if i >= self.max_strategies:
                break
                
            logger.info(f"Testing strategy {i+1}/{len(strategies)}: {strategy['name']}")
            
            try:
                result = self._apply_strategy(X, y, strategy, dataset_info)
                if result is not None:
                    self.results[strategy['name']] = result
                    
                    # Update best result
                    if result['clustering_score'] > self.best_score:
                        self.best_score = result['clustering_score']
                        self.best_result = result
                        logger.info(f"New best score: {self.best_score:.4f} with strategy: {strategy['name']}")
                        
            except Exception as e:
                logger.warning(f"Strategy {strategy['name']} failed: {e}")
                continue
        
        # Create comprehensive result
        final_result = {
            'dataset_name': dataset_name,
            'best_result': self.best_result,
            'all_results': self.results,
            'best_score': self.best_score,
            'total_strategies_tested': len(self.results),
            'optimization_summary': self._create_optimization_summary()
        }
        
        logger.info(f"Optimization completed for {dataset_name}. Best score: {self.best_score:.4f}")
        return final_result
    
    def _generate_strategies(self, X: np.ndarray, y: np.ndarray, 
                           dataset_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate all possible strategies to test.
        
        Args:
            X: Feature matrix
            y: Target labels
            dataset_info: Dataset information
            
        Returns:
            List of strategy configurations
        """
        strategies = []
        
        # Get dataset characteristics
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Strategy 1: Original adaptive smart reshaping
        strategies.append({
            'name': 'adaptive_smart_reshape_original',
            'type': 'reshaping',
            'method': 'adaptive_smart_reshape',
            'params': {'target_rows': 3, 'max_combinations': 1000},
            'description': 'Original adaptive smart reshaping with 3 rows'
        })
        
        # Strategy 2: Adaptive smart reshaping with 2 rows
        strategies.append({
            'name': 'adaptive_smart_reshape_2rows',
            'type': 'reshaping',
            'method': 'adaptive_smart_reshape',
            'params': {'target_rows': 2, 'max_combinations': 1000},
            'description': 'Adaptive smart reshaping with 2 rows'
        })
        
        # Strategy 3: Transpose optimization
        strategies.append({
            'name': 'transpose_optimized',
            'type': 'reshaping',
            'method': 'adaptive_smart_reshape',
            'params': {'target_rows': 3, 'max_combinations': 1000, 'force_transpose': True},
            'description': 'Transpose optimization for maximum dimensionality reduction'
        })
        
        # Strategy 4: Feature selection + reshaping
        if n_features > 10:
            strategies.append({
                'name': 'feature_selection_reshaping',
                'type': 'feature_selection',
                'method': 'variance_threshold',
                'params': {'threshold': 0.01, 'target_rows': 3},
                'description': 'Feature selection with variance threshold + reshaping'
            })
        
        # Strategy 5: Standardization + reshaping
        strategies.append({
            'name': 'standardized_reshaping',
            'type': 'preprocessing',
            'method': 'standardization',
            'params': {'target_rows': 3},
            'description': 'Standardization + adaptive reshaping'
        })
        
        # Strategy 6: Normalization + reshaping
        strategies.append({
            'name': 'normalized_reshaping',
            'type': 'preprocessing',
            'method': 'normalization',
            'params': {'target_rows': 3},
            'description': 'Normalization + adaptive reshaping'
        })
        
        # Strategy 7: PCA + reshaping (for high-dimensional datasets)
        if n_features > 20:
            strategies.append({
                'name': 'pca_reshaping',
                'type': 'dimensionality_reduction',
                'method': 'pca',
                'params': {'n_components': min(20, n_features), 'target_rows': 3},
                'description': 'PCA dimensionality reduction + reshaping'
            })
        
        # Strategy 8: Agglomerative clustering + reshaping
        strategies.append({
            'name': 'agglomerative_reshaping',
            'type': 'clustering_based',
            'method': 'agglomerative',
            'params': {'n_clusters': min(10, n_samples//10), 'target_rows': 3},
            'description': 'Agglomerative clustering + reshaping'
        })
        
        # Strategy 9: K-means clustering + reshaping
        strategies.append({
            'name': 'kmeans_reshaping',
            'type': 'clustering_based',
            'method': 'kmeans',
            'params': {'n_clusters': min(10, n_samples//10), 'target_rows': 3},
            'description': 'K-means clustering + reshaping'
        })
        
        # Strategy 10: Ensemble approach
        strategies.append({
            'name': 'ensemble_reshaping',
            'type': 'ensemble',
            'method': 'ensemble',
            'params': {'target_rows': 3, 'n_estimators': 3},
            'description': 'Ensemble of multiple reshaping approaches'
        })
        
        return strategies
    
    def _apply_strategy(self, X: np.ndarray, y: np.ndarray, strategy: Dict[str, Any],
                       dataset_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Apply a specific strategy to the dataset.
        
        Args:
            X: Feature matrix
            y: Target labels
            strategy: Strategy configuration
            dataset_info: Dataset information
            
        Returns:
            Result dictionary or None if strategy fails
        """
        try:
            # Handle NaN values first
            if np.isnan(X).any():
                logger.info(f"Handling NaN values in dataset")
                X = np.nan_to_num(X, nan=0.0)
            
            # Apply preprocessing if needed
            if strategy['type'] == 'preprocessing':
                X_processed = self._apply_preprocessing(X, strategy)
            elif strategy['type'] == 'feature_selection':
                X_processed = self._apply_feature_selection(X, strategy)
            elif strategy['type'] == 'dimensionality_reduction':
                X_processed = self._apply_dimensionality_reduction(X, strategy)
            elif strategy['type'] == 'clustering_based':
                X_processed = self._apply_clustering_based(X, strategy)
            elif strategy['type'] == 'ensemble':
                X_processed = self._apply_ensemble(X, strategy)
            else:
                X_processed = X.copy()
            
            # Apply reshaping
            X_reshaped = self._apply_reshaping(X_processed, strategy, dataset_info)
            
            if X_reshaped is None:
                return None
            
            # Evaluate clustering performance
            clustering_metrics = self._evaluate_clustering(X_reshaped, y)
            
            # Create result
            result = {
                'strategy_name': strategy['name'],
                'strategy_type': strategy['type'],
                'description': strategy['description'],
                'original_shape': X.shape,
                'processed_shape': X_processed.shape,
                'reshaped_shape': X_reshaped.shape,
                'reshaped_data': X_reshaped,  # Add the reshaped data
                'clustering_metrics': clustering_metrics,
                'clustering_score': clustering_metrics['combined_score'],
                'strategy_params': strategy['params']
            }
            
            return result
            
        except Exception as e:
            logger.warning(f"Strategy {strategy['name']} failed: {e}")
            return None
    
    def _apply_preprocessing(self, X: np.ndarray, strategy: Dict[str, Any]) -> np.ndarray:
        """Apply preprocessing to the data."""
        if strategy['method'] == 'standardization':
            scaler = StandardScaler()
            return scaler.fit_transform(X)
        elif strategy['method'] == 'normalization':
            # Min-max normalization
            X_min = X.min(axis=0)
            X_max = X.max(axis=0)
            return (X - X_min) / (X_max - X_min + 1e-8)
        else:
            return X.copy()
    
    def _apply_feature_selection(self, X: np.ndarray, strategy: Dict[str, Any]) -> np.ndarray:
        """Apply feature selection to the data."""
        if strategy['method'] == 'variance_threshold':
            from sklearn.feature_selection import VarianceThreshold
            threshold = strategy['params']['threshold']
            selector = VarianceThreshold(threshold=threshold)
            return selector.fit_transform(X)
        else:
            return X.copy()
    
    def _apply_dimensionality_reduction(self, X: np.ndarray, strategy: Dict[str, Any]) -> np.ndarray:
        """Apply dimensionality reduction to the data."""
        if strategy['method'] == 'pca':
            from sklearn.decomposition import PCA
            n_components = strategy['params']['n_components']
            pca = PCA(n_components=n_components, random_state=self.random_state)
            return pca.fit_transform(X)
        else:
            return X.copy()
    
    def _apply_clustering_based(self, X: np.ndarray, strategy: Dict[str, Any]) -> np.ndarray:
        """Apply clustering-based preprocessing."""
        if strategy['method'] == 'kmeans':
            n_clusters = strategy['params']['n_clusters']
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            # Use cluster centers as new features
            return kmeans.cluster_centers_[cluster_labels]
        elif strategy['method'] == 'agglomerative':
            from sklearn.cluster import AgglomerativeClustering
            n_clusters = strategy['params']['n_clusters']
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = clustering.fit_predict(X)
            # Use cluster centers as new features
            from sklearn.metrics.pairwise import euclidean_distances
            centers = np.array([X[cluster_labels == i].mean(axis=0) for i in range(n_clusters)])
            return centers[cluster_labels]
        else:
            return X.copy()
    
    def _apply_ensemble(self, X: np.ndarray, strategy: Dict[str, Any]) -> np.ndarray:
        """Apply ensemble approach."""
        # For now, return the original data
        # This could be enhanced with more sophisticated ensemble methods
        return X.copy()
    
    def _apply_reshaping(self, X: np.ndarray, strategy: Dict[str, Any], 
                        dataset_info: Dict[str, Any]) -> Optional[np.ndarray]:
        """Apply reshaping to the data."""
        try:
            from src.data.loader import DatasetTransformer
            
            # Create a temporary dataset transformer for reshaping functionality
            dt = DatasetTransformer()
            
            # Get target rows from strategy
            target_rows = strategy['params'].get('target_rows', 3)
            
            # Apply adaptive smart reshaping
            if strategy['method'] == 'adaptive_smart_reshape':
                # Handle 3D arrays
                if X.ndim == 3:
                    X_2d = X.reshape(X.shape[0], -1)
                else:
                    X_2d = X
                
                # Apply adaptive smart reshaping
                reshaped_X, feature_groups, col_indices, best_strategy, evaluation_metrics = dt.adaptive_smart_reshape(
                    X_2d, target_rows=target_rows
                )
                
                # Apply transpose if forced
                if strategy['params'].get('force_transpose', False):
                    if reshaped_X.shape[1] > reshaped_X.shape[2]:
                        reshaped_X = np.transpose(reshaped_X, (0, 2, 1))
                
                return reshaped_X
            
            return None
            
        except Exception as e:
            logger.warning(f"Reshaping failed: {e}")
            return None
    
    def _evaluate_clustering(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate clustering performance."""
        try:
            # Reshape to 2D for clustering
            if X.ndim == 3:
                X_2d = X.reshape(X.shape[0], -1)
            else:
                X_2d = X
            
            # Determine number of clusters
            n_clusters = min(len(np.unique(y)), 10, X_2d.shape[0] // 10)
            n_clusters = max(n_clusters, 2)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(X_2d)
            
            # Calculate metrics
            silhouette = silhouette_score(X_2d, cluster_labels)
            calinski_harabasz = calinski_harabasz_score(X_2d, cluster_labels)
            davies_bouldin = davies_bouldin_score(X_2d, cluster_labels)
            inertia = kmeans.inertia_
            
            # Calculate combined score (higher is better)
            combined_score = (
                silhouette * 0.4 +
                (calinski_harabasz / 1000) * 0.3 +
                (1 / (1 + davies_bouldin)) * 0.2 +
                (1 / (1 + inertia / 1000)) * 0.1
            )
            
            return {
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski_harabasz,
                'davies_bouldin_score': davies_bouldin,
                'inertia': inertia,
                'combined_score': combined_score,
                'n_clusters': n_clusters
            }
            
        except Exception as e:
            logger.warning(f"Clustering evaluation failed: {e}")
            return {
                'silhouette_score': 0.0,
                'calinski_harabasz_score': 0.0,
                'davies_bouldin_score': 1.0,
                'inertia': float('inf'),
                'combined_score': 0.0,
                'n_clusters': 0
            }
    
    def _create_optimization_summary(self) -> Dict[str, Any]:
        """Create a summary of the optimization results."""
        if not self.results:
            return {}
        
        # Sort results by clustering score
        sorted_results = sorted(self.results.items(), 
                              key=lambda x: x[1]['clustering_score'], 
                              reverse=True)
        
        # Calculate statistics
        scores = [result['clustering_score'] for result in self.results.values()]
        
        summary = {
            'total_strategies': len(self.results),
            'best_strategy': sorted_results[0][0] if sorted_results else None,
            'best_score': max(scores) if scores else 0.0,
            'worst_score': min(scores) if scores else 0.0,
            'average_score': np.mean(scores) if scores else 0.0,
            'std_score': np.std(scores) if scores else 0.0,
            'strategy_ranking': [name for name, _ in sorted_results],
            'score_distribution': {
                'excellent': len([s for s in scores if s > 0.7]),
                'good': len([s for s in scores if 0.5 < s <= 0.7]),
                'fair': len([s for s in scores if 0.3 < s <= 0.5]),
                'poor': len([s for s in scores if s <= 0.3])
            }
        }
        
        return summary
    
    def save_results(self, results: Dict[str, Any], output_dir: str = './optimization_results'):
        """Save optimization results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        dataset_name = results['dataset_name']
        
        # Save detailed results
        results_file = os.path.join(output_dir, f'{dataset_name}_optimization_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary
        summary_file = os.path.join(output_dir, f'{dataset_name}_optimization_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(results['optimization_summary'], f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")
    
    def generate_pbp_features(self, best_result: Dict[str, Any]) -> np.ndarray:
        """
        Generate PBP features from the best optimization result.
        
        Args:
            best_result: The best optimization result
            
        Returns:
            PBP features array
        """
        try:
            from src.pbp.core import pbp_vector
            
            if best_result is None or 'reshaped_data' not in best_result:
                logger.error("No reshaped data available for PBP feature generation")
                return None
            
            X_reshaped = best_result['reshaped_data']
            pbp_features = []
            
            for i in range(X_reshaped.shape[0]):
                sample_matrix = X_reshaped[i]
                pbp_vector_result = pbp_vector(sample_matrix)
                pbp_features.append(pbp_vector_result)
            
            return np.array(pbp_features)
            
        except Exception as e:
            logger.error(f"Failed to generate PBP features: {e}")
            return None


def optimize_all_datasets(datasets: Dict[str, Any], output_dir: str = './optimization_results') -> Dict[str, Any]:
    """
    Optimize all datasets using comprehensive optimization.
    
    Args:
        datasets: Dictionary of datasets to optimize
        output_dir: Output directory for results
        
    Returns:
        Dictionary containing optimization results for all datasets
    """
    optimizer = ComprehensiveOptimizer()
    all_results = {}
    
    for dataset_name, dataset_data in datasets.items():
        logger.info(f"Optimizing dataset: {dataset_name}")
        
        X = dataset_data['X']
        y = dataset_data['y']
        dataset_info = dataset_data.get('info', {})
        
        # Optimize dataset
        result = optimizer.optimize_dataset(dataset_name, X, y, dataset_info)
        
        # Generate PBP features from best result
        if result['best_result'] is not None:
            pbp_features = optimizer.generate_pbp_features(result['best_result'])
            result['pbp_features'] = pbp_features
            result['pbp_features_shape'] = pbp_features.shape if pbp_features is not None else None
        
        all_results[dataset_name] = result
        
        # Save results
        optimizer.save_results(result, output_dir)
    
    return all_results


if __name__ == "__main__":
    # Example usage
    from src.data.consolidated_loader import ConsolidatedDatasetLoader
    
    # Load datasets
    loader = ConsolidatedDatasetLoader()
    datasets = {}
    
    # Load a few datasets for testing
    test_datasets = ['seeds', 'thyroid', 'iris']
    
    for dataset_name in test_datasets:
        try:
            dataset_data = loader.load_dataset(dataset_name)
            if dataset_data is not None:
                datasets[dataset_name] = {
                    'X': dataset_data['X'],
                    'y': dataset_data['y'],
                    'info': dataset_data
                }
        except Exception as e:
            logger.warning(f"Failed to load dataset {dataset_name}: {e}")
    
    # Optimize all datasets
    results = optimize_all_datasets(datasets)
    
    # Print summary
    print("\n" + "="*60)
    print("COMPREHENSIVE OPTIMIZATION RESULTS")
    print("="*60)
    
    for dataset_name, result in results.items():
        print(f"\n{dataset_name.upper()}:")
        print(f"  Best Strategy: {result['optimization_summary']['best_strategy']}")
        print(f"  Best Score: {result['optimization_summary']['best_score']:.4f}")
        print(f"  Strategies Tested: {result['optimization_summary']['total_strategies']}")
        if result.get('pbp_features_shape'):
            print(f"  PBP Features Shape: {result['pbp_features_shape']}") 