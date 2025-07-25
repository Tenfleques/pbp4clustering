#!/usr/bin/env python3
"""
Base Dataset Loader for Pseudo-Boolean Polynomial Dimensionality Reduction

This module provides the base class for all dataset loaders with common functionality
and interfaces for loading, preprocessing, and transforming datasets into matrix format.
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger().setLevel(logging.ERROR)

class BaseDatasetLoader(ABC):
    """
    Abstract base class for all dataset loaders.
    
    This class provides common functionality for:
    - Loading datasets from various sources
    - Preprocessing and cleaning data
    - Transforming data into matrix format
    - Saving and loading processed datasets
    - Adaptive smart reshaping
    """
    
    def __init__(self, data_dir='./data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.datasets = {}
        self.transformers = {}
        
    @abstractmethod
    def load_dataset(self, dataset_name):
        """
        Load a specific dataset.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            dict: Dictionary containing X, y, metadata
        """
        pass
    
    def adaptive_smart_reshape(self, X, target_rows=2, max_combinations=1000):
        """
        Adaptive smart reshaping that tries multiple strategies and selects the best one.
        
        This method automatically tries different rearrangement strategies and selects
        the optimal approach based on comprehensive evaluation metrics.
        
        Args:
            X: Input feature matrix (n_samples, n_features)
            target_rows: Number of rows in the output matrix (2 or 3)
            max_combinations: Maximum number of combinations to evaluate
            
        Returns:
            Tuple of (reshaped_X, feature_groups, column_indices, best_strategy, evaluation_metrics)
        """
        print(f"Adaptive smart reshaping: trying multiple strategies for {X.shape[1]} features...")
        
        strategies = {
            'homogeneity_only': {
                'balance_weights': {'homogeneity': 1.0, 'importance': 0.0, 'diversity': 0.0, 'balance': 0.0},
                'use_feature_importance': False,
                'description': 'Traditional homogeneity-only approach'
            },
            'importance_focused': {
                'balance_weights': {'homogeneity': 0.2, 'importance': 0.6, 'diversity': 0.1, 'balance': 0.1},
                'use_feature_importance': True,
                'description': 'Feature importance-focused approach'
            },
            'diversity_focused': {
                'balance_weights': {'homogeneity': 0.2, 'importance': 0.2, 'diversity': 0.5, 'balance': 0.1},
                'use_feature_importance': True,
                'description': 'Feature diversity-focused approach'
            },
            'balanced': {
                'balance_weights': {'homogeneity': 0.35, 'importance': 0.35, 'diversity': 0.2, 'balance': 0.1},
                'use_feature_importance': True,
                'description': 'Balanced multi-objective approach'
            },
            'homogeneity_focused': {
                'balance_weights': {'homogeneity': 0.6, 'importance': 0.2, 'diversity': 0.1, 'balance': 0.1},
                'use_feature_importance': True,
                'description': 'Homogeneity-focused approach'
            },
            'transpose_optimized': {
                'balance_weights': {'homogeneity': 0.4, 'importance': 0.3, 'diversity': 0.2, 'balance': 0.1},
                'use_feature_importance': True,
                'description': 'Transpose-optimized approach (ensure rows < columns)'
            }
        }
        
        best_result = None
        best_score = float('inf')
        best_strategy = None
        all_evaluation_metrics = {}
        
        # Handle NaN values
        X_clean = X.copy()
        if np.isnan(X_clean).any():
            X_clean = np.nan_to_num(X_clean, nan=0.0)
        
        for strategy_name, strategy_config in strategies.items():
            try:
                print(f"  Testing {strategy_name}...")
                
                # Apply smart reshaping with current strategy
                result = self.smart_reshape_with_homogeneity(
                    X_clean, target_rows, max_combinations,
                    strategy_config['balance_weights'],
                    strategy_config['use_feature_importance']
                )
                
                if result is None:
                    continue
                    
                reshaped_X, feature_groups, column_indices = result
                
                # Apply transpose optimization if this is the transpose strategy
                if strategy_name == 'transpose_optimized':
                    if reshaped_X.shape[1] > reshaped_X.shape[2]:  # rows > columns
                        reshaped_X = reshaped_X.transpose(0, 2, 1)  # Transpose the matrix
                        feature_groups = self._transpose_feature_groups(
                            feature_groups, reshaped_X.shape[1], reshaped_X.shape[2]
                        )
                
                # Evaluate the strategy
                evaluation_metrics = self._evaluate_reshaping_strategy(
                    reshaped_X, feature_groups, {}, strategy_config
                )
                
                all_evaluation_metrics[strategy_name] = evaluation_metrics
                
                # Update best result
                if evaluation_metrics['combined_score'] < best_score:
                    best_score = evaluation_metrics['combined_score']
                    best_result = (reshaped_X, feature_groups, column_indices)
                    best_strategy = strategy_name
                    
            except Exception as e:
                print(f"    Strategy {strategy_name} failed: {e}")
                continue
        
        if best_result is None:
            print("  All strategies failed, using fallback reshaping...")
            best_result = self._fallback_reshaping(X_clean, target_rows)
            best_strategy = 'fallback'
            all_evaluation_metrics['fallback'] = {'combined_score': float('inf')}
        
        reshaped_X, feature_groups, column_indices = best_result
        
        print(f"  Best strategy: {best_strategy}")
        print(f"  Final shape: {reshaped_X.shape}")
        
        return reshaped_X, feature_groups, column_indices, best_strategy, all_evaluation_metrics
    
    def smart_reshape_with_homogeneity(self, X, target_rows=2, max_combinations=1000, 
                                     balance_weights=None, use_feature_importance=True):
        """
        Smart reshaping that groups features to minimize within-column variance.
        
        Args:
            X: Input feature matrix
            target_rows: Number of rows in output matrix
            max_combinations: Maximum combinations to evaluate
            balance_weights: Weights for different objectives
            use_feature_importance: Whether to use feature importance
            
        Returns:
            Tuple of (reshaped_X, feature_groups, column_indices)
        """
        if balance_weights is None:
            balance_weights = {'homogeneity': 0.4, 'importance': 0.3, 'diversity': 0.2, 'balance': 0.1}
        
        n_features = X.shape[1]
        n_samples = X.shape[0]
        
        # Calculate feature importance if requested
        feature_importance = None
        if use_feature_importance:
            feature_importance = self._calculate_feature_importance(X)
        
        # Calculate feature variances and correlations
        feature_vars = np.var(X, axis=0)
        feature_corrs = np.corrcoef(X.T)
        
        # Create enhanced similarity matrix
        similarity_matrix = self._create_enhanced_similarity_matrix(
            X, feature_vars, feature_corrs, feature_importance, balance_weights
        )
        
        # Find best grouping using K-means
        best_grouping = None
        best_score = float('inf')
        
        for n_clusters in range(2, min(target_rows + 3, n_features)):
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(similarity_matrix)
                
                # Create feature groups from clusters
                feature_groups = []
                for i in range(n_clusters):
                    group = np.where(cluster_labels == i)[0]
                    if len(group) > 0:
                        feature_groups.append(group)
                
                # Ensure we have exactly target_rows groups
                if len(feature_groups) == target_rows:
                    score = self._evaluate_multi_objective_grouping(
                        X, feature_groups, target_rows, feature_importance, balance_weights
                    )
                    
                    if score < best_score:
                        best_score = score
                        best_grouping = feature_groups
                        
            except Exception as e:
                continue
        
        if best_grouping is None:
            # Fallback: create equal-sized groups
            features_per_group = n_features // target_rows
            best_grouping = []
            for i in range(target_rows):
                start_idx = i * features_per_group
                end_idx = start_idx + features_per_group if i < target_rows - 1 else n_features
                best_grouping.append(np.arange(start_idx, end_idx))
        
        # Ensure we have exactly target_rows groups
        if len(best_grouping) < target_rows:
            # Pad with empty groups
            while len(best_grouping) < target_rows:
                best_grouping.append(np.array([]))
        elif len(best_grouping) > target_rows:
            # Truncate to target_rows
            best_grouping = best_grouping[:target_rows]
        
        # Create reshaped matrix
        max_features_per_group = max(len(group) for group in best_grouping)
        reshaped_X = np.zeros((n_samples, target_rows, max_features_per_group))
        
        for i, group in enumerate(best_grouping):
            if len(group) > 0:
                reshaped_X[:, i, :len(group)] = X[:, group]
        
        return reshaped_X, best_grouping, np.arange(n_features)
    
    def _calculate_feature_importance(self, X):
        """Calculate feature importance using multiple metrics."""
        try:
            from sklearn.feature_selection import mutual_info_regression, f_regression
            
            # Calculate variance
            variance_importance = np.var(X, axis=0)
            
            # Calculate mutual information (use dummy target for now)
            y_dummy = np.random.randint(0, 2, size=X.shape[0])
            mi_importance = mutual_info_regression(X, y_dummy, random_state=42)
            
            # Calculate F-scores
            f_scores, _ = f_regression(X, y_dummy)
            
            # Combine importance scores
            combined_importance = (variance_importance + mi_importance + f_scores) / 3
            return combined_importance
            
        except Exception as e:
            print(f"Warning: Could not calculate feature importance: {e}")
            return np.ones(X.shape[1])
    
    def _create_enhanced_similarity_matrix(self, X, feature_vars, feature_corrs, feature_importance, balance_weights):
        """Create enhanced similarity matrix for feature grouping."""
        n_features = X.shape[1]
        similarity_matrix = np.zeros((n_features, n_features))
        
        for i in range(n_features):
            for j in range(n_features):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # Variance similarity
                    var_sim = 1.0 / (1.0 + abs(feature_vars[i] - feature_vars[j]))
                    
                    # Correlation similarity
                    corr_sim = abs(feature_corrs[i, j])
                    
                    # Importance similarity
                    imp_sim = 1.0
                    if feature_importance is not None:
                        imp_sim = 1.0 / (1.0 + abs(feature_importance[i] - feature_importance[j]))
                    
                    # Combine similarities
                    similarity_matrix[i, j] = (
                        balance_weights['homogeneity'] * var_sim +
                        balance_weights['importance'] * imp_sim +
                        balance_weights['diversity'] * corr_sim
                    )
        
        return similarity_matrix
    
    def _evaluate_multi_objective_grouping(self, X, feature_groups, target_rows, feature_importance, balance_weights):
        """Evaluate grouping quality using multiple objectives."""
        homogeneity_score = self._evaluate_grouping(X, feature_groups, target_rows)
        importance_score = self._evaluate_importance_distribution(feature_groups, feature_importance)
        diversity_score = self._evaluate_feature_diversity(X, feature_groups)
        balance_score = self._evaluate_group_balance(feature_groups, target_rows)
        
        combined_score = (
            balance_weights['homogeneity'] * homogeneity_score +
            balance_weights['importance'] * importance_score +
            balance_weights['diversity'] * diversity_score +
            balance_weights['balance'] * balance_score
        )
        
        return combined_score
    
    def _evaluate_reshaping_strategy(self, reshaped_X, feature_groups, optimization_metrics, strategy_config):
        """Evaluate the quality of a reshaping strategy."""
        # Calculate basic metrics
        homogeneity_score = self._evaluate_grouping(reshaped_X.reshape(reshaped_X.shape[0], -1), 
                                                 feature_groups, reshaped_X.shape[1])
        
        # Calculate transpose bonus/penalty
        rows, cols = reshaped_X.shape[1], reshaped_X.shape[2]
        if rows < cols:
            transpose_bonus = -0.1  # Bonus for optimal row/column ratio
        elif rows > cols:
            transpose_bonus = 0.1   # Penalty for suboptimal ratio
        else:
            transpose_bonus = 0.0
        
        # Calculate diversity and balance scores
        diversity_score = self._evaluate_feature_diversity(reshaped_X.reshape(reshaped_X.shape[0], -1), 
                                                        feature_groups)
        balance_score = self._evaluate_group_balance(feature_groups, reshaped_X.shape[1])
        
        # Combine scores
        combined_score = homogeneity_score + diversity_score + balance_score + transpose_bonus
        
        return {
            'homogeneity_score': homogeneity_score,
            'diversity_score': diversity_score,
            'balance_score': balance_score,
            'transpose_bonus': transpose_bonus,
            'combined_score': combined_score,
            'strategy_description': strategy_config.get('description', 'Unknown')
        }
    
    def _transpose_feature_groups(self, feature_groups, new_rows, new_cols):
        """Transpose feature groups when matrix is transposed."""
        total_features = sum(len(group) for group in feature_groups)
        new_feature_groups = []
        for i in range(new_rows):
            start_idx = i * (total_features // new_rows)
            end_idx = (i + 1) * (total_features // new_rows) if i < new_rows - 1 else total_features
            new_feature_groups.append(np.arange(start_idx, end_idx))
        return new_feature_groups
    
    def _fallback_reshaping(self, X, target_rows):
        """Fallback reshaping when all strategies fail."""
        print("  Using fallback reshaping...")
        
        # Handle NaN values
        if np.isnan(X).any():
            X = np.nan_to_num(X, nan=0.0)
        
        n_samples, n_features = X.shape
        
        # Simple reshaping: divide features equally
        features_per_row = n_features // target_rows
        remaining_features = n_features % target_rows
        
        reshaped_X = np.zeros((n_samples, target_rows, features_per_row + (1 if remaining_features > 0 else 0)))
        
        feature_idx = 0
        for row in range(target_rows):
            features_in_row = features_per_row + (1 if row < remaining_features else 0)
            reshaped_X[:, row, :features_in_row] = X[:, feature_idx:feature_idx + features_in_row]
            feature_idx += features_in_row
        
        # Create simple feature groups
        feature_groups = []
        feature_idx = 0
        for row in range(target_rows):
            features_in_row = features_per_row + (1 if row < remaining_features else 0)
            feature_groups.append(np.arange(feature_idx, feature_idx + features_in_row))
            feature_idx += features_in_row
        
        return reshaped_X, feature_groups, np.arange(n_features)
    
    def _evaluate_grouping(self, X, feature_groups, target_rows):
        """Evaluate grouping quality based on within-group homogeneity."""
        if not feature_groups:
            return float('inf')
        
        total_variance = 0
        for group in feature_groups:
            if len(group) > 0:
                group_data = X[:, group]
                group_variance = np.var(group_data)
                total_variance += group_variance
        
        return total_variance / len(feature_groups)
    
    def _evaluate_importance_distribution(self, feature_groups, feature_importance):
        """Evaluate how well feature importance is distributed across groups."""
        if feature_importance is None:
            return 0.0
        
        group_importances = []
        for group in feature_groups:
            if len(group) > 0:
                group_importance = np.mean(feature_importance[group])
                group_importances.append(group_importance)
        
        if not group_importances:
            return float('inf')
        
        # Penalize uneven distribution
        return np.std(group_importances)
    
    def _evaluate_feature_diversity(self, X, feature_groups):
        """Evaluate feature diversity within groups."""
        diversity_score = 0
        for group in feature_groups:
            if len(group) > 1:
                group_data = X[:, group]
                # Calculate average pairwise correlation
                corr_matrix = np.corrcoef(group_data.T)
                # Remove diagonal
                corr_values = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
                avg_corr = np.mean(np.abs(corr_values))
                diversity_score += avg_corr
        
        return diversity_score / max(len(feature_groups), 1)
    
    def _evaluate_group_balance(self, feature_groups, target_rows):
        """Evaluate balance of feature distribution across groups."""
        group_sizes = [len(group) for group in feature_groups]
        if not group_sizes:
            return float('inf')
        
        # Penalize uneven group sizes
        return np.std(group_sizes)
    
    def save_dataset(self, dataset_name, dataset, output_dir=None):
        """Save processed dataset to files."""
        if output_dir is None:
            output_dir = self.data_dir
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save X and y as numpy arrays
            np.save(output_dir / f"{dataset_name}_X.npy", dataset['X'])
            np.save(output_dir / f"{dataset_name}_y.npy", dataset['y'])
            
            # Save metadata as JSON
            metadata = {
                'description': dataset.get('description', ''),
                'feature_names': dataset.get('feature_names', []),
                'measurement_names': dataset.get('measurement_names', []),
                'target_names': dataset.get('target_names', []),
                'data_type': dataset.get('data_type', 'unknown'),
                'domain': dataset.get('domain', 'unknown'),
                'sample_count': dataset['X'].shape[0],
                'preprocessing': dataset.get('preprocessing', ''),
                'original_shape': list(dataset.get('original_shape', dataset['X'].shape)),
                'matrix_shape': list(dataset.get('matrix_shape', (dataset['X'].shape[1], dataset['X'].shape[2]))),
                'shape': list(dataset['X'].shape),
                'n_classes': int(len(np.unique(dataset['y']))),
                'source': dataset.get('source', 'unknown')
            }
            
            with open(output_dir / f"{dataset_name}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"  → Saved {dataset_name} to {output_dir}")
            
        except Exception as e:
            print(f"Error saving dataset {dataset_name}: {e}")
    
    def load_saved_dataset(self, dataset_name, data_dir=None):
        """Load a previously saved dataset."""
        if data_dir is None:
            data_dir = self.data_dir
        
        data_dir = Path(data_dir)
        
        try:
            x_file = data_dir / f"{dataset_name}_X.npy"
            y_file = data_dir / f"{dataset_name}_y.npy"
            metadata_file = data_dir / f"{dataset_name}_metadata.json"
            
            if not all(f.exists() for f in [x_file, y_file, metadata_file]):
                return None
            
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
                'preprocessing': metadata.get('preprocessing', ''),
                'original_shape': tuple(metadata.get('original_shape', X.shape)),
                'matrix_shape': tuple(metadata.get('matrix_shape', (X.shape[1], X.shape[2]))),
                'source': metadata.get('source', 'unknown')
            }
            
            return dataset
            
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            return None
    
    def get_dataset_info(self, dataset_name):
        """Get information about a dataset."""
        dataset = self.load_saved_dataset(dataset_name)
        if dataset is None:
            return None
        
        return {
            'name': dataset_name,
            'shape': dataset['X'].shape,
            'n_classes': len(np.unique(dataset['y'])),
            'data_type': dataset.get('data_type', 'unknown'),
            'description': dataset.get('description', ''),
            'source': dataset.get('source', 'unknown')
        } 