#!/usr/bin/env python3
"""
Test Comprehensive Optimization

This script tests the comprehensive optimization approach with a few datasets.
"""

import sys
import os
import numpy as np
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data.loader import DatasetTransformer
from src.analysis.comprehensive_optimization import ComprehensiveOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_comprehensive_optimization():
    """Test comprehensive optimization with a few datasets."""
    
    # Load datasets
    dt = DatasetTransformer()
    test_datasets = ['iris', 'wine', 'seeds']
    
    datasets = {}
    
    for dataset_name in test_datasets:
        try:
            logger.info(f"Loading dataset: {dataset_name}")
            dataset_data = dt.load_dataset(dataset_name)
            
            if dataset_data is not None:
                X = dataset_data['X']
                y = dataset_data['y']
                
                # Handle different data formats
                if X.ndim == 3:
                    X_2d = X.reshape(X.shape[0], -1)
                else:
                    X_2d = X
                
                # Handle NaN values
                if np.isnan(X_2d).any():
                    logger.info(f"Handling NaN values in {dataset_name}")
                    X_2d = np.nan_to_num(X_2d, nan=0.0)
                
                datasets[dataset_name] = {
                    'X': X_2d,
                    'y': y,
                    'info': dataset_data,
                    'original_shape': X.shape,
                    'processed_shape': X_2d.shape
                }
                
                logger.info(f"Successfully loaded {dataset_name}: {X_2d.shape}")
            else:
                logger.warning(f"Failed to load dataset: {dataset_name}")
                
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            continue
    
    if not datasets:
        logger.error("No datasets loaded. Exiting.")
        return
    
    # Test comprehensive optimization
    logger.info("Testing comprehensive optimization...")
    
    optimizer = ComprehensiveOptimizer(max_strategies=3)  # Limit for testing
    
    for dataset_name, dataset_data in datasets.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing optimization for: {dataset_name}")
        logger.info(f"{'='*50}")
        
        X = dataset_data['X']
        y = dataset_data['y']
        dataset_info = dataset_data['info']
        
        # Optimize dataset
        result = optimizer.optimize_dataset(dataset_name, X, y, dataset_info)
        
        if result['best_result'] is not None:
            logger.info(f"Best strategy: {result['best_result']['strategy_name']}")
            logger.info(f"Best score: {result['best_result']['clustering_score']:.4f}")
            logger.info(f"Reshaped shape: {result['best_result']['reshaped_shape']}")
            
            # Generate PBP features
            pbp_features = optimizer.generate_pbp_features(result['best_result'])
            if pbp_features is not None:
                logger.info(f"PBP features shape: {pbp_features.shape}")
            else:
                logger.warning("Failed to generate PBP features")
        else:
            logger.warning(f"No successful optimization for {dataset_name}")
    
    logger.info("\nComprehensive optimization test completed!")

if __name__ == "__main__":
    test_comprehensive_optimization() 