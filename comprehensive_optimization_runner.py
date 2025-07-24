#!/usr/bin/env python3
"""
Comprehensive Optimization Runner

This script runs comprehensive optimization on all datasets to get the best cluster scores
before saving PBP features. It tests all adaptations and selects the optimal approach.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data.loader import DatasetTransformer
from src.analysis.comprehensive_optimization import ComprehensiveOptimizer, optimize_all_datasets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_all_datasets_for_optimization():
    """
    Load all available datasets for comprehensive optimization.
    
    Returns:
        Dictionary of datasets ready for optimization
    """
    logger.info("Loading all datasets for comprehensive optimization...")
    
    dt = DatasetTransformer()
    datasets = {}
    
    # List of datasets to optimize
    dataset_names = [
        'iris', 'breast_cancer', 'wine', 'digits', 'diabetes', 'sonar',
        'glass', 'vehicle', 'ecoli', 'yeast', 'seeds', 'thyroid', 
        'ionosphere', 'covertype', 'kddcup99'
    ]
    
    for dataset_name in dataset_names:
        try:
            logger.info(f"Loading dataset: {dataset_name}")
            dataset_data = dt.load_dataset(dataset_name)
            
            if dataset_data is not None:
                # Prepare data for optimization
                X = dataset_data['X']
                y = dataset_data['y']
                
                # Handle different data formats
                if X.ndim == 3:
                    # Flatten 3D arrays for optimization
                    X_2d = X.reshape(X.shape[0], -1)
                else:
                    X_2d = X
                
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
    
    logger.info(f"Loaded {len(datasets)} datasets for optimization")
    return datasets

def run_comprehensive_optimization():
    """
    Run comprehensive optimization on all datasets.
    """
    logger.info("Starting comprehensive optimization process...")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'./results/optimization_results_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load datasets
    datasets = load_all_datasets_for_optimization()
    
    if not datasets:
        logger.error("No datasets loaded. Exiting.")
        return
    
    # Run comprehensive optimization
    logger.info("Running comprehensive optimization...")
    results = optimize_all_datasets(datasets, output_dir)
    
    # Generate comprehensive report
    generate_optimization_report(results, output_dir)
    
    # Save PBP features for best results
    save_best_pbp_features(results, output_dir)
    
    logger.info(f"Comprehensive optimization completed. Results saved to: {output_dir}")

def generate_optimization_report(results, output_dir):
    """
    Generate a comprehensive optimization report.
    
    Args:
        results: Optimization results for all datasets
        output_dir: Output directory
    """
    logger.info("Generating comprehensive optimization report...")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_datasets': len(results),
        'datasets_processed': list(results.keys()),
        'summary': {},
        'detailed_results': {}
    }
    
    # Calculate overall statistics
    all_scores = []
    strategy_counts = {}
    improvement_stats = []
    
    for dataset_name, result in results.items():
        if result['best_result'] is not None:
            # Collect scores
            all_scores.append(result['best_score'])
            
            # Count strategies
            best_strategy = result['optimization_summary']['best_strategy']
            strategy_counts[best_strategy] = strategy_counts.get(best_strategy, 0) + 1
            
            # Calculate improvement
            if result['all_results']:
                scores = [r['clustering_score'] for r in result['all_results'].values()]
                improvement = (max(scores) - min(scores)) / min(scores) * 100 if min(scores) > 0 else 0
                improvement_stats.append(improvement)
            
            # Add detailed results
            report['detailed_results'][dataset_name] = {
                'best_strategy': best_strategy,
                'best_score': result['best_score'],
                'total_strategies_tested': result['optimization_summary']['total_strategies'],
                'score_distribution': result['optimization_summary']['score_distribution'],
                'reshaped_shape': result['best_result']['reshaped_shape'],
                'pbp_features_shape': result.get('pbp_features_shape')
            }
    
    # Overall statistics
    if all_scores:
        report['summary'] = {
            'average_best_score': np.mean(all_scores),
            'max_best_score': max(all_scores),
            'min_best_score': min(all_scores),
            'std_best_score': np.std(all_scores),
            'most_common_strategy': max(strategy_counts.items(), key=lambda x: x[1])[0] if strategy_counts else None,
            'strategy_distribution': strategy_counts,
            'average_improvement': np.mean(improvement_stats) if improvement_stats else 0,
            'datasets_with_improvement': len([s for s in improvement_stats if s > 0])
        }
    
    # Save report
    report_file = os.path.join(output_dir, 'comprehensive_optimization_report.json')
    with open(report_file, 'w') as f:
        import json
        json.dump(report, f, indent=2, default=str)
    
    # Generate markdown report
    generate_markdown_report(report, output_dir)
    
    logger.info(f"Optimization report saved to: {report_file}")

def generate_markdown_report(report, output_dir):
    """
    Generate a markdown report for the optimization results.
    
    Args:
        report: Optimization report
        output_dir: Output directory
    """
    markdown_content = f"""# Comprehensive Optimization Report

**Generated**: {report['timestamp']}
**Total Datasets**: {report['total_datasets']}
**Datasets Processed**: {', '.join(report['datasets_processed'])}

## Overall Statistics

- **Average Best Score**: {report['summary'].get('average_best_score', 0):.4f}
- **Maximum Best Score**: {report['summary'].get('max_best_score', 0):.4f}
- **Minimum Best Score**: {report['summary'].get('min_best_score', 0):.4f}
- **Standard Deviation**: {report['summary'].get('std_best_score', 0):.4f}
- **Most Common Strategy**: {report['summary'].get('most_common_strategy', 'N/A')}
- **Average Improvement**: {report['summary'].get('average_improvement', 0):.2f}%
- **Datasets with Improvement**: {report['summary'].get('datasets_with_improvement', 0)}

## Strategy Distribution

"""
    
    for strategy, count in report['summary'].get('strategy_distribution', {}).items():
        markdown_content += f"- **{strategy}**: {count} datasets\n"
    
    markdown_content += "\n## Detailed Results\n\n"
    markdown_content += "| Dataset | Best Strategy | Best Score | Strategies Tested | Reshaped Shape | PBP Features Shape |\n"
    markdown_content += "|---------|---------------|------------|-------------------|----------------|-------------------|\n"
    
    for dataset_name, details in report['detailed_results'].items():
        markdown_content += f"| {dataset_name} | {details['best_strategy']} | {details['best_score']:.4f} | {details['total_strategies_tested']} | {details['reshaped_shape']} | {details.get('pbp_features_shape', 'N/A')} |\n"
    
    # Save markdown report
    markdown_file = os.path.join(output_dir, 'comprehensive_optimization_report.md')
    with open(markdown_file, 'w') as f:
        f.write(markdown_content)
    
    logger.info(f"Markdown report saved to: {markdown_file}")

def save_best_pbp_features(results, output_dir):
    """
    Save the best PBP features for each dataset.
    
    Args:
        results: Optimization results
        output_dir: Output directory
    """
    logger.info("Saving best PBP features...")
    
    pbp_dir = os.path.join(output_dir, 'pbp_features')
    os.makedirs(pbp_dir, exist_ok=True)
    
    saved_features = {}
    
    for dataset_name, result in results.items():
        if result.get('pbp_features') is not None:
            # Save PBP features
            pbp_file = os.path.join(pbp_dir, f'{dataset_name}_pbp_features.npy')
            np.save(pbp_file, result['pbp_features'])
            
            # Save metadata
            metadata = {
                'dataset_name': dataset_name,
                'best_strategy': result['optimization_summary']['best_strategy'],
                'best_score': result['best_score'],
                'original_shape': result['best_result']['original_shape'],
                'reshaped_shape': result['best_result']['reshaped_shape'],
                'pbp_features_shape': result['pbp_features'].shape,
                'optimization_timestamp': datetime.now().isoformat()
            }
            
            metadata_file = os.path.join(pbp_dir, f'{dataset_name}_metadata.json')
            with open(metadata_file, 'w') as f:
                import json
                json.dump(metadata, f, indent=2)
            
            saved_features[dataset_name] = {
                'pbp_features_file': pbp_file,
                'metadata_file': metadata_file,
                'shape': result['pbp_features'].shape
            }
            
            logger.info(f"Saved PBP features for {dataset_name}: {result['pbp_features'].shape}")
    
    # Save summary of saved features
    summary_file = os.path.join(pbp_dir, 'saved_features_summary.json')
    with open(summary_file, 'w') as f:
        import json
        json.dump(saved_features, f, indent=2)
    
    logger.info(f"Saved PBP features for {len(saved_features)} datasets to: {pbp_dir}")

def main():
    """
    Main function to run comprehensive optimization.
    """
    logger.info("="*60)
    logger.info("COMPREHENSIVE OPTIMIZATION RUNNER")
    logger.info("="*60)
    
    try:
        run_comprehensive_optimization()
        logger.info("Comprehensive optimization completed successfully!")
        
    except Exception as e:
        logger.error(f"Comprehensive optimization failed: {e}")
        raise

if __name__ == "__main__":
    main() 