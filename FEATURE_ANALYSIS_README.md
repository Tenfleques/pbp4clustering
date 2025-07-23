# PBP Feature Analysis

This module provides comprehensive analysis of PBP (Permutation-Based Polynomial) features to identify the most significant columns for clustering based on target variables.

## Overview

The feature analysis script analyzes cached PBP features (`dataset_name_pbp_features.npy`) and corresponding target data (`dataset_name_y.npy`) to determine which features have the most significant impact on clustering performance.

## Features

The analysis provides multiple metrics to evaluate feature importance:

1. **Standard Deviation**: Measures the spread of values in each feature
2. **Correlation**: Measures linear correlation between features and targets
3. **Mutual Information**: Measures non-linear relationships between features and targets
4. **F-Statistic**: Measures statistical significance of feature-target relationships
5. **Clustering Impact**: Measures how removing each feature affects clustering quality
6. **Combined Score**: Average of all normalized metrics

### Feature Extraction and Visualization

- **Top-k Feature Extraction**: Extracts the most important features (max 3)
- **Constant Feature Filtering**: Automatically removes features with zero variance
- **Adaptive Visualization**: Creates 1D, 2D, or 3D plots based on feature count
- **Interactive Plots**: Color-coded by target classes with legends
- **Export Capability**: Save plots as high-resolution PNG files

## Usage

### Command Line Interface

```bash
# Basic usage
python3 feature_analysis.py <dataset_name>

# With custom parameters
python3 feature_analysis.py wine --top-k 10 --save

# Save to specific file
python3 feature_analysis.py iris --output my_results.json

# Extract and visualize top features
python3 feature_analysis.py wine --k 3 --visualize

# Save visualization plot
python3 feature_analysis.py iris --k 2 --visualize --save-plot

# Custom plot output file
python3 feature_analysis.py yeast --k 1 --visualize --save-plot --plot-output my_plot.png
```

### Parameters

- `dataset_name`: Name of the dataset to analyze (e.g., 'iris', 'wine', 'yeast')
- `--top-k`: Number of top features to return (default: 10)
- `--k`: Maximum number of features to extract for visualization (default: 3)
- `--data-dir`: Directory containing data files (default: 'data')
- `--output`: Output JSON file path (optional)
- `--save`: Save results to JSON file
- `--visualize`: Create visualization of top features
- `--save-plot`: Save visualization plot
- `--plot-output`: Output file path for the plot (optional)

### Programmatic Usage

```python
from feature_analysis import analyze_features, print_analysis_results

# Analyze a dataset
results = analyze_features('wine', top_k=5)

# Print formatted results
print_analysis_results(results, top_k=5)

# Access specific metrics
top_features = results['top_features']['combined']['indices']
combined_scores = results['top_features']['combined']['scores']

# Extract and visualize top features
from feature_analysis import extract_top_features, visualize_features, load_dataset_data

features, targets = load_dataset_data('wine')
extracted_features, selected_indices = extract_top_features(features, results, k=3)
visualize_features(extracted_features, targets, selected_indices, 'wine', save_plot=True)

## Output Format

The script provides detailed analysis including:

- **Dataset Information**: Feature count, sample count, target classes
- **Metric Rankings**: Top features for each analysis metric
- **Combined Importance**: Overall ranking based on all metrics
- **JSON Export**: Complete results saved to JSON file

### Example Output

```
============================================================
FEATURE ANALYSIS RESULTS FOR: WINE
============================================================
Dataset Info:
  - Features: 7
  - Samples: 356
  - Target classes: 3

Top 5 Features by Metric:
============================================================

STD:
Index    Score        Value          
-----------------------------------
6        1.0000       426.9698       
5        0.1045       44.6308        
4        0.0145       6.1709         

CORRELATION:
Index    Score        Value          
-----------------------------------
1        1.0000       0.3187         
6        0.7207       0.2297         
3        0.4151       0.1323         

COMBINED IMPORTANCE RANKING:
Index    Combined Score 
-------------------------
6        0.4796         
1        0.2386         
3        0.2286         
```

## Interpretation

### Metric Meanings

- **Standard Deviation**: Higher values indicate more variable features
- **Correlation**: Higher values indicate stronger linear relationships with targets
- **Mutual Information**: Higher values indicate stronger non-linear relationships
- **F-Statistic**: Higher values indicate more statistically significant relationships
- **Clustering Impact**: Positive values indicate features that improve clustering
- **Combined Score**: Overall importance across all metrics

### Key Insights

1. **Feature 6** in the wine dataset has the highest combined score (0.4796)
2. **Feature 1** shows strong correlation with targets (0.3187)
3. **Feature 3** has high F-statistic (26.88), indicating statistical significance
4. **Feature 4** has positive clustering impact, suggesting it improves clustering quality

## Available Datasets

The script works only with datasets that have **natural combinatorial relationships**:

### Valid Datasets for PBP Analysis:
- **iris**: Natural 2×2 structure (Sepal/Petal × Length/Width)
- **breast_cancer**: Natural 3×10 structure (Statistical Measures × Cell Features)

### Invalid Datasets (Removed):
- **wine**: Artificial chemical grouping
- **yeast**: No natural combinatorial relationships
- **Other datasets**: Need individual analysis for natural relationships

**Note**: PBP analysis should only be applied to datasets with meaningful matrix structures that reflect real relationships, not arbitrary groupings.

## Error Handling

The script includes robust error handling for:
- Missing data files
- Dimension mismatches between features and targets
- Constant features (zero variance)
- Clustering failures
- Invalid input parameters

## Dependencies

- numpy
- pandas
- scikit-learn
- matplotlib (optional, for future visualizations)
- seaborn (optional, for future visualizations)

## Future Enhancements

- Statistical significance testing
- Feature selection recommendations
- Integration with clustering algorithms
- Export to various formats (CSV, Excel)
- Additional visualization types (heatmaps, correlation matrices)
- Interactive 3D plots with rotation controls 