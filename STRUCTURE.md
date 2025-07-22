# PBP Codebase Structure

This document describes the restructured PBP (Pseudo-Boolean Polynomial) codebase organization.

## Overview

## Directory Structure

```
pbp4clustering/
├── src/                          # Main source code directory
│   ├── __init__.py              # Package initialization
│   ├── pbp/                     # PBP core functionality
│   │   ├── __init__.py         # PBP module initialization
│   │   └── core.py             # Core PBP implementation
│   ├── data/                    # Data loading and transformation
│   │   ├── __init__.py         # Data module initialization
│   │   └── loader.py           # Dataset loading functionality
│   └── analysis/                # Analysis and comparison tools
│       ├── __init__.py         # Analysis module initialization
│       ├── comparison.py        # Comprehensive comparison functionality
│       └── testing.py          # Dataset testing functionality
├── pbp_runner.py               # Entry point for PBP functionality
├── dataset_runner.py           # Entry point for dataset loading
├── comparison_runner.py        # Entry point for comprehensive comparison
├── testing_runner.py           # Entry point for dataset testing
├── example_usage.py            # Updated example usage
├── run_all_analyses.py         # Updated analysis runner
└── data/                       # Data directory
    └── ...                     # Dataset files
```

## Module Descriptions

### src/pbp/core.py
Contains the core PBP implementation including:
- `pbp_vector()` - Main PBP vector generation function
- `create_pbp()` - Create polynomial basis representation
- `truncate_pBp()` - Truncate PBP by degree
- All supporting functions for PBP calculations

### src/data/loader.py
Contains dataset loading and transformation functionality:
- `DatasetTransformer` class for loading various datasets
- Support for 10+ datasets (iris, breast_cancer, wine, digits, etc.)
- Dataset reshaping to matrix format for PBP analysis
- Data normalization and preprocessing

### src/analysis/comparison.py
Contains comprehensive comparison functionality:
- `ComprehensiveComparison` class for comparing PBP vs PCA/t-SNE/UMAP
- Clustering evaluation with silhouette and Davies-Bouldin scores
- Feature selection with PBP approach

### src/analysis/testing.py
Contains dataset testing functionality:
- `DatasetTester` class for testing PBP on all datasets
- Visualization of clustering results
- Performance evaluation and reporting

## Entry Point Scripts

### pbp_runner.py
Provides access to all PBP core functionality:
```python
from pbp_runner import pbp_vector, create_pbp, truncate_pBp
```

### dataset_runner.py
Provides access to dataset loading functionality:
```python
from dataset_runner import DatasetTransformer
```

### comparison_runner.py
Runs comprehensive comparison analysis on all datasets.

### testing_runner.py
Runs dataset testing with visualization and evaluation.

## Usage Examples

### Basic PBP Usage
```python
import numpy as np
from pbp_runner import pbp_vector

# Create a sample matrix
c = np.array([[1, 2], [3, 4]])
result = pbp_vector(c)
print(result)
```

### Dataset Loading
```python
from dataset_runner import DatasetTransformer

transformer = DatasetTransformer()
iris_data = transformer.load_iris_dataset()
print(iris_data['X'].shape)
```

### Running Analysis
```python
# Run comprehensive comparison
python comparison_runner.py

# Run dataset testing
python testing_runner.py

# Run all analyses
python run_all_analyses.py
```

## Import Structure

### From src modules
```python
from src.pbp.core import pbp_vector
from src.data.loader import DatasetTransformer
from src.analysis.comparison import ComprehensiveComparison
from src.analysis.testing import DatasetTester
```

### From entry points
```python
from pbp_runner import pbp_vector
from dataset_runner import DatasetTransformer
```

## Development Guidelines

When adding new functionality:

1. **Core PBP functions**: Add to `src/pbp/core.py`
2. **Data loading**: Add to `src/data/loader.py`
3. **Analysis tools**: Add to `src/analysis/` with appropriate submodule
4. **Entry points**: Create new runner script in root directory
5. **Documentation**: Update this file and relevant docstrings


```bash
# Test PBP functionality
python pbp_runner.py

# Test dataset loading
python dataset_runner.py

# Test comprehensive comparison
python comparison_runner.py

# Test dataset testing
python testing_runner.py

# Test example usage
python example_usage.py

# Test all analyses
python run_all_analyses.py
```