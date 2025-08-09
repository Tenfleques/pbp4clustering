# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PhD research project implementing Pseudo-Boolean Polynomial (PBP) transformations for clustering. The core innovation transforms multi-dimensional matrices into polynomial basis representations to improve clustering performance.

## Key Commands

### Running experiments
```bash
# Run HAR dataset with specific aggregation function
python run_har.py --data-dir ./data/har --results-dir ./results --agg sum

# Run with different configurations
python run_har.py --agg entropy --axis-feature-format --include-body-acc

# Batch run multiple configurations
bash run_top_plots.sh  # Runs 18 different HAR configurations
```

### Common dataset runners
- `run_har.py` - Primary UCI HAR human activity recognition dataset
- `run_iris.py` - Iris dataset clustering
- `run_wine.py` - Wine quality dataset
- `run_seeds.py` - Seeds dataset
- Other runners follow same pattern: `run_*.py`

## Architecture

### Core Pipeline
1. **Data Loading** (`datasets/*.py`): Converts raw data → (N, m, n) matrices
   - N = number of samples
   - m = number of features/axes  
   - n = time steps or dimensions

2. **PBP Transformation** (`pbp_transform.py`): Main transformation entry point
   - Calls `src/core.py` for polynomial basis computation
   - Uses `src/aggregation_functions.py` for term reduction

3. **Visualization** (`visualize.py`): Creates PCA scatter plots with true labels

### Key Modules

- `src/core.py`: Core PBP algorithms - polynomial basis creation, permutation matrices, variable encoding
- `src/aggregation_functions.py`: 25+ aggregation functions (sum, mean, entropy, adaptive, etc.)
- `datasets/`: Dataset loaders that normalize to consistent matrix format
- `pbp_transform.py`: Main transformation orchestrator

### Data Flow Pattern
```
Raw Data → Matrix(N,m,n) → PBP Transform → Polynomial Vector → Aggregation → PCA → Visualization
```

## Development Notes

- No formal dependency management - install common scientific Python packages manually (numpy, pandas, scikit-learn, matplotlib, scipy, bitarray)
- No testing framework - validate changes by running experiments and checking visualizations
- Results saved to `results/` directory with hierarchical organization by parameters
- Data files downloaded automatically to `data/` directory on first run

## Important Implementation Details

- All datasets must be converted to (N, m, n) matrix format
- PBP creates 2^m - 1 dimensional vectors from m×n matrices
- Aggregation functions reduce polynomial terms to fixed-length representations
- Visualization uses PCA to project high-dimensional PBP vectors to 2D