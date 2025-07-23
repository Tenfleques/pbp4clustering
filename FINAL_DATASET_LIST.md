# Final Valid Dataset List for PBP Analysis

## Summary

After comprehensive analysis of the entire codebase, **ONLY 2 datasets have natural combinatorial relationships** that benefit from PBP (Permutation-Based Polynomial) analysis.

## ✅ VALID DATASETS (Natural Combinatorial Relationships)

### 1. Iris Dataset
- **Matrix Structure**: 2×2 (Sepal/Petal × Length/Width)
- **Natural Relationships**: 
  - Sepal vs Petal (different flower parts)
  - Length vs Width (spatial dimensions)
- **Real Features**: sepal length, sepal width, petal length, petal width
- **Domain Logic**: Biological relationship between flower parts and their measurements
- **PBP Benefit**: High - reflects real biological structure

### 2. Breast Cancer Dataset
- **Matrix Structure**: 3×10 (Statistical Measures × Cell Features)
- **Natural Relationships**:
  - Mean vs Standard Error vs Worst (statistical measures of same features)
  - 10 different cell characteristics
- **Real Features**: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension (each with mean, SE, worst)
- **Domain Logic**: Statistical relationship between different measures of the same underlying cell features
- **PBP Benefit**: High - reflects real statistical relationships

## ❌ REMOVED DATASETS (Artificial Grouping)

The following 6 datasets were **REMOVED** because they use artificial grouping that doesn't reflect real relationships:

1. **Diabetes Dataset** - Artificial "Metabolic/Reproductive" grouping
2. **Glass Dataset** - Artificial "Chemical/Physical/Optical" grouping  
3. **Sonar Dataset** - Artificial frequency band grouping
4. **Vehicle Dataset** - Artificial geometric feature grouping
5. **Ecoli Dataset** - Artificial "Cytoplasmic/Membrane" grouping
6. **Digits Dataset** - Spatial arrangement doesn't create meaningful relationships

## Usage

```bash
# Valid datasets for PBP analysis
python3 feature_analysis.py iris --k 2 --visualize
python3 feature_analysis.py breast_cancer --k 3 --visualize

# Invalid datasets (properly rejected)
python3 feature_analysis.py diabetes --k 3 --visualize  # ERROR
python3 feature_analysis.py glass --k 3 --visualize     # ERROR
```

## Files Remaining

```
data/
├── iris_X.npy                    # Original 2×2 matrices
├── iris_y.npy                    # Target labels
├── iris_pbp_features.npy         # PBP features
├── iris_metadata.json            # Dataset info
├── breast_cancer_X.npy           # Original 3×10 matrices
├── breast_cancer_y.npy           # Target labels
├── breast_cancer_pbp_features.npy # PBP features
└── breast_cancer_metadata.json   # Dataset info
```

## Code Updates

- `feature_analysis.py`: Only accepts `['iris', 'breast_cancer']`
- `example_feature_analysis.py`: Updated to use only valid datasets
- All invalid datasets removed from data directory

## Conclusion

The codebase now contains **ONLY datasets with natural combinatorial relationships** that properly benefit from PBP analysis. This ensures the integrity of the PBP approach and prevents meaningless results from artificial groupings. 