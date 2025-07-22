# PBP Clustering and Dimensionality Reduction

## Overview

This repository contains the implementation and analysis of the Pseudo-Boolean Polynomial (PBP) method for dimensionality reduction and clustering. The codebase has been reorganized for easy reproduction and distribution.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd pbp4clustering
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the example:**
   ```bash
   python example_usage.py
   ```

## 🔬 Key Features

### **PBP Method**
- **Interpretable Features**: Each feature has clear mathematical meaning
- **Sample Independence**: Each sample processed independently
- **Deterministic Results**: Reproducible every time
- **No Population Bias**: No assumptions about data distribution

### **Supported Datasets**
- **Iris**: 150 sample
- **Breast Cancer**: 569 samples
- **Wine**: 356 samples
- **Digits**: 1797 samples
- **Diabetes**: 768 samples
- **Sonar**: 208 samples
- **Glass**: 426 samples
- **Vehicle**: 376 samples
- **Ecoli**: 336 samples
- **Yeast**: 1484 samples

### **Core Functions**
- `pbp_vector()`: Create PBP vector representation
- `create_pbp()`: Create full PBP polynomial representation
- `truncate_pBp()`: Truncate PBP polynomials
- `DatasetTransformer`: Load and transform datasets

### **Dataset Characteristics**
The method has been tested on 10 diverse datasets with varying characteristics:
- **Small datasets** (150-500 samples): Iris, Wine, Sonar, Glass, Vehicle, Ecoli
- **Medium datasets** (500-1000 samples): Breast Cancer, Diabetes  
- **Large datasets** (1000+ samples): Digits (1797 samples), Yeast (1484 samples)
- **Clustering complexity**: 2-10 clusters across different datasets

## 🛠️ Usage

### Basic Usage

```python
from src.pbp.core import pbp_vector, create_pbp
from src.data.loader import DatasetTransformer

# Load dataset
transformer = DatasetTransformer()
iris_data = transformer.load_iris_dataset()
X = iris_data['X']

# Apply PBP transformation to a sample
sample_matrix = X[0]  # Shape: (3, 3) for iris
pbp_vector_result = pbp_vector(sample_matrix)
print(f"PBP vector length: {len(pbp_vector_result)}")

# Create full PBP representation
pbp_df = create_pbp(sample_matrix)
print(pbp_df)
```

### Clustering Example

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from src.pbp.core import pbp_vector

# Transform all samples
pbp_vectors = []
for i in range(X.shape[0]):
    sample_matrix = X[i]
    pbp_vector_result = pbp_vector(sample_matrix)
    pbp_vectors.append(pbp_vector_result)

X_pbp = np.array(pbp_vectors)

# Apply clustering
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X_pbp)
silhouette_avg = silhouette_score(X_pbp, cluster_labels)

print(f"Silhouette score: {silhouette_avg:.3f}")
```

## 📊 Results

### Performance Comparison

| Dataset | Method | Silhouette Score | Davies-Bouldin | Best Method |
|---------|--------|------------------|----------------|-------------|
| Iris | **PBP** | **0.656** | **0.487** | **PBP** |
| Iris | PCA | 0.466 | 0.818 | - |
| Iris | UMAP | 0.650 | 0.479 | - |
| Iris | t-SNE | 0.302 | 1.204 | - |
| Wine | **PBP** | **0.783** | **0.397** | **PBP** |
| Wine | PCA | 0.716 | 0.584 | - |
| Wine | UMAP | 0.708 | 0.608 | - |
| Wine | t-SNE | 0.570 | 0.626 | - |
| Glass | **PBP** | **0.770** | **0.465** | **PBP** |
| Glass | PCA | 0.733 | 0.647 | - |
| Glass | UMAP | 0.738 | 0.540 | - |
| Glass | t-SNE | 0.464 | 0.852 | - |

### Key Advantages

#### **PBP Advantages**
- **Interpretable Features**: Each feature has clear mathematical meaning
- **Sample Independence**: Each sample processed independently
- **Deterministic Results**: Reproducible every time
- **No Population Bias**: No assumptions about data distribution
- **Competitive Performance**: Outperforms or matches other methods on multiple datasets

#### **Performance Highlights**
- **Wine Dataset**: PBP achieves the best Silhouette score (0.783) and Davies-Bouldin score (0.397)
- **Glass Dataset**: PBP achieves the best Silhouette score (0.770) and Davies-Bouldin score (0.465)
- **Iris Dataset**: PBP achieves the best Silhouette score (0.656) and Davies-Bouldin score (0.487)

#### **Traditional Method Limitations**
- **UMAP**: Non-interpretable features, stochastic results
- **t-SNE**: Stochastic results, no interpretability, poor performance on some datasets
- **PCA**: Linear assumption, limited interpretability, often inferior performance

## 🧪 Testing

Run the test suite:

```bash
python test_datasets.py
```

### **Comprehensive Evaluation Results**

The PBP method has been evaluated against PCA, UMAP, and t-SNE across 10 datasets:

**Top Performers by Dataset:**
- **Wine**: PBP (Silhouette: 0.783, Davies-Bouldin: 0.397)
- **Glass**: PBP (Silhouette: 0.770, Davies-Bouldin: 0.465)  
- **Iris**: PBP (Silhouette: 0.656, Davies-Bouldin: 0.487)
- **Breast Cancer**: PBP (Silhouette: 0.648, Davies-Bouldin: 0.622)
- **Ecoli**: PBP (Silhouette: 0.535, Davies-Bouldin: 0.529)
- **Yeast**: PBP (Silhouette: 0.530, Davies-Bouldin: 0.508)

**Method Comparison Summary:**
- **PBP**: Best performance on 6/10 datasets
- **UMAP**: Best performance on 3/10 datasets  
- **PCA**: Best performance on 1/10 datasets
- **t-SNE**: Best performance on 0/10 datasets

## 📚 Documentation

- **API Documentation**: See individual Python files for function documentation
- **Examples**: See `example_usage.py` for usage examples

## 🔧 Development

### Project Structure

```
pbp4clustering/
├── src/                    # Source code
│   ├── pbp/               # PBP core implementation
│   ├── data/              # Dataset loading utilities
│   └── analysis/          # Analysis and comparison tools
├── data/                  # Dataset files (.npy, .json)
├── results/               # Generated results
│   ├── figures/           # Visualization plots
│   ├── tables/            # Performance tables
│   └── models/            # Saved models
├── articles/              # Research papers and documentation
└── scripts/               # Execution scripts
```

### Available Scripts

- `run_all_analyses.py`: Comprehensive analysis across all datasets
- `pbp_runner.py`: PBP-specific analysis and visualization
- `comparison_runner.py`: Method comparison (PBP vs PCA/UMAP/t-SNE)
- `testing_runner.py`: Testing and validation scripts
- `example_usage.py`: Basic usage examples

### Adding New Datasets

1. Add dataset loading function to `src/data/loader.py`
2. Update the `DatasetTransformer` class
3. Add dataset to the available datasets list

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings to all functions
- Keep functions under 50 lines when possible

## 📄 Citation

```
Chikake, T., & Goldengorin, B. (2025). 
Pseudo-Boolean Polynomial Method for Interpretable Dimensionality Reduction: 
A Paradigm Shift from Abstract to Meaningful Feature Extraction.
[Manuscript in preparation]

Conference Version:
Chikake, T., & Goldengorin, B. (2024). 
Pseudo-Boolean Polynomial Method for Interpretable Dimensionality Reduction.
[Conference proceedings]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Contact

- **Tendai Chikake**: Primary investigator
- **Boris Goldengorin**: Co-investigator

## 📋 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**PBP: Revolutionizing Dimensionality Reduction and Clustering** 🧮📊🔬 