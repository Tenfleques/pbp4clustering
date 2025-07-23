# Comprehensive PBP Analysis Report
============================================================

## Clustering Analysis Results

### crystal
- **KMeans**:
  - Silhouette Score: 0.422
  - Calinski-Harabasz Score: 12.256
  - Number of Clusters: 3
  - Cluster Balance: 0.500
- **Hierarchical**:
  - Silhouette Score: 0.496
  - Calinski-Harabasz Score: 14.285
  - Number of Clusters: 3
  - Cluster Balance: 0.750
- **DBSCAN**:
  - Silhouette Score: 0.515
  - Calinski-Harabasz Score: 5.889
  - Number of Clusters: 3
  - Cluster Balance: 0.333

### spectroscopy
- **KMeans**:
  - Silhouette Score: 0.620
  - Calinski-Harabasz Score: 16.048
  - Number of Clusters: 3
  - Cluster Balance: 0.333
- **Hierarchical**:
  - Silhouette Score: 0.620
  - Calinski-Harabasz Score: 16.048
  - Number of Clusters: 3
  - Cluster Balance: 0.333
- **DBSCAN**:
  - Silhouette Score: 0.049
  - Calinski-Harabasz Score: 0.931
  - Number of Clusters: 2
  - Cluster Balance: 0.250

### protein
- **KMeans**:
  - Silhouette Score: 0.180
  - Calinski-Harabasz Score: 4.357
  - Number of Clusters: 3
  - Cluster Balance: 0.250
- **Hierarchical**:
  - Silhouette Score: 0.519
  - Calinski-Harabasz Score: 10.120
  - Number of Clusters: 3
  - Cluster Balance: 0.500
- **DBSCAN**:
  - Silhouette Score: 0.165
  - Calinski-Harabasz Score: 0.971
  - Number of Clusters: 3
  - Cluster Balance: 0.400

### dna
- **KMeans**:
  - Silhouette Score: 1.000
  - Calinski-Harabasz Score: 1.000
  - Number of Clusters: 3
  - Cluster Balance: 0.333
- **Hierarchical**:
  - Silhouette Score: 1.000
  - Calinski-Harabasz Score: 1.000
  - Number of Clusters: 3
  - Cluster Balance: 0.333
- **DBSCAN**:
  - Silhouette Score: 1.000
  - Calinski-Harabasz Score: 1.000
  - Number of Clusters: 3
  - Cluster Balance: 0.333

## Feature Importance Analysis

### crystal
Top features by aggregated importance:
- **b_parameter**: nan
- **alpha_angle**: nan
- **gamma_angle**: nan

### spectroscopy
Top features by aggregated importance:
- **absorption_band_2**: 0.258
- **absorption_band_4**: 0.219
- **absorption_band_3**: 0.192

### protein
Top features by aggregated importance:
- **loop_content**: nan
- **beta_sheet_content**: nan
- **alpha_helix_content**: nan

### dna
Top features by aggregated importance:
- **sequence_length**: nan
- **gc_content**: nan
- **unique_4mers**: nan

## Cross-Domain Natural Relationship Comparison

### crystal
- **Domain**: Materials
- **Structure**: Crystal_System × Lattice_Parameters
- **Natural Relationships**: Crystal systems × Lattice parameters
- **Feature Correlations**: 0.659
- **Interaction Strength**: nan

### spectroscopy
- **Domain**: Chemistry
- **Structure**: Chemical_Type × Absorption_Features
- **Natural Relationships**: Chemical functional groups × Absorption characteristics
- **Feature Correlations**: 0.574
- **Interaction Strength**: 0.431

### protein
- **Domain**: Biology
- **Structure**: Structural_Elements × Spatial_Coordinates
- **Natural Relationships**: Structural elements × Spatial coordinates
- **Feature Correlations**: 0.671
- **Interaction Strength**: nan

### dna
- **Domain**: Genomics
- **Structure**: Nucleotide_Position × Sequence_Features
- **Natural Relationships**: Sequential and structural relationships
- **Feature Correlations**: 0.921
- **Interaction Strength**: nan

## Performance Benchmarking

### crystal
- **PBP Silhouette Score**: 0.422
- **PBP Improvement**: -0.093
- **Natural Relationship Strength**: 0.659
Traditional methods:
  - **KMeans**: 0.422
  - **Hierarchical**: 0.496
  - **DBSCAN**: 0.515

### spectroscopy
- **PBP Silhouette Score**: 0.620
- **PBP Improvement**: 0.000
- **Natural Relationship Strength**: 0.574
Traditional methods:
  - **KMeans**: 0.620
  - **Hierarchical**: 0.620
  - **DBSCAN**: 0.049

### protein
- **PBP Silhouette Score**: 0.180
- **PBP Improvement**: -0.339
- **Natural Relationship Strength**: 0.671
Traditional methods:
  - **KMeans**: 0.180
  - **Hierarchical**: 0.519
  - **DBSCAN**: 0.165

### dna
- **PBP Silhouette Score**: 1.000
- **PBP Improvement**: 0.000
- **Natural Relationship Strength**: 0.921
Traditional methods:
  - **KMeans**: 1.000
  - **Hierarchical**: 1.000
  - **DBSCAN**: 1.000
