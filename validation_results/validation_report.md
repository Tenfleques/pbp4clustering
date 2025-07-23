# PBP Algorithm Validation Report
==================================================

## Natural Relationship Detection

### crystal
- Natural relationship score: 1.070
- Original correlation: 0.738
- Random correlation: 0.690

### spectroscopy
- Natural relationship score: 0.889
- Original correlation: 0.631
- Random correlation: 0.710

### protein
- Natural relationship score: 1.000
- Original correlation: 1.000
- Random correlation: 1.000

### dna
- Natural relationship score: 1.237
- Original correlation: 0.918
- Random correlation: 0.742

## Clustering Algorithm Performance

### crystal - KMEANS
- Original silhouette: 0.062
- Random silhouette: 0.150
- Improvement: -0.088

### crystal - HIERARCHICAL
- Original silhouette: 0.140
- Random silhouette: 0.150
- Improvement: -0.011

### spectroscopy - KMEANS
- Original silhouette: 0.088
- Random silhouette: -0.048
- Improvement: 0.136

### spectroscopy - HIERARCHICAL
- Original silhouette: 0.088
- Random silhouette: 0.209
- Improvement: -0.122

### protein - KMEANS
- Original silhouette: 0.000
- Random silhouette: 0.000
- Improvement: 0.000

### protein - HIERARCHICAL
- Original silhouette: 0.000
- Random silhouette: 0.000
- Improvement: 0.000

### dna - KMEANS
- Original silhouette: 0.295
- Random silhouette: 0.182
- Improvement: 0.113

### dna - HIERARCHICAL
- Original silhouette: 0.295
- Random silhouette: 0.182
- Improvement: 0.113

## Parameter Tuning Results

### crystal
- Best K-means: k=2 (score=0.062)
- Best Hierarchical: k=2 (score=0.140)

### spectroscopy
- Best K-means: k=2 (score=0.088)
- Best Hierarchical: k=2 (score=0.088)

### protein
- Best K-means: k=2 (score=-1.000)
- Best Hierarchical: k=2 (score=-1.000)

### dna
- Best K-means: k=2 (score=0.295)
- Best Hierarchical: k=2 (score=0.295)

## Cross-Domain Comparison

### crystal
- Structure: Crystal_System × Lattice_Parameters
- Natural relationships: Crystal systems × Lattice parameters
- Feature correlations: 0.738
- Feature variance: 50.594

### spectroscopy
- Structure: Chemical_Type × Absorption_Features
- Natural relationships: Chemical functional groups × Absorption characteristics
- Feature correlations: 0.631
- Feature variance: 0.012

### protein
- Structure: Structural_Elements × Spatial_Coordinates
- Natural relationships: Structural elements × Spatial coordinates
- Feature correlations: 1.000
- Feature variance: 0.480

### dna
- Structure: Nucleotide_Position × Sequence_Features
- Natural relationships: Sequential and structural relationships
- Feature correlations: 0.918
- Feature variance: 33301184.334
