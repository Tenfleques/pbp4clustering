# Aggregation Function Optimization Report

## Summary
- Total datasets: 10
- Successful optimizations: 10
- Success rate: 100.0%

## Optimal Aggregation Functions by Dataset

| Dataset | Best Function | Silhouette | Calinski-Harabasz | Davies-Bouldin |
|---------|---------------|------------|-------------------|----------------|
| iris | entropy | 0.6854 | 1391.67 | 0.4442 |
| breast_cancer | robust_adaptive | 0.5336 | 804.91 | 0.7177 |
| wine | iqr | 0.5483 | 176.58 | 0.4886 |
| digits | adaptive | 0.4099 | 2972.11 | 0.8131 |
| diabetes | entropy | 0.3896 | 362.65 | 0.9853 |
| sonar | sum | 0.3347 | 37.24 | 1.8916 |
| glass | iqr | 0.8856 | 2231.25 | 0.2309 |
| seeds | sum | 0.2482 | 58.92 | 1.3833 |
| thyroid | sum | 0.0070 | 5.99 | 3.7320 |
| pima | iqr | 0.8021 | 254.38 | 0.7974 |

## Performance Summary

### Silhouette
- Mean: 0.4844
- Std: 0.2509
- Min: 0.0070
- Max: 0.8856

### Calinski Harabasz
- Mean: 829.5704
- Std: 987.2656
- Min: 5.9939
- Max: 2972.1082

### Davies Bouldin
- Mean: 1.1484
- Std: 0.9748
- Min: 0.2309
- Max: 3.7320

### Inertia
- Mean: 2342.7730
- Std: 3751.6339
- Min: 7.5247
- Max: 12820.0766

## Detailed Results by Dataset

### iris
**Best function:** entropy
**Best combined score:** 2.0326

All function results:

- entropy: Silhouette=0.6854, Calinski-Harabasz=1391.67, Davies-Bouldin=0.4442
- gini: Silhouette=0.6414, Calinski-Harabasz=788.04, Davies-Bouldin=0.4769
- median: Silhouette=0.6259, Calinski-Harabasz=774.92, Davies-Bouldin=0.4883
- trimmed_mean: Silhouette=0.6259, Calinski-Harabasz=774.92, Davies-Bouldin=0.4883
- robust_adaptive: Silhouette=0.6259, Calinski-Harabasz=774.92, Davies-Bouldin=0.4883
- iqr: Silhouette=0.5771, Calinski-Harabasz=711.94, Davies-Bouldin=0.6268
- adaptive: Silhouette=0.5306, Calinski-Harabasz=481.05, Davies-Bouldin=0.6949
- rms: Silhouette=0.5196, Calinski-Harabasz=426.15, Davies-Bouldin=0.7148
- mean: Silhouette=0.4952, Calinski-Harabasz=271.93, Davies-Bouldin=0.7345
- sum: Silhouette=0.4952, Calinski-Harabasz=271.93, Davies-Bouldin=0.7345

### breast_cancer
**Best function:** robust_adaptive
**Best combined score:** 1.2668

All function results:

- robust_adaptive: Silhouette=0.5336, Calinski-Harabasz=804.91, Davies-Bouldin=0.7177
- trimmed_mean: Silhouette=0.5275, Calinski-Harabasz=713.35, Davies-Bouldin=0.7846
- iqr: Silhouette=0.5378, Calinski-Harabasz=659.95, Davies-Bouldin=0.7578
- mean: Silhouette=0.5067, Calinski-Harabasz=528.18, Davies-Bouldin=0.8924
- sum: Silhouette=0.5066, Calinski-Harabasz=528.08, Davies-Bouldin=0.8915
- rms: Silhouette=0.5265, Calinski-Harabasz=505.01, Davies-Bouldin=0.9031
- median: Silhouette=0.5194, Calinski-Harabasz=478.42, Davies-Bouldin=0.9289
- adaptive: Silhouette=0.4369, Calinski-Harabasz=465.07, Davies-Bouldin=0.9817
- entropy: Silhouette=0.2813, Calinski-Harabasz=193.85, Davies-Bouldin=1.2911
- gini: Silhouette=0.2652, Calinski-Harabasz=208.12, Davies-Bouldin=1.4192

### wine
**Best function:** iqr
**Best combined score:** 0.6760

All function results:

- iqr: Silhouette=0.5483, Calinski-Harabasz=176.58, Davies-Bouldin=0.4886
- adaptive: Silhouette=0.4733, Calinski-Harabasz=160.09, Davies-Bouldin=0.6766
- robust_adaptive: Silhouette=0.4567, Calinski-Harabasz=141.74, Davies-Bouldin=0.7538
- trimmed_mean: Silhouette=0.4546, Calinski-Harabasz=140.86, Davies-Bouldin=0.7555
- sum: Silhouette=0.4119, Calinski-Harabasz=167.20, Davies-Bouldin=0.9435
- gini: Silhouette=0.4323, Calinski-Harabasz=129.77, Davies-Bouldin=0.8246
- mean: Silhouette=0.4235, Calinski-Harabasz=132.13, Davies-Bouldin=0.8883
- median: Silhouette=0.3809, Calinski-Harabasz=134.69, Davies-Bouldin=0.8586
- entropy: Silhouette=0.3770, Calinski-Harabasz=111.48, Davies-Bouldin=0.9780
- rms: Silhouette=0.3320, Calinski-Harabasz=89.28, Davies-Bouldin=1.0784

### digits
**Best function:** adaptive
**Best combined score:** 3.3007

All function results:

- adaptive: Silhouette=0.4099, Calinski-Harabasz=2972.11, Davies-Bouldin=0.8131
- median: Silhouette=0.4471, Calinski-Harabasz=1393.36, Davies-Bouldin=0.7812
- sum: Silhouette=0.2767, Calinski-Harabasz=912.19, Davies-Bouldin=1.0254
- robust_adaptive: Silhouette=0.2758, Calinski-Harabasz=804.27, Davies-Bouldin=1.0164
- trimmed_mean: Silhouette=0.2621, Calinski-Harabasz=776.83, Davies-Bouldin=1.0896
- mean: Silhouette=0.2553, Calinski-Harabasz=752.77, Davies-Bouldin=1.0783
- rms: Silhouette=0.2468, Calinski-Harabasz=724.46, Davies-Bouldin=1.0952
- entropy: Silhouette=0.2573, Calinski-Harabasz=679.77, Davies-Bouldin=1.0357
- iqr: Silhouette=0.2577, Calinski-Harabasz=678.78, Davies-Bouldin=1.0487
- gini: Silhouette=0.2401, Calinski-Harabasz=549.74, Davies-Bouldin=1.1129

### diabetes
**Best function:** entropy
**Best combined score:** 0.6537

All function results:

- entropy: Silhouette=0.3896, Calinski-Harabasz=362.65, Davies-Bouldin=0.9853
- gini: Silhouette=0.4022, Calinski-Harabasz=303.86, Davies-Bouldin=0.7478
- iqr: Silhouette=0.3074, Calinski-Harabasz=199.15, Davies-Bouldin=1.1391
- sum: Silhouette=0.2661, Calinski-Harabasz=212.01, Davies-Bouldin=1.2222
- rms: Silhouette=0.2685, Calinski-Harabasz=175.40, Davies-Bouldin=1.2208
- trimmed_mean: Silhouette=0.2694, Calinski-Harabasz=165.19, Davies-Bouldin=1.2260
- robust_adaptive: Silhouette=0.2669, Calinski-Harabasz=164.61, Davies-Bouldin=1.2287
- median: Silhouette=0.2687, Calinski-Harabasz=160.86, Davies-Bouldin=1.2337
- mean: Silhouette=0.2624, Calinski-Harabasz=164.07, Davies-Bouldin=1.2466
- adaptive: Silhouette=0.2541, Calinski-Harabasz=153.51, Davies-Bouldin=1.2854

### sonar
**Best function:** sum
**Best combined score:** 0.1828

All function results:

- sum: Silhouette=0.3347, Calinski-Harabasz=37.24, Davies-Bouldin=1.8916
- adaptive: Silhouette=0.3251, Calinski-Harabasz=33.22, Davies-Bouldin=1.8345
- rms: Silhouette=0.3149, Calinski-Harabasz=37.28, Davies-Bouldin=1.9284
- mean: Silhouette=0.3160, Calinski-Harabasz=35.28, Davies-Bouldin=1.9652
- robust_adaptive: Silhouette=0.3033, Calinski-Harabasz=31.80, Davies-Bouldin=2.1834
- trimmed_mean: Silhouette=0.2761, Calinski-Harabasz=32.51, Davies-Bouldin=2.2046
- median: Silhouette=0.2725, Calinski-Harabasz=30.42, Davies-Bouldin=2.3374
- iqr: Silhouette=0.2441, Calinski-Harabasz=26.55, Davies-Bouldin=2.1048
- entropy: Silhouette=0.1949, Calinski-Harabasz=43.62, Davies-Bouldin=1.8376
- gini: Silhouette=0.1237, Calinski-Harabasz=22.12, Davies-Bouldin=2.6583

### glass
**Best function:** iqr
**Best combined score:** 3.0938

All function results:

- iqr: Silhouette=0.8856, Calinski-Harabasz=2231.25, Davies-Bouldin=0.2309
- entropy: Silhouette=0.8481, Calinski-Harabasz=1298.59, Davies-Bouldin=0.2360
- adaptive: Silhouette=0.6894, Calinski-Harabasz=1007.06, Davies-Bouldin=0.6941
- rms: Silhouette=0.6894, Calinski-Harabasz=1005.96, Davies-Bouldin=0.6942
- sum: Silhouette=0.6874, Calinski-Harabasz=917.99, Davies-Bouldin=0.6753
- mean: Silhouette=0.6796, Calinski-Harabasz=882.27, Davies-Bouldin=0.7343
- median: Silhouette=0.7118, Calinski-Harabasz=784.02, Davies-Bouldin=0.7459
- trimmed_mean: Silhouette=0.7118, Calinski-Harabasz=784.02, Davies-Bouldin=0.7459
- robust_adaptive: Silhouette=0.7118, Calinski-Harabasz=784.02, Davies-Bouldin=0.7459
- gini: Silhouette=0.0000, Calinski-Harabasz=0.00, Davies-Bouldin=inf

### seeds
**Best function:** sum
**Best combined score:** 0.1688

All function results:

- sum: Silhouette=0.2482, Calinski-Harabasz=58.92, Davies-Bouldin=1.3833
- mean: Silhouette=0.2482, Calinski-Harabasz=58.92, Davies-Bouldin=1.3833
- median: Silhouette=0.2482, Calinski-Harabasz=58.92, Davies-Bouldin=1.3833
- trimmed_mean: Silhouette=0.2482, Calinski-Harabasz=58.92, Davies-Bouldin=1.3833
- rms: Silhouette=0.2482, Calinski-Harabasz=58.92, Davies-Bouldin=1.3833
- adaptive: Silhouette=0.2482, Calinski-Harabasz=58.92, Davies-Bouldin=1.3833
- robust_adaptive: Silhouette=0.2482, Calinski-Harabasz=58.92, Davies-Bouldin=1.3833
- entropy: Silhouette=0.0000, Calinski-Harabasz=0.00, Davies-Bouldin=inf
- gini: Silhouette=0.0000, Calinski-Harabasz=0.00, Davies-Bouldin=inf
- iqr: Silhouette=0.0000, Calinski-Harabasz=0.00, Davies-Bouldin=inf

### thyroid
**Best function:** sum
**Best combined score:** -0.3602

All function results:

- sum: Silhouette=0.0070, Calinski-Harabasz=5.99, Davies-Bouldin=3.7320
- mean: Silhouette=0.0070, Calinski-Harabasz=5.99, Davies-Bouldin=3.7320
- median: Silhouette=0.0070, Calinski-Harabasz=5.99, Davies-Bouldin=3.7320
- trimmed_mean: Silhouette=0.0070, Calinski-Harabasz=5.99, Davies-Bouldin=3.7320
- adaptive: Silhouette=0.0070, Calinski-Harabasz=5.99, Davies-Bouldin=3.7320
- robust_adaptive: Silhouette=0.0070, Calinski-Harabasz=5.99, Davies-Bouldin=3.7320
- rms: Silhouette=0.0071, Calinski-Harabasz=5.99, Davies-Bouldin=3.7329
- entropy: Silhouette=0.0000, Calinski-Harabasz=0.00, Davies-Bouldin=inf
- gini: Silhouette=0.0000, Calinski-Harabasz=0.00, Davies-Bouldin=inf
- iqr: Silhouette=0.0000, Calinski-Harabasz=0.00, Davies-Bouldin=inf

### pima
**Best function:** iqr
**Best combined score:** 0.9768

All function results:

- iqr: Silhouette=0.8021, Calinski-Harabasz=254.38, Davies-Bouldin=0.7974
- entropy: Silhouette=0.7759, Calinski-Harabasz=182.50, Davies-Bouldin=0.7777
- sum: Silhouette=0.2657, Calinski-Harabasz=200.39, Davies-Bouldin=1.5796
- rms: Silhouette=0.2613, Calinski-Harabasz=196.53, Davies-Bouldin=1.6001
- adaptive: Silhouette=0.2612, Calinski-Harabasz=196.48, Davies-Bouldin=1.6004
- mean: Silhouette=0.2604, Calinski-Harabasz=195.98, Davies-Bouldin=1.6034
- median: Silhouette=0.2604, Calinski-Harabasz=195.98, Davies-Bouldin=1.6034
- trimmed_mean: Silhouette=0.2604, Calinski-Harabasz=195.98, Davies-Bouldin=1.6034
- robust_adaptive: Silhouette=0.2604, Calinski-Harabasz=195.98, Davies-Bouldin=1.6034
- gini: Silhouette=0.0000, Calinski-Harabasz=0.00, Davies-Bouldin=inf
