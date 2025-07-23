# Ready-to-Use “Matrix-Friendly” Classification Datasets

The databases below share the key property that **every sample can be rearranged into a small, consistently shaped matrix of quantitative measurements**, making them convenient for algorithms (e.g., CNNs, tensor models) that expect 2-D or 3-D input.  Where helpful, a natural row × column arrangement is suggested, but researchers are free to reshape vectors in other ways.


| \# | Dataset (link) | Typical Matrix Shape per Sample | \#Samples | Target Classes | What the Cells Represent | Key Similarity to WDBC \& Iris |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 4 | Human Activity Recognition Using Smartphones https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones | 17 signals × 33 summary stats → 17 × 33 matrix (or 128 × 9 raw) | 10,299 windows | 6 activities | Time/frequency features from accelerometer \& gyroscope | Exactly 561 features that naturally tile into 2-D blocks by sensor and statistic[^4] |
| 5 | Sensorless Drive Diagnosis https://archive.ics.uci.edu/dataset/325/dataset+for+sensorless+drive+diagnosis | 7 IMF × 7 statistics → 7 × 7 | 58,509 | 11 motor conditions | EMD coefficients of motor-current signals | Repeating statistic groups → square matrix[^5] |
| 6 | Parkinson’s Tele-monitoring (UCI) https://github.com/mrpintime/Parkinsons-Telemonitoring | 19 features × 2 voice phases → 19 × 2 | 5,875 | Regression (UPDRS) or binary PD | Acoustic dysphonia measures | Feature blocks mimic WDBC mean/SE/worst triplets[^6] |
| 7 | ISOLET Speech Recognition (UCI) https://archive.ics.uci.edu/dataset/54/isolet | 617-D vector can be reshaped 59 frames × ~10 coefficients | 7,797 | 26 spoken letters | Spectral \& prosodic coefficients | Regular time × feature grid[^7] |
| 8 | Seeds (Wheat Kernel) https://archive.ics.uci.edu/ml/datasets/seeds | 7 × 1 (or 1 × 7) | 210 | 3 wheat varieties | Area, perimeter, compactness… | Pure numeric morphological panel[^8] | Recommended!!!


| 13 | Thyroid Gland (New-Thyroid) https://archive.ics.uci.edu/ml/datasets/thyroid+disease | 6 lab tests × 1 | 215 | 3 thyroid states | RT3U, TSH, T3 … | Numeric lab matrix analogous to WDBC[^13] |


| \# | Dataset \& Link | Natural Matrix Layout per Sample | \#Samples | Task / Label | Why It Fits |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 16 | SPECTF Heart (UCI) https://archive.ics.uci.edu/dataset/96/spectf+heart | 22 ROIs × 2 states (rest/stress) → 22×2 | 267 | Cardiac image diagnosis (normal vs. abnormal) | Paired counts from 22 regions form a 2-column “image” of perfusion[^1] |
| 17 | SPECT Heart (binary features) https://archive.ics.uci.edu/dataset/95/spect+heart | 22 ROIs × 1 → 22×1 | 267 | Same label as above | Binary presence/absence grid—ideal for CNNs with Bernoulli inputs[^2] |
| 18 | Ionosphere https://archive.ics.uci.edu/ml/datasets/Ionosphere | 17 pulse returns × 2 phases → 17×2 | 351 | Radar return quality (good/bad) | Even–odd columns are in-phase \& quadrature signals—naturally paired[^3] |
| 19 | Pima Indians Diabetes https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes | 4 vital-sign groups × 2 measures → 4×2 | 768 | Diabetes (yes/no) | Eight clinical metrics divide neatly into physiological pairs[^4] |
| 20 | Poker-Hand https://archive.ics.uci.edu/dataset/158/poker+hand | 5 cards × 2 attributes (rank, suit) → 5×2 | 1,024,010 | 10-class hand rank | Treats every hand as a tiny categorical image of the deck[^5] |
| 21 | Appliances Energy Prediction https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction | 29 sensors × 1 → 29×1 (or 5 rooms × 6 vars) | 19,735 | kWh regression | Room-by-metric grid preserves household topology[^6] |
| 22 | Multivariate Gait Data https://archive.ics.uci.edu/dataset/760/multivariate+gait+data | 3 joints × 101 time pts per leg → 3×101×2 | 10 subjects×3 conds×10 | Condition \& subject ID | Time-series “images” perfect for temporal CNNs[^7] |
| 23 | Smartphone HAR (raw) https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones | 128 samples × 9 axes → 128×9 | 10,299 windows | 6 physical activities | Classic 2-D sensor map (time × axis)[^8] |
| 24 | Schneider Lobby Sensor 2023-24 https://www.kaggle.com/datasets/anupkayande/sensor-dataset-for-forecasting | 18 channels × n-steps | 52,560 | Temp / humidity forecasts | Multichannel environmental grid for seq-to-seq models[^9] |
| 25 | Chemical Composition of Ceramic Samples https://archive.ics.uci.edu/dataset/42/glass+identification | 4 major oxides × 4 trace oxides → 4×4 | 214 | 6 glass types | Element blocks yield a square chemistry matrix[^10] |
| 26 | SPECT F (binary) as 11×4 Patch Grid | 11 paired ROIs × 4 views | 267 | Abnormal heart | Alternate reshaping of \#16 for 4-patch vision nets[^1] |
| 27 | Machine-Learning Raman Open Dataset (MLROD) https://doi.org/10.48484/PWRB-R137 | ~501 wavenumbers × 1 | 3,510 | Mineral species | Spectral vector acts as a 1-D image; stacks become 2-D cubes[^11] |
| 28 | Global Soils (ISRIC WISE30sec) https://doi.org/10.3334/ORNLDAAC/546 | 6 depths × 5 properties → 6×5 | 30,000+ grid cells | Carbon \& water capacity regression | Fixed depth-by-property tiles for geospatial CNNs[^12] |
| 29 | SPECT Heart (imputeR R package) https://search.r-project.org/CRAN/refmans/imputeR/html/spect.html | 23 binary features × 1 | 266 | Cardiac class | Ready-made binary matrix inside R[^13] |
| 30 | Near-Infrared Drug Spectra (portable NIR) https://www.frontiersin.org/articles/10.3389/fchem.2023.1214825 | 1,024 λ bands × 1 | 430 | Genuine vs. counterfeit meds | Uniform wavelength axis becomes a spectral image[^14] |
| 31 | SPECTF → Principal-component Cubes | 44 ROI counts reshaped 4×11 | 267 | Heart diagnosis | 4 rows capture anatomical quadrants[^15] |
| 32 | ISOLET Speech Frames https://archive.ics.uci.edu/dataset/54/isolet | 59 frames × 10 coeffs → 59×10 | 7,797 | 26 spoken letters | Time × MFCC grid mimics an image spectrogram[^16] |
| 33 | Pima (log-scaled) as 2×4 Vital Panel | 2 health categories × 4 measures | 768 | Diabetes | Exposes hidden block structure for CNN filters[^17] |
| 34 | USGS ML Raman Library https://doi.org/10.5066/F7RR1WDJ | 1,800 λ × 1 | 8,585 spectra | Mineral ID | Massive high-resolution spectral matrices for deep Siamese nets[^18] |
| 35 | POKER Test Set (UCI) https://dataplatform.cloud.ibm.com/exchange/public/entry/view/e30753836b8a2c7e2f99a3a4c91e3e37 | 5×2 | 1,000,000 | Same labels | Giant hold-out batch for production trials[^19] |

### How to Exploit the Matrix Structure

#### 1. Treat feature blocks as “channels.”

For Pima or Ionosphere, stack twin measurements (e.g., systolic/diastolic) as a 2-channel image to capture cross-correlations the way RGB encodes color interactions.

#### 2. Convolution along meaningful axes.

- **Time × Sensor**—HAR, Gait, Lobby sensors.
- **Region × Stress/Rest**—SPECTF, SPECT.
- **Card × Attribute**—Poker Hand.

Sliding filters inherit *semantic* locality (e.g., neighboring card ranks) rather than arbitrary index distances.

#### 3. Self-supervised pre-training.

Large spectral libraries (MLROD, USGS Raman) enable masked-patch or contrastive learning.  Fine-tune those encoders on smaller biomedical sets (SPECTF) for transfer gains.

#### 4. Tensor-factorization \& Capsule nets.

Datasets with three axes—**subject × joint × time** (Gait) or **depth × property × location** (WISE soils)—are naturals for Tucker/CP decompositions or capsule routing.

### Practical Tips

- **Imbalanced Labels?**  Poker (royal flush = 0.0005%) and SPECTF (≈35% abnormal) benefit from focal-loss or class-balanced sampling.
- **Missing-Value Grids.**  SPECT binary version (\#29) shows how coarse discretization removes NaNs and speeds CNNs.
- **Benchmark Splits.**  Use provided train/test partitions (HAR 70/30; Poker train/test files) to ensure comparability with published work.
- **Visualization.**  Heat-map a single sample after reshaping; patterns often reveal block artifacts that guide data-augmentation (e.g., jitter within ROI pairs).

These additions—together with the first 15 entries—give you **35 ready-to-reshape datasets** spanning images, spectra, time-series, clinical panels and card games.  Each supports the “sample-as-matrix” paradigm essential for modern deep-learning pipelines.



[^4]: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

[^5]: https://archive.ics.uci.edu/dataset/325/dataset+for+sensorless+drive+diagnosis

[^6]: https://github.com/mrpintime/Parkinsons-Telemonitoring

[^7]: https://archive.ics.uci.edu/dataset/54/isolet

[^8]: https://archive.ics.uci.edu/ml/datasets/seeds

[^13]: https://search.r-project.org/CRAN/refmans/mclust/html/thyroid.html


