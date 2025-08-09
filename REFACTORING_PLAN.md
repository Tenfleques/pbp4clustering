# Code Refactoring Plan - Clustering Project

## Executive Summary
The codebase contains ~1000+ lines of duplicated code across 11 run_*.py files. This plan outlines a systematic approach to modularize repeated code while preserving all functionality.

## Current State Analysis

### Duplication Statistics
- **9 files** contain identical metrics calculation functions (~107 lines each = 963 lines total)
- **10 files** contain identical zero-column filtering code
- **10 files** contain identical KMeans clustering setup
- **All files** follow nearly identical workflow pattern

### Key Areas of Duplication
1. Metrics calculation (clustering & supervised learning metrics)
2. Command-line argument parsing
3. Data preprocessing (zero column filtering)
4. Clustering workflow (KMeans setup and execution)
5. Results formatting and output

## Refactoring Strategy

### Phase 1: Core Modules Creation (Priority: HIGH)

#### 1.1 Create `src/metrics.py`
Consolidate all metrics calculation functions:
```python
# Functions to implement:
- safe_cluster_metrics(X, y_true, y_pred, kmeans_model)
- safe_supervised_metrics(X, y, cv_splits=3)
- calculate_all_metrics(X, y_true, y_pred, kmeans_model, cv_splits=3)
```

#### 1.2 Create `src/pipeline.py`
Standard clustering pipeline:
```python
# Functions to implement:
- run_clustering_pipeline(X, y, agg_func, results_dir, plot=True, dataset_name="")
- filter_zero_columns(X)
- setup_kmeans(n_clusters, random_state=0, n_init=10)
```

#### 1.3 Create `src/utils.py`
Common utility functions:
```python
# Functions to implement:
- aggregate_rows_in_blocks(X, block_size)
- format_results(metrics_dict, dataset_name, agg_func)
- print_metrics_summary(cluster_metrics, supervised_metrics, metadata)
```

### Phase 2: Argument Parser Module (Priority: MEDIUM)

#### 2.1 Create `src/cli_args.py`
Standardized argument parsing:
```python
# Functions to implement:
- get_base_parser() -> ArgumentParser  # Common args: --agg, --results-dir, --plot
- add_data_dir_arg(parser) -> ArgumentParser
- add_dataset_specific_args(parser, dataset_name) -> ArgumentParser
```

### Phase 3: Runner Base Class (Priority: MEDIUM)

#### 3.1 Create `src/base_runner.py`
Abstract base class for all runners:
```python
class BaseRunner:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        
    def load_data(self, args) -> Tuple[np.ndarray, np.ndarray]:
        # To be overridden by specific runners
        raise NotImplementedError
        
    def run(self, args):
        # Standard pipeline implementation
        X, y = self.load_data(args)
        X_pbp = matrices_to_pbp_vectors(X, args.agg)
        X_pbp = filter_zero_columns(X_pbp)
        
        if args.plot:
            scatter_features(X_pbp, y, ...)
            
        pred = cluster_and_predict(X_pbp, y)
        metrics = calculate_all_metrics(X_pbp, y, pred)
        print_metrics_summary(metrics, self.dataset_name, args.agg)
```

### Phase 4: Refactor Individual Runners (Priority: HIGH)

#### 4.1 Simplified Runner Template
Each run_*.py file becomes ~30-40 lines:
```python
from src.base_runner import BaseRunner
from src.cli_args import get_base_parser
from datasets.xxx import load_xxx_data

class XxxRunner(BaseRunner):
    def __init__(self):
        super().__init__("xxx")
        
    def load_data(self, args):
        return load_xxx_data(args.data_dir)

if __name__ == "__main__":
    parser = get_base_parser()
    parser.add_argument("--data-dir", ...)  # If needed
    args = parser.parse_args()
    
    runner = XxxRunner()
    runner.run(args)
```

## Implementation Plan

### Week 1: Core Modules
- [ ] Day 1-2: Implement `src/metrics.py` with comprehensive tests
- [ ] Day 3-4: Implement `src/pipeline.py` and `src/utils.py`
- [ ] Day 5: Test modules with one runner (suggest starting with `run_iris2.py`)

### Week 2: Infrastructure
- [ ] Day 1-2: Implement `src/cli_args.py`
- [ ] Day 3-4: Implement `src/base_runner.py`
- [ ] Day 5: Refactor 2-3 runners as proof of concept

### Week 3: Full Migration
- [ ] Day 1-3: Refactor remaining runners
- [ ] Day 4: Update tests and documentation
- [ ] Day 5: Final testing and validation

## Benefits

### Immediate Benefits
- **70-80% code reduction** (~1000+ lines removed)
- **Single source of truth** for metrics and pipeline logic
- **Easier debugging** - fix bugs in one place
- **Consistent behavior** across all datasets

### Long-term Benefits
- **Easier to add new datasets** - just extend BaseRunner
- **Simpler testing** - test core modules once
- **Better maintainability** - clear separation of concerns
- **Easier to add new metrics** - modify one file instead of 9

## Risk Mitigation

### Preserving Functionality
1. Keep original files until refactoring is complete
2. Run side-by-side comparisons of outputs
3. Version control allows easy rollback

### Testing Strategy
1. Create unit tests for new modules before migration
2. Compare outputs before/after refactoring
3. Test each dataset individually

## Files to Create

```
src/
├── metrics.py         # ~150 lines (consolidating 963 lines)
├── pipeline.py        # ~100 lines
├── utils.py          # ~50 lines
├── cli_args.py       # ~50 lines
└── base_runner.py    # ~80 lines
```

## Files to Modify

All run_*.py files will be reduced from ~200 lines to ~40 lines each:
- run_abalone.py
- run_htru2.py
- run_iris2.py
- run_penguins.py
- run_retail.py
- run_seeds.py
- run_spectroscopy.py
- run_wdbc.py
- run_wine.py
- run_har.py (special case - simpler)
- run_iris.py (special case - custom visualization)

## Success Metrics

- [ ] All tests pass with identical outputs
- [ ] Code duplication reduced by >70%
- [ ] New dataset can be added in <50 lines of code
- [ ] All existing functionality preserved
- [ ] Improved code readability and maintainability

## Next Steps

1. **Review and approve this plan**
2. **Create feature branch** for refactoring
3. **Start with Phase 1** - Core modules creation
4. **Incremental testing** after each phase
5. **Documentation updates** as final step