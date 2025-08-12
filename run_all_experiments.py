#!/usr/bin/env python3
"""
Comprehensive batch script to run all dataset/argument combinations.
Excludes HAR dataset as requested. Results are saved to CSV and individual files.
"""

import subprocess
import csv
import json
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import argparse
from src.sorting_functions import get_recommended_sorting_functions


# Core aggregation functions to test
AGGREGATION_FUNCTIONS = [
    "sum", "mean", "std", "median", "max", "min", 
    "range", "iqr", "entropy", "gini", "rms",
    "trimmed_mean", "adaptive", "robust_adaptive"
]

# Sorting functions to test (None means default ascending without explicit sort)
SORT_FUNCTIONS_DEFAULT = [None] + get_recommended_sorting_functions()

# Standard datasets (no special arguments)
STANDARD_DATASETS = [
    "iris", "wdbc", "htru2", "seeds", "penguins", 
    "pima", "banknote", "balance", "parkinsons", 
    "sonar", "ionosphere", "wine", "spectroscopy", "retail", "power"
]

# Custom datasets with their argument combinations
CUSTOM_DATASETS = {
    "abalone": [
        {"option": "A"},
        {"option": "B"}
    ],
    "vehicle": [
        {"mode": "2x9"},
        {"mode": "3x6"}
    ],
    "digits": [
        {"rows": 2, "col_blocks": 2},
        {"rows": 2, "col_blocks": 4},
        {"rows": 2, "col_blocks": 8},
        {"rows": 3, "col_blocks": 2},
        {"rows": 3, "col_blocks": 4},
        {"rows": 3, "col_blocks": 8},
    ],
    "spectroscopy": [
        {"row-blocks": 2},
        {"row-blocks": 3}
    ],
    "wine": [
        {"wine-merge-types": True},
        {"wine-merge-types": False}
    ],  
    "retail": [{}],
    "power": [
        {"power-days-limit": 60, "power-hours-block": 4},
        {"power-days-limit": 30, "power-hours-block": 6}
    ]
}


def extract_metrics_from_output(output: str) -> Dict[str, Any]:
    """Extract all metrics from the run_generic.py output."""
    metrics = {}
    
    # Extract basic info
    patterns = {
        'dataset': r'Metrics \((\w+), PBP\)',
        'agg_func': r'agg=(\w+)',
        'n_samples': r'n_samples=(\d+)',
        'n_features': r'n_features=(\d+)', 
        'n_clusters': r'n_clusters=(\d+)',
        
        # Clustering metrics
        'v_measure': r'v_measure=([\d.]+|nan)',
        'adjusted_rand': r'adjusted_rand=([\d.]+|nan)',
        'silhouette': r'silhouette=([\d.]+|nan)', 
        'calinski_harabasz': r'calinski_harabasz=([\d.]+|nan)',
        'davies_bouldin': r'davies_bouldin=([\d.]+|nan)',
        'inertia': r'inertia=([\d.]+|nan)',
        
        # Supervised metrics
        'linear_sep_cv': r'linear_sep_cv=([\d.]+|nan)',
        'cv_score': r'cv_score=([\d.]+|nan)',
        'margin_score': r'margin_score=([\d.]+|nan)',
        'boundary_complexity': r'boundary_complexity=([\d.]+|nan)',
    }
    
    for metric, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            value = match.group(1)
            if value == 'nan':
                metrics[metric] = float('nan')
            elif metric in ['n_samples', 'n_features', 'n_clusters']:
                metrics[metric] = int(value)
            else:
                try:
                    metrics[metric] = float(value)
                except ValueError:
                    metrics[metric] = value
    
    return metrics


def run_single_experiment(
    dataset: str, 
    agg_func: str, 
    custom_args: Dict[str, Any],
    results_dir: str,
    timeout: int = 300
) -> Optional[Dict[str, Any]]:
    """Run a single experiment and extract results."""
    
    # Build command
    cmd = [
        ".venv/bin/python", "run_generic.py",
        "--dataset", dataset,
        "--agg", agg_func,
        "--results-dir", results_dir,
        "--no-plot",  # Disable plotting for batch runs
        "--cv-splits", "3"  # Standard CV splits for batch
    ]
    
    # Add custom arguments
    for key, value in custom_args.items():
        if value is None:
            continue
        if key == "measures":
            cmd.extend([f"--{key}", value])
        else:
            cmd.extend([f"--{key}", str(value)])
    
    print(f"Running: {dataset} (agg={agg_func}) {custom_args}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="."
        )
        
        if result.returncode == 0:
            # Surface cache events for visibility
            if "Loaded cached results:" in result.stdout:
                print("  [cache] hit")
            if "Saved cache:" in result.stdout:
                print("  [cache] save")

            metrics = extract_metrics_from_output(result.stdout)
            
            # Add experiment info
            metrics.update({
                'dataset': dataset,
                'agg_func': agg_func,
                'success': True,
                'error': None,
                **custom_args
            })
            
            return metrics
        else:
            # Print full traceback to terminal, but avoid dumping it into CSV
            print("  ❌ Failed. Full error output follows:\n" + result.stderr)
            # Keep CSV clean: store only a concise one-line summary (last line of stderr)
            last_line = result.stderr.strip().splitlines()[-1] if result.stderr.strip().splitlines() else ""
            return {
                'dataset': dataset,
                'agg_func': agg_func,
                'success': False,
                'error': last_line[:200],
                **custom_args
            }
            
    except subprocess.TimeoutExpired:
        print(f"  ⏰ Timeout after {timeout}s")
        return {
            'dataset': dataset,
            'agg_func': agg_func,
            'success': False,
            'error': f"Timeout after {timeout}s",
            **custom_args
        }
    except Exception as e:
        print(f"  ❌ Exception: {e}")
        return {
            'dataset': dataset,
            'agg_func': agg_func,
            'success': False,
            'error': str(e),
            **custom_args
        }


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive experiments on all datasets")
    parser.add_argument("--output-dir", default="./batch_results", help="Output directory for results")
    parser.add_argument("--agg-funcs", nargs="+", default=AGGREGATION_FUNCTIONS, help="Aggregation functions to test")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per experiment (seconds)")
    parser.add_argument("--sort-funcs", nargs="+", default=None, help="Sorting functions to test; use 'none' for default")
    parser.add_argument("--datasets", nargs="+", help="Specific datasets to test (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be run without executing")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue even if experiments fail")
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"experiment_results_{timestamp}.csv"
    log_file = output_dir / f"experiment_log_{timestamp}.txt"
    
    # Determine which datasets to run
    if args.datasets:
        datasets_to_run = {}
        for ds in args.datasets:
            if ds in STANDARD_DATASETS:
                datasets_to_run[ds] = [{}]
            elif ds in CUSTOM_DATASETS:
                datasets_to_run[ds] = CUSTOM_DATASETS[ds]
            else:
                print(f"Warning: Unknown dataset '{ds}', skipping")
    else:
        datasets_to_run = {**{ds: [{}] for ds in STANDARD_DATASETS}, **CUSTOM_DATASETS}
    
    # Determine which sorting functions to run
    if args.sort_funcs is None:
        sorts_to_run: List[Optional[str]] = SORT_FUNCTIONS_DEFAULT
    else:
        sorts_to_run = [None if s.lower() == "none" else s for s in args.sort_funcs]

    # Generate all experiment combinations
    experiments = []
    for dataset, arg_combinations in datasets_to_run.items():
        for custom_args in arg_combinations:
            for agg_func in args.agg_funcs:
                for sort_name in sorts_to_run:
                    exp_args = dict(custom_args)
                    exp_args.update({"sort": sort_name})
                    experiments.append((dataset, agg_func, exp_args))
    
    print(f"\n🧪 Batch Experiment Plan")
    print(f"Total experiments: {len(experiments)}")
    print(f"Datasets: {sorted(datasets_to_run.keys())}")
    print(f"Aggregation functions: {args.agg_funcs}")
    print(f"Sorting functions: {sorts_to_run}")
    print(f"Results will be saved to: {results_file}")
    print(f"Log will be saved to: {log_file}")
    
    if args.dry_run:
        print(f"\n📋 DRY RUN - Experiments that would be executed:")
        for i, (dataset, agg_func, custom_args) in enumerate(experiments, 1):
            args_str = " ".join(f"--{k} {v}" for k, v in custom_args.items())
            print(f"{i:3d}. {dataset:12s} --agg {agg_func:15s} {args_str}")
        return 0
    
    # Setup CSV file
    all_results = []
    csv_headers = [
        'dataset', 'agg_func', 'success', 'error',
        'n_samples', 'n_features', 'n_clusters',
        'v_measure', 'adjusted_rand', 'silhouette', 'calinski_harabasz', 'davies_bouldin', 'inertia',
        'linear_sep_cv', 'cv_score', 'margin_score', 'boundary_complexity',
        'sort',
        'option', 'mode', 'rows', 'col_blocks', 'measures', 'row-blocks', 'wine-merge-types', 'power-days-limit', 'power-hours-block'  # Custom args
    ]
    
    with open(results_file, 'w', newline='') as csvfile, open(log_file, 'w') as logfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers, extrasaction='ignore')
        writer.writeheader()
        
        successful = 0
        failed = 0
        
        for i, (dataset, agg_func, custom_args) in enumerate(experiments, 1):
            logfile.write(f"\n[{datetime.now()}] Experiment {i}/{len(experiments)}: {dataset} {agg_func} {custom_args}\n")
            logfile.flush()
            
            result = run_single_experiment(
                dataset, agg_func, custom_args, 
                str(output_dir / "plots"),
                args.timeout
            )
            
            if result:
                if result.get('success', False):
                    all_results.append(result)
                    writer.writerow(result)
                    csvfile.flush()
                    successful += 1
                    print(f"  ✅ Success ({successful}/{i})")
                else:
                    failed += 1
                    print(f"  ❌ Failed ({failed}/{i})")
                    if not args.continue_on_error:
                        print("Stopping due to error (use --continue-on-error to continue)")
                        break
                
                logfile.write(f"Result: {result.get('success', False)}\n")
            else:
                failed += 1
                print(f"  ❌ No result returned")
                if not args.continue_on_error:
                    break
            
            # Progress update
            if i % 10 == 0:
                print(f"\n📊 Progress: {i}/{len(experiments)} ({i/len(experiments)*100:.1f}%) - Success: {successful}, Failed: {failed}")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"🎯 BATCH EXPERIMENTS COMPLETE")
    print('='*60)
    print(f"Total experiments: {len(experiments)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/(successful+failed)*100:.1f}%" if (successful+failed) > 0 else "N/A")
    print(f"\nResults saved to: {results_file}")
    print(f"Log saved to: {log_file}")
    
    # Save summary JSON
    summary = {
        "timestamp": timestamp,
        "total_experiments": len(experiments),
        "successful": successful,
        "failed": failed,
        "success_rate": successful/(successful+failed) if (successful+failed) > 0 else 0,
        "datasets": sorted(datasets_to_run.keys()),
        "agg_functions": args.agg_funcs,
        "results_file": str(results_file),
        "log_file": str(log_file)
    }
    
    summary_file = output_dir / f"experiment_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_file}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())