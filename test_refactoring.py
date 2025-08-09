#!/usr/bin/env python3
"""
Test script to compare outputs between original and refactored runners.
This script runs both versions and compares their metrics to ensure consistency.
"""

import subprocess
import sys
import re
from pathlib import Path
from typing import Dict, Tuple, Optional
import argparse


def extract_metrics(output: str) -> Dict[str, float]:
    """Extract metrics from runner output."""
    metrics = {}
    
    # Extract clustering metrics
    patterns = {
        'v_measure': r'v_measure=([\d.]+|nan)',
        'adjusted_rand': r'adjusted_rand=([\d.]+|nan)',
        'silhouette': r'silhouette=([\d.]+|nan)',
        'calinski_harabasz': r'calinski_harabasz=([\d.]+|nan)',
        'davies_bouldin': r'davies_bouldin=([\d.]+|nan)',
        'inertia': r'inertia=([\d.]+|nan)',
        'linear_sep_cv': r'linear_sep_cv=([\d.]+|nan)',
        'cv_score': r'cv_score=([\d.]+|nan)',
        'margin_score': r'margin_score=([\d.]+|nan)',
        'boundary_complexity': r'boundary_complexity=([\d.]+|nan)',
        'n_samples': r'n_samples=(\d+)',
        'n_features': r'n_features=(\d+)',
        'n_clusters': r'n_clusters=(\d+)',
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
                metrics[metric] = float(value)
    
    return metrics


def run_command(cmd: str, timeout: int = 60) -> Tuple[bool, str, str]:
    """Run a command and return success status, stdout, and stderr."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"Command timed out after {timeout} seconds"
    except Exception as e:
        return False, "", str(e)


def compare_metrics(original: Dict[str, float], refactored: Dict[str, float], tolerance: float = 0.0001) -> Tuple[bool, list]:
    """Compare two sets of metrics within tolerance."""
    differences = []
    all_match = True
    
    # Get all keys from both dictionaries
    all_keys = set(original.keys()) | set(refactored.keys())
    
    for key in sorted(all_keys):
        orig_val = original.get(key, None)
        ref_val = refactored.get(key, None)
        
        if orig_val is None:
            differences.append(f"  ❌ {key}: missing in original")
            all_match = False
        elif ref_val is None:
            differences.append(f"  ❌ {key}: missing in refactored")
            all_match = False
        elif isinstance(orig_val, (int, float)) and isinstance(ref_val, (int, float)):
            # Handle NaN values
            import math
            if math.isnan(orig_val) and math.isnan(ref_val):
                differences.append(f"  ✅ {key}: both NaN")
            elif math.isnan(orig_val) or math.isnan(ref_val):
                differences.append(f"  ❌ {key}: {orig_val} vs {ref_val}")
                all_match = False
            elif abs(orig_val - ref_val) > tolerance:
                differences.append(f"  ❌ {key}: {orig_val:.6f} vs {ref_val:.6f} (diff: {abs(orig_val - ref_val):.6f})")
                all_match = False
            else:
                differences.append(f"  ✅ {key}: {orig_val:.6f} ≈ {ref_val:.6f}")
    
    return all_match, differences


def test_runner(runner_name: str, args: str = "--agg sum --no-plot", verbose: bool = False) -> bool:
    """Test a single runner by comparing original vs refactored."""
    print(f"\n{'='*60}")
    print(f"Testing: {runner_name}")
    print(f"Arguments: {args}")
    print('='*60)
    
    original_script = f"run_{runner_name}.py"
    refactored_script = f"run_{runner_name}_refactored.py"
    
    # Check if both scripts exist
    if not Path(original_script).exists():
        print(f"❌ Original script not found: {original_script}")
        return False
    
    if not Path(refactored_script).exists():
        print(f"❌ Refactored script not found: {refactored_script}")
        return False
    
    # Run original
    print(f"\n📊 Running original: {original_script}")
    orig_success, orig_stdout, orig_stderr = run_command(f".venv/bin/python {original_script} {args}")
    
    if not orig_success:
        print(f"❌ Original failed to run")
        if verbose:
            print(f"Error: {orig_stderr}")
        return False
    
    orig_metrics = extract_metrics(orig_stdout)
    if verbose:
        print(f"Original metrics extracted: {len(orig_metrics)} values")
    
    # Run refactored
    print(f"📊 Running refactored: {refactored_script}")
    ref_success, ref_stdout, ref_stderr = run_command(f".venv/bin/python {refactored_script} {args}")
    
    if not ref_success:
        print(f"❌ Refactored failed to run")
        if verbose:
            print(f"Error: {ref_stderr}")
        return False
    
    ref_metrics = extract_metrics(ref_stdout)
    if verbose:
        print(f"Refactored metrics extracted: {len(ref_metrics)} values")
    
    # Compare metrics
    print(f"\n📈 Comparing metrics:")
    match, differences = compare_metrics(orig_metrics, ref_metrics)
    
    for diff in differences:
        print(diff)
    
    if match:
        print(f"\n✅ SUCCESS: All metrics match within tolerance!")
    else:
        print(f"\n❌ FAILURE: Metrics do not match!")
    
    return match


def main():
    parser = argparse.ArgumentParser(description="Test refactored runners against originals")
    parser.add_argument("runners", nargs="*", help="Specific runners to test (e.g., iris2 seeds)")
    parser.add_argument("--all", action="store_true", help="Test all available refactored runners")
    parser.add_argument("--agg", default="sum", help="Aggregation function to test (default: sum)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue testing even if one fails")
    args = parser.parse_args()
    
    # Determine which runners to test
    if args.all:
        # Find all refactored runners
        refactored_files = list(Path(".").glob("run_*_refactored.py"))
        runners = [f.stem.replace("run_", "").replace("_refactored", "") 
                  for f in refactored_files]
    elif args.runners:
        runners = args.runners
    else:
        # Default test set
        runners = ["iris2", "seeds", "htru2", "penguins", "wdbc"]
    
    print(f"🧪 Testing Refactored Runners")
    print(f"Runners to test: {', '.join(runners)}")
    
    # Test arguments
    test_args = f"--agg {args.agg} --no-plot"
    
    # Track results
    results = {}
    
    for runner in runners:
        try:
            success = test_runner(runner, test_args, verbose=args.verbose)
            results[runner] = success
            
            if not success and not args.continue_on_error:
                print(f"\n⚠️  Stopping due to failure. Use --continue-on-error to test remaining runners.")
                break
                
        except Exception as e:
            print(f"\n❌ Exception testing {runner}: {e}")
            results[runner] = False
            if not args.continue_on_error:
                break
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 SUMMARY")
    print('='*60)
    
    for runner, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{runner:20s}: {status}")
    
    passed = sum(1 for s in results.values() if s)
    total = len(results)
    print(f"\nTotal: {passed}/{total} passed")
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())