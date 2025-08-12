#!/usr/bin/env python3
"""
Batch runner for dimensionality reduction baselines (PCA, t-SNE, UMAP).
Runs all dataset/method combinations via run_dimred_baselines.py and writes a CSV
for easy comparison with PBP batch results.
"""

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import subprocess


DATASETS = [
    "iris", "wdbc", "htru2", "seeds", "penguins",
    "banknote", "ionosphere", "sonar", "parkinsons",
    "pima", "vehicle", "spectroscopy",
]

METHODS = ["pca", "tsne", "umap"]


def extract_metrics(stdout: str) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    patt = {
        "dataset": r"Metrics \((\w+), [A-Z]+\)",
        "method": r"Metrics \(\w+, ([A-Z]+)\)",
        "n_samples": r"n_samples=(\d+)",
        "n_features": r"n_features=(\d+)",
        "n_clusters": r"n_clusters=(\d+)",
        "v_measure": r"v_measure=([\d.]+|nan)",
        "adjusted_rand": r"adjusted_rand=([\d.]+|nan)",
        "silhouette": r"silhouette=([\d.]+|nan)",
        "calinski_harabasz": r"calinski_harabasz=([\d.]+|nan)",
        "davies_bouldin": r"davies_bouldin=([\d.]+|nan)",
        "inertia": r"inertia=([\d.]+|nan)",
        "linear_sep_cv": r"linear_sep_cv=([\d.]+|nan)",
        "cv_score": r"cv_score=([\d.]+|nan)",
        "margin_score": r"margin_score=([\d.]+|nan)",
        "boundary_complexity": r"boundary_complexity=([\d.]+|nan)",
    }
    for k, rgx in patt.items():
        m = re.search(rgx, stdout)
        if m:
            val = m.group(1)
            if val == "nan":
                metrics[k] = float("nan")
            elif k in {"n_samples", "n_features", "n_clusters"}:
                metrics[k] = int(val)
            else:
                try:
                    metrics[k] = float(val)
                except ValueError:
                    metrics[k] = val
    return metrics


def run_one(dataset: str, method: str, results_dir: str, timeout: int) -> Dict[str, Any]:
    cmd = [
        ".venv/bin/python", "run_dimred_baselines.py",
        "--dataset", dataset,
        "--method", method,
    
        "--no-plot",
        "--results-dir", results_dir,
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=".")
        if res.returncode == 0:
            m = extract_metrics(res.stdout)
            m.update({
                "dataset": dataset,
                "method": method,
                "success": True,
                "error": None,
            })
            return m
        else:
            print(f"  ❌ Failed {dataset}/{method}. Full error follows:\n{res.stderr}")
            last = res.stderr.strip().splitlines()[-1] if res.stderr.strip().splitlines() else ""
            return {
                "dataset": dataset,
                "method": method,
                "success": False,
                "error": last[:200],
            }
    except subprocess.TimeoutExpired:
        print(f"  ⏰ Timeout {dataset}/{method} after {timeout}s")
        return {"dataset": dataset, "method": method, "success": False, "error": f"timeout {timeout}s"}


def main() -> int:
    ap = argparse.ArgumentParser("Run PCA/t-SNE/UMAP baselines across datasets")
    ap.add_argument("--output-dir", default="./batch_results", help="Directory for CSV/log outputs")
    ap.add_argument("--results-dir", default="./baseline_results/plots", help="Figures/plots output root passed to runner")
    ap.add_argument("--datasets", nargs="+", help="Subset of datasets to run")
    ap.add_argument("--methods", nargs="+", choices=METHODS, help="Subset of methods to run")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"baseline_results_{ts}.csv"
    log_path = out_dir / f"baseline_log_{ts}.txt"

    datasets = args.datasets if args.datasets else DATASETS
    methods = args.methods if args.methods else METHODS

    jobs: List[Tuple[str, str]] = [(ds, m) for ds in datasets for m in methods]

    print("\n🧪 Baseline Experiment Plan")
    print(f"Total runs: {len(jobs)}")
    print(f"Datasets: {sorted(datasets)}")
    print(f"Methods: {methods}")
    print(f"CSV: {csv_path}")
    print(f"Log: {log_path}")

    if args.dry_run:
        for i, (ds, m) in enumerate(jobs, 1):
            print(f"{i:3d}. {ds:12s} {m}")
        return 0

    headers = [
        "dataset", "method", "success", "error",
        "n_samples", "n_features", "n_clusters",
        "v_measure", "adjusted_rand", "silhouette", "calinski_harabasz", "davies_bouldin", "inertia",
        "linear_sep_cv", "cv_score", "margin_score", "boundary_complexity",
    ]

    successes = 0
    fails = 0
    with open(csv_path, "w", newline="") as fcsv, open(log_path, "w") as flog:
        writer = csv.DictWriter(fcsv, fieldnames=headers, extrasaction='ignore')
        writer.writeheader()
        for i, (ds, m) in enumerate(jobs, 1):
            flog.write(f"\n[{datetime.now()}] {i}/{len(jobs)} {ds}/{m}\n")
            flog.flush()
            res = run_one(ds, m, str(args.results_dir), args.timeout)
            if res.get("success"):
                successes += 1
                writer.writerow(res)
                fcsv.flush()
                print(f"  ✅ {ds}/{m} ({successes}/{i})")
            else:
                fails += 1
                print(f"  ❌ {ds}/{m} ({fails}/{i})")
                if not args.continue_on_error:
                    print("Stopping due to error (use --continue-on-error to continue)")
                    break

    print("\n============================================================")
    print("🎯 BASELINE EXPERIMENTS COMPLETE")
    print("============================================================")
    print(f"Total runs: {len(jobs)}  Successes: {successes}  Fails: {fails}")
    print(f"CSV: {csv_path}")
    print(f"Log: {log_path}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


