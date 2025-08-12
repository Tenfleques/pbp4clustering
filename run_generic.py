#!/usr/bin/env python3
import argparse
from typing import Any, Dict, Optional, List, Tuple, Callable

import numpy as np

# Loaders
from datasets.iris_loader import load_iris_matrices
from datasets.wdbc_loader import load_wdbc_matrices
from datasets.htru2_loader import load_htru2_matrices
from datasets.seeds_loader import load_seeds_matrices
from datasets.penguins_loader import load_penguins_matrices
from datasets.abalone_loader import load_abalone_matrices
from datasets.pima_loader import load_pima_matrices
from datasets.banknote_loader import load_banknote_matrices
from datasets.balance_loader import load_balance_matrices
from datasets.parkinsons_loader import load_parkinsons_matrices
from datasets.sonar_loader import load_sonar_matrices
from datasets.ionosphere_loader import load_ionosphere_matrices
from datasets.vehicle_loader import load_vehicle_matrices
from datasets.digits_loader import load_digits_matrices
from datasets.spectroscopy_coffee_loader import load_coffee_spectra
from datasets.wine_loader import load_wine_quality
from datasets.online_retail_loader import load_online_retail_matrices
from datasets.power_loader import load_household_power_matrices
from datasets.har_loader import (
    load_har_six_axis,
    load_har_axis_feature_format,
    load_har_axis_feature_columns,
)

# Standard pipeline
from src.cli_args import get_base_parser, add_data_dir_arg
from src.pipeline import run_clustering_pipeline
from src.utils import format_results, aggregate_rows_in_blocks
from src.base_runner import SimpleRunner
from src.sorting_functions import get_all_sorting_functions


SUPPORTED_DATASETS = {
    "iris": load_iris_matrices,
    "wdbc": load_wdbc_matrices,
    "htru2": load_htru2_matrices,
    "seeds": load_seeds_matrices,
    "penguins": load_penguins_matrices,
    "abalone": load_abalone_matrices,
    "pima": load_pima_matrices,
    "banknote": load_banknote_matrices,
    "balance": load_balance_matrices,
    "parkinsons": load_parkinsons_matrices,
    "sonar": load_sonar_matrices,
    "ionosphere": load_ionosphere_matrices,
    "vehicle": load_vehicle_matrices,
    "digits": load_digits_matrices,
    # extras
    "spectroscopy": load_coffee_spectra,
    "wine": load_wine_quality,
    "retail": load_online_retail_matrices,
    "power": load_household_power_matrices,
    "har": load_har_six_axis,
}


def build_parser() -> argparse.ArgumentParser:
    parser = get_base_parser("Run standardized PBP pipeline on many datasets")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(SUPPORTED_DATASETS.keys()),
        help="Dataset to load",
    )
    # Generic data dir (optional; most loaders have sensible defaults)
    add_data_dir_arg(parser, default=None, required=False)

    # Sorting function selection
    parser.add_argument(
        "--sort",
        type=str,
        choices=sorted(get_all_sorting_functions().keys()),
        default=None,
        help="Sorting function to use inside PBP (default: None/ascending)",
    )

    # Dataset-specific optional args (superset; unused are ignored)
    parser.add_argument("--option", choices=["A", "B"], default="B", help="Abalone option: A=(3x1), B=(2x4)")
    parser.add_argument("--mode", choices=["2x9", "3x6"], default="2x9", help="Vehicle mode: 2x9 or 3x6")
    parser.add_argument("--rows", type=int, choices=[2, 3], default=2, help="Digits rows bands: 2 or 3")
    parser.add_argument("--col-blocks", type=int, choices=[2, 4, 8], default=4, help="Digits column blocks")
    parser.add_argument("--measures", type=str, default=None, help="Digits measures, comma-separated (e.g., mean,std,nonzero_frac,entropy)")
    # Spectroscopy-specific: reduce rows to small blocks before PBP
    parser.add_argument("--row-blocks", type=int, choices=[2, 3], default=3, help="Spectroscopy row-block aggregation (2 or 3)")
    # Wine-specific: merge red and white into a single dataset or use red only
    parser.add_argument(
        "--wine-merge-types",
        dest="wine_merge_types",
        action="store_true",
        default=True,
        help="Wine: merge red and white datasets (default)",
    )
    parser.add_argument(
        "--no-wine-merge-types",
        dest="wine_merge_types",
        action="store_false",
        help="Wine: use red only (do not merge types)",
    )
    # Power-specific
    parser.add_argument("--power-days-limit", type=int, default=60, help="Power: number of days to include")
    parser.add_argument("--power-hours-block", type=int, default=4, help="Power: hours per block (divides 24)")
    # HAR-specific
    parser.add_argument(
        "--har-format",
        choices=["six_axis", "axis_feature_format", "axis_feature_columns"],
        default="six_axis",
        help="HAR: loader format",
    )
    parser.add_argument(
        "--har-include-body-acc",
        dest="har_include_body_acc",
        action="store_true",
        default=False,
        help="HAR: include body acceleration signals",
    )
    parser.add_argument(
        "--har-time-agg",
        default="mean",
        choices=[
            "mean","median","sum","min","max","std","var","rms","absmean","mad","iqr","energy","entropy"
        ],
        help="HAR axis_feature_columns: time aggregation",
    )
    parser.add_argument(
        "--plot-separators",
        dest="plot_separators",
        action="store_true",
        default=False,
        help="Overlay linear separator lines/planes (default on)",
    )
    parser.add_argument(
        "--no-plot-separators",
        dest="plot_separators",
        action="store_false",
        help="Disable separator overlays",
    )
    return parser


def call_loader(
    name: str,
    loader: Callable,
    data_dir: Optional[str],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if name == "abalone":
        if data_dir:
            return loader(data_dir=data_dir, option=args.option)
        return loader(option=args.option)
    if name == "vehicle":
        if data_dir:
            return loader(data_dir=data_dir, mode=args.mode)
        return loader(mode=args.mode)
    if name == "wine":
        if data_dir:
            return loader(data_dir=data_dir, merge_types=args.wine_merge_types)
        return loader(merge_types=args.wine_merge_types)
    if name == "power":
        if data_dir:
            return loader(data_dir=data_dir, days_limit=args.power_days_limit, hours_block=args.power_hours_block)
        return loader(days_limit=args.power_days_limit, hours_block=args.power_hours_block)
    if name == "har":
        # Route to specific HAR loader based on har_format
        if args.har_format == "six_axis":
            if data_dir:
                return load_har_six_axis(data_dir=data_dir)
            return load_har_six_axis()
        if args.har_format == "axis_feature_format":
            if data_dir:
                return load_har_axis_feature_format(data_dir=data_dir, include_body_acc=args.har_include_body_acc)
            return load_har_axis_feature_format(include_body_acc=args.har_include_body_acc)
        # axis_feature_columns
        if data_dir:
            return load_har_axis_feature_columns(
                data_dir=data_dir,
                include_body_acc=args.har_include_body_acc,
                time_agg=args.har_time_agg,
            )
        return load_har_axis_feature_columns(
            include_body_acc=args.har_include_body_acc,
            time_agg=args.har_time_agg,
        )
    if name == "digits":
        measures: Optional[List[str]] = None
        if args.measures:
            measures = [m.strip() for m in args.measures.split(",") if m.strip()]
        if data_dir:
            return loader(data_dir=data_dir, rows=args.rows, col_blocks=args.col_blocks, measures=measures)
        return loader(rows=args.rows, col_blocks=args.col_blocks, measures=measures)
    # Default signatures: with optional data_dir
    import inspect
    sig = inspect.signature(loader)
    if "data_dir" in sig.parameters and data_dir is not None:
        return loader(data_dir=data_dir)
    return loader()


def main():
    parser = build_parser()
    args = parser.parse_args()

    dataset = args.dataset
    loader = SUPPORTED_DATASETS[dataset]

    X_mats, y, meta = call_loader(dataset, loader, args.data_dir, args)
    # Apply dataset-specific preprocessing if needed
    if dataset == "spectroscopy":
        X_mats = aggregate_rows_in_blocks(X_mats, args.row_blocks)
    label_names = meta.get("labels_map") if isinstance(meta, dict) else None
    # Provide semantic axis names where available
    feature_names = None
    if isinstance(meta, dict):
        if isinstance(meta.get("feature_names"), list):
            feature_names = meta["feature_names"]
        elif isinstance(meta.get("cols"), list):
            feature_names = meta["cols"]

    # Use BaseRunner.SimpleRunner to leverage caching and unified pipeline
    loader_zero_arg = (lambda: (X_mats, y, meta))
    runner = SimpleRunner(dataset, loader_zero_arg)
    results = runner.run(args)

    # Neat table/text summary using standardized formatter
    print(
        format_results(
            results["metrics"],
            dataset_name=dataset,
            agg_func=args.agg,
            n_samples=results["X_pbp"].shape[0],
            n_features=results["X_pbp"].shape[1],
            n_clusters=len(set(int(v) for v in y)),
        )
    )


if __name__ == "__main__":
    main()


