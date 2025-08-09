#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import numpy as np


CONFIG_COLS = [
    "format",
    "include_body_acc",
    "time_agg",
    "rows",
    "pbp_agg",
]


def sanitize(val: Any) -> str:
    if pd.isna(val):
        return "na"
    if isinstance(val, bool):
        return "true" if val else "false"
    return str(val).replace("/", "-").replace(" ", "_")


def make_results_dir(base_dir: Path, row: pd.Series) -> Path:
    fmt = sanitize(row["format"])
    pbp = sanitize(row.get("pbp_agg", "sum"))
    rows = sanitize(row.get("rows", "na"))
    time_agg = sanitize(row.get("time_agg", "na"))
    inc_body = sanitize(row.get("include_body_acc", False))
    sub = f"fmt={fmt}/pbp={pbp}/rows={rows}/tagg={time_agg}/bodyacc={inc_body}"
    return base_dir / sub


def pick_top_configs(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    # Filter multiclass only and valid separability values
    df_mc = df[df["label_mode"] == "multiclass"].copy()
    if df_mc.empty:
        return df_mc

    # Deduplicate configs: keep best (max) linear_sep_cv per config
    df_mc["linear_sep_cv"] = pd.to_numeric(df_mc["linear_sep_cv"], errors="coerce")
    df_mc["v_measure"] = pd.to_numeric(df_mc["v_measure"], errors="coerce")
    df_mc["calinski_harabasz"] = pd.to_numeric(df_mc.get("calinski_harabasz", np.nan), errors="coerce")

    # idx of best per group
    grp = df_mc.groupby(CONFIG_COLS, dropna=False)
    idx = grp["linear_sep_cv"].idxmax()
    best = df_mc.loc[idx].dropna(subset=["linear_sep_cv"])  # type: ignore

    # Sort by linear separability, then v-measure, then calinski
    best = best.sort_values(by=["linear_sep_cv", "v_measure", "calinski_harabasz"], ascending=False)
    return best.head(top_n)


def pick_top_per_format(df: pd.DataFrame, per_format_n: int = 5) -> pd.DataFrame:
    out = []
    for fmt in ["six_axis", "axis_feature_format", "axis_feature_columns"]:
        sub = df[df["format"] == fmt]
        out.append(sub.head(per_format_n))
    return pd.concat(out, axis=0, ignore_index=True)


def write_report(report_path: Path, top_overall: pd.DataFrame, top_per_fmt: pd.DataFrame) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("## HAR transformations and evaluation summary")
    lines.append("")
    lines.append("**Transformations evaluated**:")
    lines.append("- Row reduction: optional PCA to 2 or 3 rows before PBP")
    lines.append("- Axis-feature formats: (a) 3-row x/y/z with concatenated feature windows; (b) 3-row with standalone feature columns via time aggregation (mean/median/sum/min/max/std/var/rms/absmean/mad/iqr/energy/entropy)")
    lines.append("- PBP aggregation functions: sum, mean, median, trimmed_mean, rms, adaptive, robust_adaptive, std, var, max, min, iqr, range, entropy, gini")
    lines.append("- Metrics: linear separability (LinearSVC CV), ARI, V-measure, silhouette, Calinski-Harabasz, Davies-Bouldin, inertia, CV accuracy (KNN-5), margin, boundary complexity")
    lines.append("")
    lines.append("**Selection objective**: prioritize multiclass linear separability; break ties with V-measure and Calinski-Harabasz.")
    lines.append("")

    def to_list_table(df: pd.DataFrame, title: str) -> List[str]:
        cols = [
            "format", "pbp_agg", "rows", "time_agg", "include_body_acc",
            "linear_sep_cv", "v_measure", "adjusted_rand", "silhouette", "calinski_harabasz",
        ]
        present_cols = [c for c in cols if c in df.columns]
        rows: List[str] = []
        rows.append(f"### {title}")
        rows.append("")
        rows.append("| " + " | ".join(present_cols) + " |")
        rows.append("|" + "|".join(["---"] * len(present_cols)) + "|")
        for _, r in df[present_cols].iterrows():
            vals = [str(r[c]) for c in present_cols]
            rows.append("| " + " | ".join(vals) + " |")
        rows.append("")
        return rows

    lines += to_list_table(top_overall, "Top overall (by linear separability)")
    lines += to_list_table(top_per_fmt, "Top per format (by linear separability)")

    report_path.write_text("\n".join(lines))


def write_plot_script(script_path: Path, base_results_dir: Path, rows: pd.DataFrame, data_dir: str) -> None:
    script_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = ["#!/usr/bin/env bash", "set -euo pipefail"]
    for _, r in rows.iterrows():
        res_dir = make_results_dir(base_results_dir, r)
        cmd = [
            "python", "clustering/run_har.py",
            "--data-dir", data_dir,
            "--results-dir", str(res_dir),
            "--agg", str(r.get("pbp_agg", "sum")),
        ]
        fmt = r["format"]
        if fmt == "axis_feature_format":
            cmd.append("--axis-feature-format")
        elif fmt == "axis_feature_columns":
            cmd.append("--axis-feature-columns")
            if pd.notna(r.get("time_agg")) and str(r.get("time_agg")) != "na":
                cmd += ["--time-agg", str(r.get("time_agg"))]
        # include body acc
        if bool(r.get("include_body_acc", False)):
            cmd.append("--include-body-acc")
        # rows reduction
        if pd.notna(r.get("rows")) and str(r.get("rows")) != "na":
            cmd += ["--rows", str(int(r.get("rows")))]
        lines.append(" ".join(cmd))

    script_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Summarize HAR results and emit a plot script for top configs")
    parser.add_argument("--csv", default="./results/har_eval_stream.csv", help="Path to results CSV")
    parser.add_argument("--report", default="./results/har_report.md", help="Path to write markdown report")
    parser.add_argument("--script", default="./run_top_plots.sh", help="Path to write bash script for plots")
    parser.add_argument("--plots-dir", default="./results/plots", help="Base directory for plot outputs")
    parser.add_argument("--data-dir", default="./data/har", help="HAR data directory for runner")
    parser.add_argument("--top-n", type=int, default=15, help="Number of top overall configs to include")
    parser.add_argument("--per-format-n", type=int, default=5, help="Number of top per-format configs to include")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    top_overall = pick_top_configs(df, top_n=args.top_n)
    top_per_fmt = pick_top_per_format(top_overall, per_format_n=args.per_format_n)

    write_report(Path(args.report), top_overall, top_per_fmt)
    # Combine selections (unique by config) for plotting
    sel = pd.concat([top_overall, top_per_fmt], axis=0)
    sel = sel.drop_duplicates(subset=CONFIG_COLS, keep="first")
    write_plot_script(Path(args.script), Path(args.plots_dir), sel, args.data_dir)
    print(f"Wrote report to {args.report} and script to {args.script}")


if __name__ == "__main__":
    main()


