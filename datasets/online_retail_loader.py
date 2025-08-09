from pathlib import Path
from typing import Tuple, Dict, Any
import io

import numpy as np
import pandas as pd
import requests

UCI_ONLINE_RETAIL_II_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx"


FEATURE_ROWS = [
    "total_quantity",
    "total_value",
    "unique_products",
    "num_invoices",
    "avg_unit_price",
    "cancel_ratio",
]


def _download_excel(url: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
    return out_path


def _read_retail_excel(path: Path) -> pd.DataFrame:
    # Online Retail II has two sheets for 2009-2010 and 2010-2011
    xls = pd.ExcelFile(path)
    frames = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    # Standardize column names
    # Actual columns: ['Invoice','StockCode','Description','Quantity','InvoiceDate','Price','Customer ID','Country']
    df = df.rename(columns={
        'Invoice': 'invoiceno',
        'StockCode': 'stockcode',
        'Quantity': 'quantity',
        'InvoiceDate': 'invoicedate',
        'Price': 'unitprice',
        'Customer ID': 'customerid',
        'Country': 'country',
    })
    # Ensure types
    df = df.dropna(subset=["invoiceno", "stockcode", "quantity", "invoicedate", "unitprice", "customerid", "country"], how="any")
    df["invoicedate"] = pd.to_datetime(df["invoicedate"])  # type: ignore
    df["month"] = df["invoicedate"].dt.to_period("M").astype(str)
    df["value"] = df["quantity"] * df["unitprice"]
    df["is_cancel"] = df["invoiceno"].astype(str).str.startswith("C").astype(int)
    return df


def _build_customer_monthly_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    # Keep top 12 calendar months present to align columns
    months = sorted(df["month"].unique().tolist())
    if len(months) > 12:
        months = months[:12]
    month_index = {m: i for i, m in enumerate(months)}

    customers = df["customerid"].unique().tolist()
    countries = df.drop_duplicates(["customerid"])  # country per customer (most frequent)
    cust_to_country = countries.set_index("customerid")["country"].to_dict()

    num_customers = len(customers)
    m = len(FEATURE_ROWS)
    n = len(months)
    X = np.zeros((num_customers, m, n), dtype=np.float32)
    y_labels = []

    # Precompute per customer-month aggregates
    grouped = df[df["month"].isin(months)].groupby(["customerid", "month"])  # type: ignore
    agg = grouped.agg(
        total_quantity=("quantity", "sum"),
        total_value=("value", "sum"),
        unique_products=("stockcode", "nunique"),
        num_invoices=("invoiceno", "nunique"),
        avg_unit_price=("unitprice", "mean"),
        cancel_ratio=("is_cancel", "mean"),
    ).reset_index()

    cust_month_to_row = {(int(r.customerid), r.month): r for _, r in agg.iterrows()}

    for idx, cust in enumerate(customers):
        country = cust_to_country.get(cust, "Unknown")
        y_labels.append(country)
        for mth in months:
            r = cust_month_to_row.get((int(cust), mth))
            if r is None:
                continue
            col = month_index[mth]
            X[idx, 0, col] = float(r.total_quantity)
            X[idx, 1, col] = float(r.total_value)
            X[idx, 2, col] = float(r.unique_products)
            X[idx, 3, col] = float(r.num_invoices)
            X[idx, 4, col] = float(r.avg_unit_price)
            X[idx, 5, col] = float(r.cancel_ratio)

    # Encode country labels as integers
    unique_countries = sorted(set(y_labels))
    country_to_id = {c: i for i, c in enumerate(unique_countries)}
    y = np.array([country_to_id[c] for c in y_labels], dtype=int)

    metadata = {
        "n_samples": num_customers,
        "matrix_shape": (m, n),
        "rows": FEATURE_ROWS,
        "cols": months,
        "label_name": "country",
        "label_map": country_to_id,
    }
    return X, y, metadata


def load_online_retail_matrices(data_dir: str = "./data/retail") -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load UCI Online Retail II and build per-customer matrices (6×12) with country labels.

    Returns:
        X: (N, 6, 12)
        y: (N,) int, encoded country id
        metadata: dict
    """
    out_path = _download_excel(UCI_ONLINE_RETAIL_II_URL, Path(data_dir) / "online_retail_II.xlsx")
    df = _read_retail_excel(out_path)
    return _build_customer_monthly_matrix(df) 