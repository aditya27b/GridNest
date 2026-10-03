"""
Dataset Audit Module for Energy Intelligence Pipeline.
Performs rigorous structural, statistical, and temporal validation on raw electricity data.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

def audit_dataset(file_path: str, output_dir: str = "reports") -> dict:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {file_path}")

    print(f"\n=======================================================")
    print(f" AUDITING DATASET: {path.name}")
    print(f"=======================================================")

    # 1. Inspect first few lines and check header
    df_preview = pd.read_csv(path, nrows=5)
    cols = df_preview.columns.tolist()

    # Identify target and ID columns
    id_col = "CONS_NO" if "CONS_NO" in cols else cols[0]
    target_col = None
    for cand in ["CHK_STATE", "FLAG", "label", "theft", "target"]:
        if cand in cols:
            target_col = cand
            break

    # Check for row-0 index anomaly (common in Electricity_Theft_Data.csv)
    df_check_first = pd.read_csv(path, nrows=2)
    has_header_index_row = False
    if pd.isna(df_check_first.iloc[0][id_col]) or str(df_check_first.iloc[0][id_col]).strip() == "":
        has_header_index_row = True
        print(f"[!] DETECTED MALFORMED ROW 0 (index header). Will skip during analysis.")

    # Load full dataset with appropriate skip if needed
    if has_header_index_row:
        df = pd.read_csv(path, skiprows=[1])
    else:
        df = pd.read_csv(path)

    total_rows, total_cols = df.shape
    print(f"[*] Total Valid Rows:    {total_rows:,}")
    print(f"[*] Total Columns:       {total_cols:,}")

    # ID validation
    unique_ids = df[id_col].nunique(dropna=True)
    null_ids = df[id_col].isna().sum()
    duplicate_ids = total_rows - unique_ids
    print(f"[*] Unique Consumers:    {unique_ids:,}")
    print(f"[*] Null IDs:            {null_ids}")
    print(f"[*] Duplicate IDs:       {duplicate_ids}")

    # Target analysis
    target_dist = {}
    imbalance_ratio = None
    if target_col and target_col in df.columns:
        counts = df[target_col].value_counts(dropna=False).to_dict()
        target_dist = {str(k): int(v) for k, v in counts.items()}
        print(f"[*] Target Column:       '{target_col}'")
        print(f"    Distribution:        {target_dist}")
        if 0.0 in counts and 1.0 in counts:
            imbalance_ratio = round(counts[0.0] / max(1, counts[1.0]), 2)
            print(f"    Class Imbalance:     {imbalance_ratio}:1 (Normal:Theft)")
        elif 0 in counts and 1 in counts:
            imbalance_ratio = round(counts[0] / max(1, counts[1]), 2)
            print(f"    Class Imbalance:     {imbalance_ratio}:1 (Normal:Theft)")
    else:
        print("[!] No standard target column found. (Unsupervised mode)")

    # Date Columns Analysis
    date_cols = [c for c in df.columns if c not in [id_col, target_col]]
    print(f"[*] Date Reading Columns: {len(date_cols)}")

    # Check chronological ordering
    sample_dates = date_cols[:10]
    parsed_dates = []
    date_format = None
    for fmt in ["%d-%m-%y", "%Y/%m/%d", "%Y-%m-%d", "%d/%m/%Y"]:
        try:
            parsed = pd.to_datetime(sample_dates, format=fmt)
            date_format = fmt
            parsed_dates = pd.to_datetime(date_cols, format=fmt)
            break
        except Exception:
            continue

    is_chronological = False
    date_range_str = "Unknown"
    if len(parsed_dates) == len(date_cols):
        is_chronological = parsed_dates.is_monotonic_increasing
        min_date = parsed_dates.min().strftime('%Y-%m-%d')
        max_date = parsed_dates.max().strftime('%Y-%m-%d')
        date_range_str = f"{min_date} to {max_date} ({len(date_cols)} days)"
        print(f"[*] Detected Date Format: {date_format}")
        print(f"[*] Date Range:          {date_range_str}")
        print(f"[*] Chronological Order: {'YES (Valid)' if is_chronological else 'NO (ALPHABETICAL TRAP DETECTED - REQUIRES RESORTING)'}")

    # Numerical & Consumption Value Statistics
    print(f"[*] Computing matrix-level consumption statistics...")
    vals = df[date_cols].apply(pd.to_numeric, errors='coerce').values
    total_cells = vals.size
    missing_cells = int(np.isnan(vals).sum())
    missing_pct = round(missing_cells / total_cells * 100, 2)
    zero_cells = int((vals == 0).sum())
    zero_pct = round(zero_cells / total_cells * 100, 2)
    negative_cells = int((vals < 0).sum())

    non_null_vals = vals[~np.isnan(vals)]
    min_val = float(np.min(non_null_vals)) if len(non_null_vals) > 0 else 0.0
    max_val = float(np.max(non_null_vals)) if len(non_null_vals) > 0 else 0.0
    mean_val = float(np.mean(non_null_vals)) if len(non_null_vals) > 0 else 0.0
    median_val = float(np.median(non_null_vals)) if len(non_null_vals) > 0 else 0.0

    print(f"    - Missing Cells:     {missing_cells:,} ({missing_pct}%)")
    print(f"    - Flat Zero Cells:   {zero_cells:,} ({zero_pct}%)")
    print(f"    - Corrupted Negative Cells: {negative_cells:,}")
    print(f"    - Min Consumption:   {min_val}")
    print(f"    - Max Consumption:   {max_val}")
    print(f"    - Mean Consumption:  {mean_val:.2f} kWh")
    print(f"    - Median Consumption:{median_val:.2f} kWh")

    # Generate Audit Summary Dictionary
    report = {
        "dataset_name": path.name,
        "total_rows": total_rows,
        "total_columns": total_cols,
        "id_column": id_col,
        "target_column": target_col,
        "target_distribution": target_dist,
        "imbalance_ratio": imbalance_ratio,
        "num_date_columns": len(date_cols),
        "date_range": date_range_str,
        "date_format": date_format,
        "is_chronologically_sorted": is_chronological,
        "has_header_index_row": has_header_index_row,
        "matrix_stats": {
            "total_cells": total_cells,
            "missing_cells": missing_cells,
            "missing_percentage": missing_pct,
            "zero_cells": zero_cells,
            "zero_percentage": zero_pct,
            "negative_cells": negative_cells,
            "min_kwh": min_val,
            "max_kwh": max_val,
            "mean_kwh": round(mean_val, 2),
            "median_kwh": round(median_val, 2)
        }
    }

    # Save to disk
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / f"audit_{path.stem}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[+] Audit report saved successfully to: {report_file}")
    print(f"=======================================================\n")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit electricity dataset for anomaly detection pipeline.")
    parser.add_argument("--data", type=str, default="data/Electricity_Theft_Data.csv", help="Path to raw CSV dataset")
    parser.add_argument("--output", type=str, default="reports", help="Directory to save audit report")
    args = parser.parse_args()
    audit_dataset(args.data, args.output)
