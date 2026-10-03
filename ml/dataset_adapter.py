"""
Dataset Adapter Module.
Loads, normalizes, and chronologically orders wide electricity time-series datasets.
"""
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import pandas as pd
import numpy as np

def load_clean_timeseries(
    file_path: str,
    max_consumers: Optional[int] = None
) -> Tuple[np.ndarray, List[pd.Timestamp], np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Loads wide time-series dataset, drops index headers, sorts columns chronologically,
    and returns (consumer_ids, sorted_dates, consumption_matrix, labels, metadata).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Inspect first 2 rows to check for index header
    df_head = pd.read_csv(path, nrows=2)
    id_col = "CONS_NO" if "CONS_NO" in df_head.columns else df_head.columns[0]
    
    skiprows = [1] if pd.isna(df_head.iloc[0][id_col]) or str(df_head.iloc[0][id_col]).strip() == "" else None

    # Load data
    df = pd.read_csv(path, skiprows=skiprows, nrows=max_consumers)
    
    # Identify target column
    target_col = None
    for cand in ["CHK_STATE", "FLAG", "label", "theft", "target"]:
        if cand in df.columns:
            target_col = cand
            break

    # Consumer IDs
    consumer_ids = df[id_col].astype(str).values

    # Target labels
    if target_col and target_col in df.columns:
        labels = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int).values
    else:
        labels = np.zeros(len(df), dtype=int)

    # Date columns
    raw_date_cols = [c for c in df.columns if c not in [id_col, target_col]]

    # Parse dates chronologically
    date_objs = None
    for fmt in ["%d-%m-%y", "%Y/%m/%d", "%Y-%m-%d", "%d/%m/%Y"]:
        try:
            parsed = pd.to_datetime(raw_date_cols, format=fmt)
            date_objs = parsed
            break
        except Exception:
            continue

    if date_objs is None:
        # Fallback to general parser
        date_objs = pd.to_datetime(raw_date_cols, errors="coerce")

    # Chronological sort order
    sort_order = np.argsort(date_objs.values)
    sorted_date_objs = [pd.Timestamp(d) for d in date_objs.values[sort_order]]
    sorted_date_cols = [raw_date_cols[i] for i in sort_order]

    # Extract consumption values matrix in chronological order
    consumption_matrix = df[sorted_date_cols].apply(pd.to_numeric, errors="coerce").values

    metadata = {
        "dataset_name": path.name,
        "num_consumers": len(consumer_ids),
        "num_days": len(sorted_date_objs),
        "start_date": sorted_date_objs[0].strftime("%Y-%m-%d"),
        "end_date": sorted_date_objs[-1].strftime("%Y-%m-%d"),
        "target_col": target_col,
        "positive_labels": int((labels == 1).sum()),
        "negative_labels": int((labels == 0).sum())
    }

    return consumer_ids, sorted_date_objs, consumption_matrix, labels, metadata
