"""
Vectorized, Time-Aware Feature Engineering Engine.
Extracts personal baselines, deviations, persistence run-lengths, peer comparisons,
and data quality proxies with ZERO look-ahead temporal leakage.
"""
from typing import List, Optional, Tuple, Dict, Any
import warnings
import numpy as np
import pandas as pd
from ml.config import FEATURE_COLUMNS

warnings.filterwarnings("ignore", category=RuntimeWarning)

def extract_features_at_timestep(
    matrix: np.ndarray,
    t: int,
    dates: List[pd.Timestamp],
    epsilon: float = 1e-5
) -> pd.DataFrame:
    """
    Extracts feature vector for all consumers at time step t.
    Uses strictly historical data (< t) for baselines, and current reading at t.
    """
    num_consumers, num_days = matrix.shape
    if t < 30 or t >= num_days:
        raise ValueError(f"Target day index t={t} must be between 30 and {num_days-1}")

    curr_date = dates[t]
    c_t = matrix[:, t].copy()
    
    # Identify corrupted negative readings at t
    c_t_neg = (c_t < 0)
    c_t[c_t_neg] = np.nan

    # Historical slices strictly before t: [t-30 : t]
    w30 = matrix[:, (t-30):t].copy()
    w14 = matrix[:, (t-14):t].copy()
    w7  = matrix[:, (t-7):t].copy()

    # Treat negative values in history as NaNs for consumption, but track them
    neg_history = (w30 < 0)
    w30_clean = np.where(neg_history, np.nan, w30)
    w14_clean = np.where(w14 < 0, np.nan, w14)
    w7_clean  = np.where(w7 < 0, np.nan, w7)

    # 1. Personal Baselines (Strictly backward-looking)
    mean_7  = np.nanmean(w7_clean, axis=1)
    mean_14 = np.nanmean(w14_clean, axis=1)
    mean_30 = np.nanmean(w30_clean, axis=1)

    median_7  = np.nanmedian(w7_clean, axis=1)
    median_30 = np.nanmedian(w30_clean, axis=1)

    std_7  = np.nanstd(w7_clean, axis=1)
    std_30 = np.nanstd(w30_clean, axis=1)

    min_30 = np.nanmin(w30_clean, axis=1)
    max_30 = np.nanmax(w30_clean, axis=1)

    # Safe fallback if consumer has all NaNs in window
    overall_pop_mean = np.nanmean(w30_clean)
    if np.isnan(overall_pop_mean):
        overall_pop_mean = 5.0

    mean_7  = np.where(np.isnan(mean_7), overall_pop_mean, mean_7)
    mean_14 = np.where(np.isnan(mean_14), overall_pop_mean, mean_14)
    mean_30 = np.where(np.isnan(mean_30), overall_pop_mean, mean_30)
    median_7  = np.where(np.isnan(median_7), mean_7, median_7)
    median_30 = np.where(np.isnan(median_30), mean_30, median_30)
    std_7  = np.where(np.isnan(std_7), 0.0, std_7)
    std_30 = np.where(np.isnan(std_30), 0.0, std_30)
    min_30 = np.where(np.isnan(min_30), 0.0, min_30)
    max_30 = np.where(np.isnan(max_30), mean_30, max_30)

    load_factor_30 = mean_30 / (max_30 + epsilon)

    # Effective current consumption: replace NaN with 0 for deviation calculation if needed,
    # or keep track of missingness
    c_eval = np.where(np.isnan(c_t), 0.0, c_t)

    # Robust Dispersion & Scale (from main)
    q75_30 = np.nanpercentile(w30_clean, 75, axis=1)
    q25_30 = np.nanpercentile(w30_clean, 25, axis=1)
    norm_iqr_30 = (q75_30 - q25_30) / (median_30 + epsilon)
    cv_30 = std_30 / (mean_30 + epsilon)

    # Deviation Features
    dev_7d_pct = (c_eval - mean_7) / (mean_7 + epsilon)
    dev_30d_pct = (c_eval - mean_30) / (mean_30 + epsilon)
    dev_30d_median_pct = (c_eval - median_30) / (median_30 + epsilon)

    neg_dev_pct = np.maximum(0.0, -dev_30d_pct)
    pos_dev_pct = np.maximum(0.0, dev_30d_pct)

    drop_ratio_mean = c_eval / (mean_30 + epsilon)
    drop_magnitude = np.maximum(0.0, mean_30 - c_eval)

    z_score_30 = np.clip((c_eval - mean_30) / (std_30 + epsilon), -5.0, 5.0)

    # 3. Dynamic Changes & 7-day Linear Trend Slope
    c_prev1 = np.where(np.isnan(matrix[:, t-1]), mean_7, matrix[:, t-1])
    c_prev7 = np.where(np.isnan(matrix[:, t-7]), mean_7, matrix[:, t-7])
    change_1d = c_eval - c_prev1
    change_7d = c_eval - c_prev7

    # Vectorized linear regression slope over [t-6, ..., t] (7 days, x in 0..6)
    w_trend = np.column_stack([w7_clean[:, 1:], c_eval])  # shape (N, 7)
    x = np.arange(7)
    x_mean = 3.0
    x_diff = x - x_mean
    x_denom = np.sum(x_diff ** 2)  # = 28.0
    # Center y around its row mean
    y_mean = np.nanmean(w_trend, axis=1, keepdims=True)
    y_diff = np.where(np.isnan(w_trend), 0.0, w_trend - y_mean)
    rolling_slope_7 = np.sum(y_diff * x_diff, axis=1) / x_denom

    # 4. Persistence Run-Lengths & Streaks
    # Consecutive days below 40% of 30-day baseline ending at t
    thresh_low = 0.40 * mean_30
    w_recent_14 = np.column_stack([w14_clean, c_eval])
    is_low = (w_recent_14 < thresh_low[:, None])
    
    consec_low = np.zeros(num_consumers, dtype=int)
    for day_i in range(is_low.shape[1] - 1, -1, -1):
        active = is_low[:, day_i]
        consec_low = np.where(active & (consec_low == (is_low.shape[1] - 1 - day_i)), consec_low + 1, consec_low)

    # Consecutive zeros ending at t
    is_zero = (w_recent_14 == 0)
    consec_zero = np.zeros(num_consumers, dtype=int)
    for day_i in range(is_zero.shape[1] - 1, -1, -1):
        active = is_zero[:, day_i]
        consec_zero = np.where(active & (consec_zero == (is_zero.shape[1] - 1 - day_i)), consec_zero + 1, consec_zero)

    # Maximum zero streak across 30 days
    is_zero_30 = (w30_clean == 0.0) | np.isnan(w30_clean)
    cur_streak = np.zeros(num_consumers, dtype=float)
    max_zero_streak_30 = np.zeros(num_consumers, dtype=float)
    for col_idx in range(is_zero_30.shape[1]):
        cur_streak = np.where(is_zero_30[:, col_idx], cur_streak + 1.0, 0.0)
        max_zero_streak_30 = np.maximum(max_zero_streak_30, cur_streak)

    # Low consumption ratio over past 30 days
    is_low_30 = (w30_clean < thresh_low[:, None])
    low_ratio_30d = np.nanmean(is_low_30, axis=1)
    low_ratio_30d = np.where(np.isnan(low_ratio_30d), 0.0, low_ratio_30d)

    # CUSUM Cumulative Deficit (Change-Point Detector for stealth micro-theft)
    deficit_30 = np.maximum(0.0, mean_30[:, None] - np.nan_to_num(w30_clean, nan=0.0))
    cusum_drop_30 = np.sum(deficit_30, axis=1) / (mean_30 * 30.0 + epsilon)

    # 5. Data Quality & Communication Signals
    missing_7d  = np.mean(np.isnan(w7), axis=1)
    missing_30d = np.mean(np.isnan(w30), axis=1)
    zero_ratio_30d = np.mean(w30 == 0, axis=1)
    neg_flag = np.any(neg_history, axis=1).astype(float)

    # 6. Peer Group & Cross-Sectional Comparison
    peer_med = float(np.nanmedian(c_eval))
    peer_dev_pct = (c_eval - peer_med) / (peer_med + epsilon)

    # 7. Calendar and Cyclical Harmonics
    dow = curr_date.dayofweek
    is_wknd = 1.0 if dow >= 5 else 0.0
    day_of_year = curr_date.dayofyear
    sin_doy = np.sin(2 * np.pi * day_of_year / 365.25)
    cos_doy = np.cos(2 * np.pi * day_of_year / 365.25)

    data_dict = {
        "rolling_mean_7": mean_7,
        "rolling_mean_14": mean_14,
        "rolling_mean_30": mean_30,
        "rolling_median_7": median_7,
        "rolling_median_30": median_30,
        "rolling_std_7": std_7,
        "rolling_std_30": std_30,
        "rolling_min_30": min_30,
        "rolling_max_30": max_30,
        "load_factor_30": load_factor_30,
        "norm_iqr_30": norm_iqr_30,
        "cv_30": cv_30,
        "deviation_7d_pct": dev_7d_pct,
        "deviation_30d_pct": dev_30d_pct,
        "deviation_30d_median_pct": dev_30d_median_pct,
        "negative_deviation_pct": neg_dev_pct,
        "positive_deviation_pct": pos_dev_pct,
        "drop_ratio_mean": drop_ratio_mean,
        "drop_magnitude": drop_magnitude,
        "z_score_30": z_score_30,
        "change_1d": change_1d,
        "change_7d": change_7d,
        "rolling_slope_7": rolling_slope_7,
        "consecutive_low_days": consec_low,
        "consecutive_zero_days": consec_zero,
        "max_zero_streak_30": max_zero_streak_30,
        "low_consumption_ratio_30d": low_ratio_30d,
        "cusum_drop_30": cusum_drop_30,
        "missing_ratio_7d": missing_7d,
        "missing_ratio_30d": missing_30d,
        "zero_ratio_30d": zero_ratio_30d,
        "negative_value_flag": neg_flag,
        "peer_median": np.full(num_consumers, peer_med),
        "peer_deviation_pct": peer_dev_pct,
        "day_of_week": np.full(num_consumers, dow),
        "is_weekend": np.full(num_consumers, is_wknd),
        "sin_day_of_year": np.full(num_consumers, sin_doy),
        "cos_day_of_year": np.full(num_consumers, cos_doy),
    }

    df_feats = pd.DataFrame(data_dict, columns=FEATURE_COLUMNS)
    # Ensure zero NaNs or Infs
    df_feats = df_feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df_feats

def generate_training_feature_dataset(
    consumer_ids: np.ndarray,
    dates: List[pd.Timestamp],
    matrix: np.ndarray,
    labels: np.ndarray,
    timesteps: Optional[List[int]] = None
) -> Tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """
    Extracts time-aware features across multiple sample timesteps to build
    a comprehensive training/evaluation feature matrix.
    """
    num_consumers, num_days = matrix.shape
    if timesteps is None:
        # Default: sample points spaced across the year (e.g. days 60, 120, 180, 240, 300, 360)
        timesteps = [t for t in range(45, num_days, 45)]

    print(f"[*] Extracting time-aware features at {len(timesteps)} temporal checkpoints: {timesteps}")
    all_dfs = []
    all_labels = []
    all_cids = []
    all_dates = []

    for t in timesteps:
        df_t = extract_features_at_timestep(matrix, t, dates)
        all_dfs.append(df_t)
        all_labels.append(labels)
        all_cids.append(consumer_ids)
        all_dates.extend([dates[t]] * num_consumers)

    X = pd.concat(all_dfs, ignore_index=True)
    y = np.concatenate(all_labels)
    cids = np.concatenate(all_cids)
    dates_col = pd.Series(all_dates, name="timestamp")

    print(f"[+] Feature matrix created successfully: {X.shape[0]:,} rows × {X.shape[1]} features.")
    return X, y, pd.Series(cids, name="consumer_id")
