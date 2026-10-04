from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "model_config.yaml"

def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    return {}

CONFIG = load_config()

# Standard Feature Columns extracted across all time series
FEATURE_COLUMNS = [
    # Personal Baselines
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_30",
    "rolling_median_7",
    "rolling_median_30",
    "rolling_std_7",
    "rolling_std_30",
    "rolling_min_30",
    "rolling_max_30",
    "load_factor_30",
    # Robust Dispersion & Scale (Learned from Main)
    "norm_iqr_30",
    "cv_30",
    # Deviations & Drops
    "deviation_7d_pct",
    "deviation_30d_pct",
    "deviation_30d_median_pct",
    "negative_deviation_pct",
    "positive_deviation_pct",
    "drop_ratio_mean",
    "drop_magnitude",
    "z_score_30",
    # Dynamics / Trends
    "change_1d",
    "change_7d",
    "rolling_slope_7",
    # Persistence & Streaks
    "consecutive_low_days",
    "consecutive_zero_days",
    "max_zero_streak_30",
    "low_consumption_ratio_30d",
    "cusum_drop_30",
    # Data Quality & Communication
    "missing_ratio_7d",
    "missing_ratio_30d",
    "zero_ratio_30d",
    "negative_value_flag",
    # Peer Group & Cross-Sectional
    "peer_median",
    "peer_deviation_pct",
    # Calendar & Cyclical
    "day_of_week",
    "is_weekend",
    "sin_day_of_year",
    "cos_day_of_year"
]