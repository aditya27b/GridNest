"""
Feature Engineering for the Realistic Multi-Sensor Dataset.
Exploits: recorded_kwh, voltage, current, power_factor, peak_share,
tamper_flag, fault_flag, heartbeat_ok, consumer metadata, and transformer energy.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def build_realistic_features(
    readings_path: str = "data/realistic/readings.csv.gz",
    consumers_path: str = "data/realistic/consumers.csv",
    transformer_path: str = "data/realistic/transformer_energy.csv",
    train_cutoff: str = "2026-02-01",
) -> pd.DataFrame:
    """
    Build comprehensive feature matrix for every consumer using:
    - Personal consumption baselines & deviations
    - Meter vs physics discrepancy (implied energy from V*I*PF vs recorded)
    - Persistence run-lengths
    - Communication health (heartbeat, missing readings)
    - Meter hardware signals (tamper_flag, fault_flag)
    - Peer group comparisons
    - Transformer-level loss analysis
    - Load shape changes (peak_share)
    """
    print("[*] Loading realistic dataset files...")
    df_r = pd.read_csv(readings_path)
    df_c = pd.read_csv(consumers_path)
    df_t = pd.read_csv(transformer_path)

    df_r["date"] = pd.to_datetime(df_r["date"])
    df_t["date"] = pd.to_datetime(df_t["date"])

    # Merge consumer metadata
    df_r = df_r.merge(df_c, on="consumer_id", how="left")

    # --- Physics-Based Discrepancy ---
    # Implied energy from voltage * current * power_factor * 24h / 1000
    df_r["implied_kwh"] = df_r["voltage_v"] * df_r["current_a"] * df_r["power_factor"] * 24.0 / 1000.0
    df_r["discrepancy_kwh"] = df_r["implied_kwh"] - df_r["recorded_kwh"]
    df_r["discrepancy_ratio"] = df_r["discrepancy_kwh"] / (df_r["implied_kwh"] + 1e-5)

    # --- Define temporal windows ---
    cutoff = pd.Timestamp(train_cutoff)
    # Baseline: first 5 months of stable readings
    df_base = df_r[df_r["date"] < pd.Timestamp("2025-06-01")].copy()
    # Evaluation: recent window where anomalies manifest
    df_eval = df_r[df_r["date"] >= cutoff].copy()
    # Mid-history for trend comparison
    df_mid = df_r[(df_r["date"] >= pd.Timestamp("2025-09-01")) & (df_r["date"] < cutoff)].copy()

    cids = df_c["consumer_id"].values
    feats = pd.DataFrame({"consumer_id": cids})

    print("[*] Computing personal baseline features...")
    # --- Personal Baselines (from clean early history) ---
    base_g = df_base.groupby("consumer_id")["recorded_kwh"]
    feats["base_mean"] = feats["consumer_id"].map(base_g.mean()).fillna(10.0)
    feats["base_median"] = feats["consumer_id"].map(base_g.median()).fillna(10.0)
    feats["base_std"] = feats["consumer_id"].map(base_g.std()).fillna(1.0)
    feats["base_min"] = feats["consumer_id"].map(base_g.min()).fillna(0.0)
    feats["base_max"] = feats["consumer_id"].map(base_g.max()).fillna(20.0)
    feats["base_peak_share"] = feats["consumer_id"].map(
        df_base.groupby("consumer_id")["peak_share"].median()
    ).fillna(0.35)

    print("[*] Computing recent evaluation window features...")
    # --- Recent Window Features ---
    eval_g = df_eval.groupby("consumer_id")
    feats["recent_mean"] = feats["consumer_id"].map(eval_g["recorded_kwh"].mean()).fillna(0.0)
    feats["recent_median"] = feats["consumer_id"].map(eval_g["recorded_kwh"].median()).fillna(0.0)
    feats["recent_std"] = feats["consumer_id"].map(eval_g["recorded_kwh"].std()).fillna(0.0)
    feats["recent_min"] = feats["consumer_id"].map(eval_g["recorded_kwh"].min()).fillna(0.0)

    # --- Deviation from personal baseline ---
    feats["deviation_mean_pct"] = (feats["recent_mean"] - feats["base_mean"]) / (feats["base_mean"] + 1e-5)
    feats["deviation_median_pct"] = (feats["recent_median"] - feats["base_median"]) / (feats["base_median"] + 1e-5)
    feats["negative_deviation"] = np.maximum(0.0, -feats["deviation_mean_pct"])
    feats["positive_deviation"] = np.maximum(0.0, feats["deviation_mean_pct"])
    feats["z_score_recent"] = np.clip(
        (feats["recent_mean"] - feats["base_mean"]) / (feats["base_std"] + 1e-5), -5.0, 5.0
    )

    print("[*] Computing physics discrepancy features (V*I*PF vs recorded)...")
    # --- Meter vs Physics Discrepancy (KEY for theft detection) ---
    feats["disc_ratio_mean"] = feats["consumer_id"].map(
        eval_g["discrepancy_ratio"].mean()
    ).fillna(0.0)
    feats["disc_ratio_max"] = feats["consumer_id"].map(
        eval_g["discrepancy_ratio"].max()
    ).fillna(0.0)
    feats["disc_ratio_median"] = feats["consumer_id"].map(
        eval_g["discrepancy_ratio"].median()
    ).fillna(0.0)
    feats["disc_kwh_mean"] = feats["consumer_id"].map(
        eval_g["discrepancy_kwh"].mean()
    ).fillna(0.0)
    feats["disc_ratio_std"] = feats["consumer_id"].map(
        eval_g["discrepancy_ratio"].std()
    ).fillna(0.0)

    # Positive discrepancy = implied > recorded = possible theft
    feats["disc_positive_days"] = feats["consumer_id"].map(
        df_eval.groupby("consumer_id")["discrepancy_ratio"].apply(lambda x: (x > 0.05).sum())
    ).fillna(0.0)
    feats["disc_positive_ratio"] = feats["disc_positive_days"] / (
        feats["consumer_id"].map(eval_g.size()).fillna(1.0) + 1e-5
    )

    print("[*] Computing persistence run-length features...")
    # --- Persistence: consecutive days of anomalous drop ---
    def max_consecutive_below(series, threshold_ratio=0.50):
        below = (series < threshold_ratio).astype(int)
        runs = below * (below.groupby((below != below.shift()).cumsum()).cumcount() + 1)
        return runs.max() if len(runs) > 0 else 0

    # Ratio of recent to baseline per day
    base_map = dict(zip(feats["consumer_id"], feats["base_mean"]))
    df_eval_copy = df_eval.copy()
    df_eval_copy["ratio_to_base"] = df_eval_copy["recorded_kwh"] / df_eval_copy["consumer_id"].map(base_map).replace(0, 1e-5)

    consec_low = df_eval_copy.groupby("consumer_id")["ratio_to_base"].apply(
        lambda x: max_consecutive_below(x, 0.50)
    )
    feats["consecutive_low_days"] = feats["consumer_id"].map(consec_low).fillna(0.0)

    consec_zero = df_eval_copy.groupby("consumer_id")["recorded_kwh"].apply(
        lambda x: max_consecutive_below(x.fillna(0), 0.01)
    )
    feats["consecutive_zero_days"] = feats["consumer_id"].map(consec_zero).fillna(0.0)

    # Low consumption ratio (fraction of eval days below 50% of baseline)
    low_ratio = df_eval_copy.groupby("consumer_id")["ratio_to_base"].apply(
        lambda x: (x < 0.50).mean()
    )
    feats["low_consumption_ratio"] = feats["consumer_id"].map(low_ratio).fillna(0.0)

    print("[*] Computing communication & meter health features...")
    # --- Communication Health ---
    feats["missing_ratio"] = feats["consumer_id"].map(
        eval_g["recorded_kwh"].apply(lambda x: x.isna().mean())
    ).fillna(0.0)
    feats["heartbeat_fail_ratio"] = feats["consumer_id"].map(
        eval_g["heartbeat_ok"].apply(lambda x: (x == 0).mean())
    ).fillna(0.0)
    feats["heartbeat_fail_count"] = feats["consumer_id"].map(
        eval_g["heartbeat_ok"].apply(lambda x: (x == 0).sum())
    ).fillna(0.0)

    # --- Meter Hardware Signals ---
    feats["tamper_count"] = feats["consumer_id"].map(eval_g["tamper_flag"].sum()).fillna(0.0)
    feats["tamper_ratio"] = feats["tamper_count"] / (
        feats["consumer_id"].map(eval_g.size()).fillna(1.0) + 1e-5
    )
    feats["fault_count"] = feats["consumer_id"].map(eval_g["fault_flag"].sum()).fillna(0.0)
    feats["fault_ratio"] = feats["fault_count"] / (
        feats["consumer_id"].map(eval_g.size()).fillna(1.0) + 1e-5
    )

    print("[*] Computing load shape (peak_share) change features...")
    # --- Load Shape Change (peak_share drift) ---
    feats["recent_peak_share"] = feats["consumer_id"].map(
        eval_g["peak_share"].median()
    ).fillna(0.35)
    feats["peak_share_change"] = feats["base_peak_share"] - feats["recent_peak_share"]
    feats["peak_share_drop"] = np.maximum(0.0, feats["peak_share_change"])

    print("[*] Computing peer group & transformer features...")
    # --- Peer Group Features (by transformer) ---
    if "transformer_id" not in df_eval.columns:
        df_eval = df_eval.merge(df_c[["consumer_id", "transformer_id"]], on="consumer_id", how="left")
    peer_medians = df_eval.groupby(["transformer_id", "date"])["recorded_kwh"].median().reset_index()
    peer_medians.columns = ["transformer_id", "date", "peer_median_kwh"]
    merged_eval2 = df_eval.merge(peer_medians, on=["transformer_id", "date"], how="left")
    merged_eval2["peer_dev"] = (merged_eval2["recorded_kwh"] - merged_eval2["peer_median_kwh"]) / (
        merged_eval2["peer_median_kwh"] + 1e-5
    )
    feats["peer_deviation_mean"] = feats["consumer_id"].map(
        merged_eval2.groupby("consumer_id")["peer_dev"].mean()
    ).fillna(0.0)

    # --- Transformer Loss Features ---
    consumer_agg = df_eval.groupby(["transformer_id", "date"])["recorded_kwh"].sum().reset_index()
    consumer_agg.columns = ["transformer_id", "date", "consumer_total_kwh"]
    tx_merged = consumer_agg.merge(df_t, on=["transformer_id", "date"], how="left")
    tx_merged["loss_kwh"] = tx_merged["input_kwh"] - tx_merged["consumer_total_kwh"]
    tx_merged["loss_pct"] = tx_merged["loss_kwh"] / (tx_merged["input_kwh"] + 1e-5)
    tx_loss_avg = tx_merged.groupby("transformer_id")["loss_pct"].mean()
    consumer_tx = df_c.set_index("consumer_id")["transformer_id"]
    feats["transformer_loss_pct"] = feats["consumer_id"].map(consumer_tx).map(tx_loss_avg).fillna(0.04)

    # --- Trend: mid-history vs recent (gradual theft detection) ---
    mid_mean = df_mid.groupby("consumer_id")["recorded_kwh"].mean()
    feats["mid_mean"] = feats["consumer_id"].map(mid_mean).fillna(feats["base_mean"])
    feats["trend_mid_to_recent"] = (feats["recent_mean"] - feats["mid_mean"]) / (feats["mid_mean"] + 1e-5)

    # --- Consumer metadata features ---
    feats["sanctioned_load_kw"] = feats["consumer_id"].map(
        df_c.set_index("consumer_id")["sanctioned_load_kw"]
    ).fillna(5.0)
    feats["load_utilization"] = feats["recent_mean"] / (feats["sanctioned_load_kw"] * 24.0 / 1000.0 + 1e-5)
    feats["category_commercial"] = feats["consumer_id"].map(
        df_c.set_index("consumer_id")["category"]
    ).eq("commercial").astype(float)
    feats["category_industrial"] = feats["consumer_id"].map(
        df_c.set_index("consumer_id")["category"]
    ).eq("industrial").astype(float)

    # --- Clean up: replace any NaN/Inf ---
    feature_cols = [c for c in feats.columns if c != "consumer_id"]
    feats[feature_cols] = feats[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    print(f"[+] Feature matrix: {feats.shape[0]} consumers × {len(feature_cols)} features.")
    return feats, feature_cols


if __name__ == "__main__":
    feats, feature_cols = build_realistic_features()
    print(feats[feature_cols].describe().T[["mean", "std", "min", "max"]])
