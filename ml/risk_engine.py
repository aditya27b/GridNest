"""
Configurable Risk Aggregation Engine.
Combines ML anomaly probabilities, persistence signatures, peer deviations,
and cause evidence into an operational risk score (0-100) and inspection priority.
"""
from typing import Dict, Any, Tuple
import numpy as np
from ml.config import CONFIG

def calculate_risk_score(
    features: Dict[str, float],
    ml_theft_prob: float,
    ml_anomaly_score: float,
    probable_cause: str
) -> Tuple[float, str, str]:
    """
    Computes final risk score, risk level, and operational inspection priority.
    """
    cfg = CONFIG.get("risk_engine", {})
    weights = cfg.get("weights", {
        "anomaly_score": 0.30,
        "persistence": 0.25,
        "peer_deviation": 0.20,
        "communication_health": 0.15,
        "meter_health": 0.10
    })

    dev = abs(float(features.get("deviation_30d_pct", 0.0)))
    peer = abs(float(features.get("peer_deviation_pct", 0.0)))
    consec_low = float(features.get("consecutive_low_days", 0))
    comm_miss = float(features.get("missing_ratio_7d", 0.0))
    neg_flag = float(features.get("negative_value_flag", 0.0))

    # Normalized component signals [0.0, 1.0]
    s_anomaly = np.clip(max(ml_theft_prob, ml_anomaly_score / 100.0), 0.0, 1.0)
    s_persistence = np.clip(consec_low / 14.0, 0.0, 1.0)
    s_peer = np.clip(peer / 0.80, 0.0, 1.0)
    s_baseline = np.clip(dev / 0.80, 0.0, 1.0)
    s_health = np.clip(neg_flag + (1.0 if comm_miss > 0.4 else 0.0), 0.0, 1.0)

    # Weighted baseline risk
    raw_risk = 100.0 * (
        weights.get("anomaly_score", 0.30) * s_anomaly +
        weights.get("persistence", 0.25) * s_persistence +
        weights.get("peer_deviation", 0.20) * s_peer +
        weights.get("communication_health", 0.15) * s_baseline +
        weights.get("meter_health", 0.10) * s_health
    )

    # Operational Cause Modifiers
    if probable_cause == "TEMPORARY_LEGITIMATE_VARIATION":
        raw_risk *= 0.35
    elif probable_cause == "SEASONAL_VARIATION":
        raw_risk *= 0.30
    elif probable_cause == "NORMAL":
        raw_risk *= 0.20
    elif probable_cause == "COMMUNICATION_FAILURE":
        # Telecom issues are maintenance tickets, not high-theft P1 emergencies
        raw_risk = min(raw_risk * 0.50, 35.0)
    elif probable_cause == "METER_MALFUNCTION":
        # Hardware replacement needed, capped at P2
        raw_risk = min(max(raw_risk, 45.0), 65.0)
    elif probable_cause == "POSSIBLE_THEFT_TAMPERING":
        if consec_low >= 14:
            raw_risk = max(raw_risk, 80.0)
        else:
            raw_risk = max(raw_risk, 65.0)

    final_risk = round(float(np.clip(raw_risk, 0.0, 100.0)), 2)

    # Priority mapping
    if final_risk >= 75.0:
        level = "CRITICAL"
        priority = "P1"
    elif final_risk >= 50.0:
        level = "HIGH"
        priority = "P2"
    elif final_risk >= 25.0:
        level = "MEDIUM"
        priority = "P3"
    else:
        level = "LOW"
        priority = "MONITOR"

    return final_risk, level, priority
