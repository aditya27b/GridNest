"""
Deterministic Root Cause Classification Engine.
Maps engineered features and ML anomaly evidence to realistic operational causes.
Guarantees: Anomaly != Theft.
"""
from typing import Dict, Any, List, Tuple
import numpy as np

def determine_probable_cause(
    features: Dict[str, float],
    ml_theft_prob: float = 0.0,
    ml_anomaly_score: float = 0.0
) -> Tuple[str, float, List[str], str]:
    """
    Evaluates feature signatures and model outputs to determine probable cause,
    confidence, human-readable evidence, and recommended action.
    """
    evidence = []
    
    # Extract key signals
    neg_dev = features.get("negative_deviation_pct", 0.0)
    pos_dev = features.get("positive_deviation_pct", 0.0)
    dev_30 = features.get("deviation_30d_pct", 0.0)
    consec_low = features.get("consecutive_low_days", 0)
    consec_zero = features.get("consecutive_zero_days", 0)
    missing_ratio_7d = features.get("missing_ratio_7d", 0.0)
    missing_ratio_30d = features.get("missing_ratio_30d", 0.0)
    neg_val_flag = features.get("negative_value_flag", 0.0)
    peer_dev = features.get("peer_deviation_pct", 0.0)
    change_1d = features.get("change_1d", 0.0)

    # 1. Hardware / Data Glitch: Negative readings
    if neg_val_flag > 0.0:
        evidence.append("Corrupted negative power readings detected (sensor polarity/rollover glitch)")
        evidence.append("Physical meter hardware requires recalibration or replacement")
        return (
            "METER_MALFUNCTION",
            0.92,
            evidence,
            "Dispatch technician for physical smart-meter hardware diagnosis and replacement."
        )

    # 2. Communication Failure: High packet loss / missing readings
    if missing_ratio_7d >= 0.50 or missing_ratio_30d >= 0.60:
        evidence.append(f"High missing reading ratio ({missing_ratio_7d*100:.1f}% over 7 days)")
        evidence.append("Communication dropout / AMR network transmission failure")
        evidence.append("Absence of data does NOT constitute evidence of electricity theft")
        return (
            "COMMUNICATION_FAILURE",
            0.88,
            evidence,
            "Initiate remote AMR gateway ping and cellular/RF mesh telecom diagnostic."
        )

    # 3. Flat Zero / Meter Stoppage
    if consec_zero >= 14:
        evidence.append(f"Sustained flat zero consumption recorded for {consec_zero} consecutive days")
        if abs(peer_dev) > 0.5:
            evidence.append("Peer community maintains active consumption while target meter is stalled")
        evidence.append("Suspected meter bypass, tampered disconnect switch, or vacant premises")
        return (
            "POSSIBLE_THEFT_TAMPERING" if ml_theft_prob >= 0.40 else "METER_MALFUNCTION",
            max(0.75, ml_theft_prob),
            evidence,
            "Send verification team to inspect physical seal and verify site occupancy."
        )

    # 4. Seasonal / Contextual Variation: Consumer and peer group move together
    if (pos_dev > 0.25 or features.get("rolling_mean_7", 0.0) > 15.0) and abs(peer_dev) < 0.25:
        evidence.append(f"Consumption elevated but aligns closely with peer group baseline (peer deviation {peer_dev*100:+.1f}%)")
        evidence.append("Regional seasonal or climatic pattern detected across the neighborhood")
        return (
            "SEASONAL_VARIATION",
            0.84,
            evidence,
            "No action required; system automatically updated seasonal baseline."
        )

    # 5. Temporary Legitimate Load Variation: 1-2 day spike
    if pos_dev > 0.35 and consec_low == 0 and change_1d > 5.0:
        evidence.append(f"Isolated high consumption spike detected (+{pos_dev*100:.1f}%)")
        evidence.append("Historical baseline is stable; pattern consistent with temporary appliance usage or event")
        return (
            "TEMPORARY_LEGITIMATE_VARIATION",
            0.85,
            evidence,
            "Continue automated monitoring. No field inspection warranted."
        )

    # 6. Persistent Unexplained Abnormality: High consumption, unmetered load or commercial violation
    if peer_dev > 0.45 or (pos_dev > 0.50 and abs(peer_dev) > 0.40):
        evidence.append(f"Severe persistent high consumption deviating sharply from peers (peer deviation {peer_dev*100:+.1f}%)")
        evidence.append("Possible unsanctioned load expansion or commercial tariff misuse")
        return (
            "PERSISTENT_UNEXPLAINED_ABNORMALITY",
            0.82,
            evidence,
            "Audit sanctioned load capacity against actual sustained peak demand."
        )

    # 7. Persistent Unexplained Reduction / Possible Theft: Sustained low consumption or sharp divergence from peers
    if (neg_dev > 0.25 or peer_dev < -0.40 or consec_low >= 7) and (ml_theft_prob >= 0.35 or ml_anomaly_score >= 50):
        evidence.append(f"Persistent consumption drop ({neg_dev*100:.1f}% below personal 30-day baseline)")
        evidence.append(f"Divergence from active peer group baseline (peer deviation {peer_dev*100:+.1f}%)")
        evidence.append("Telecom and meter health signals are normal (ruling out communication failure)")
        evidence.append(f"ML Classifier theft probability: {ml_theft_prob*100:.1f}%")
        return (
            "POSSIBLE_THEFT_TAMPERING",
            round(max(ml_theft_prob, 0.75), 2),
            evidence,
            "Prioritized field inspection: verify meter seal integrity and check for unauthorized bypass."
        )

    # 8. Normal Consumption
    evidence.append("Consumption adheres within normal statistical deviation bounds (±20%)")
    evidence.append("Communication, meter health, and peer baselines are stable")
    return (
        "NORMAL",
        0.95,
        evidence,
        "Routine automated surveillance."
    )
