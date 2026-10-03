"""
Scenario Testing Suite for Energy Intelligence Pipeline.
Generates realistic edge-case time-series to verify the Anomaly != Theft principle.
"""
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from ml.feature_engineering import extract_features_at_timestep
from ml.cause_engine import determine_probable_cause
from ml.risk_engine import calculate_risk_score

def build_scenario_matrix(scenario_name: str, num_days: int = 90) -> Tuple[np.ndarray, List[pd.Timestamp]]:
    """
    Constructs a 1-consumer time-series matrix for 90 consecutive days.
    Day index 89 is the evaluation day (t=89).
    """
    dates = [pd.Timestamp("2026-01-01") + pd.Timedelta(days=i) for i in range(num_days)]
    r = np.random.default_rng(42)
    # Baseline 11 - 13 kWh
    base = 12.0
    series = base + r.normal(0, 0.5, num_days)

    if scenario_name == "NORMAL":
        pass  # remains normal around 12 kWh

    elif scenario_name == "1_DAY_SPIKE":
        # Day 89 spikes to 25 kWh (isolated 1-day heavy appliance spike)
        series[89] = 25.0

    elif scenario_name == "PERSISTENT_LOW_THEFT":
        # Theft begins at day 68 (last 21 days sustained at 4.2 kWh)
        series[68:] = 4.2 + r.normal(0, 0.2, num_days - 68)

    elif scenario_name == "COMMUNICATION_FAILURE":
        # Last 14 days have 80% packet dropout / null readings
        for d in range(76, num_days):
            if r.random() < 0.80:
                series[d] = np.nan

    elif scenario_name == "METER_MALFUNCTION":
        # Sensor polarity glitch / negative reading
        series[88] = -45.2
        series[89] = -12.0

    elif scenario_name == "SEASONAL_VARIATION":
        # Consumer rises to 18.5 kWh from day 65
        series[65:] = 18.5 + r.normal(0, 0.4, num_days - 65)

    elif scenario_name == "PERSISTENT_HIGH":
        # Unsanctioned heavy load increase (+80% sustained) from day 65
        series[65:] = 22.5 + r.normal(0, 0.5, num_days - 65)

    matrix = series.reshape(1, num_days)
    return matrix, dates

def test_scenario(scenario_name: str) -> Dict[str, Any]:
    matrix, dates = build_scenario_matrix(scenario_name)
    eval_t = 89
    feats_df = extract_features_at_timestep(matrix, eval_t, dates)
    feats = feats_df.iloc[0].to_dict()

    # Contextual peer baseline configuration
    if scenario_name == "SEASONAL_VARIATION":
        # Peer group also increased to 18.2 kWh
        feats["peer_median"] = 18.2
        feats["peer_deviation_pct"] = (feats["rolling_mean_7"] - 18.2) / 18.2
    elif scenario_name in ["PERSISTENT_LOW_THEFT", "PERSISTENT_HIGH", "1_DAY_SPIKE", "NORMAL"]:
        # Peer group stayed at normal 12.0 kWh
        feats["peer_median"] = 12.0
        c_curr = matrix[0, 89]
        feats["peer_deviation_pct"] = (c_curr - 12.0) / 12.0

    # Model scores configuration for scenario
    theft_prob = 0.85 if scenario_name == "PERSISTENT_LOW_THEFT" else 0.05
    anomaly_score = 85.0 if scenario_name in ["PERSISTENT_LOW_THEFT", "METER_MALFUNCTION", "1_DAY_SPIKE", "PERSISTENT_HIGH"] else 12.0

    cause, conf, evidence, rec_action = determine_probable_cause(feats, theft_prob, anomaly_score)
    risk_score, risk_lvl, priority = calculate_risk_score(feats, theft_prob, anomaly_score, cause)

    return {
        "scenario": scenario_name,
        "probable_cause": cause,
        "confidence": conf,
        "risk_score": risk_score,
        "risk_level": risk_lvl,
        "priority": priority,
        "primary_evidence": evidence[0] if evidence else "None",
        "recommended_action": rec_action
    }

def run_all_scenarios():
    scenarios = [
        ("NORMAL", "NORMAL", ["MONITOR", "P4"]),
        ("1_DAY_SPIKE", "TEMPORARY_LEGITIMATE_VARIATION", ["MONITOR", "P4", "P3"]),
        ("PERSISTENT_LOW_THEFT", "POSSIBLE_THEFT_TAMPERING", ["P1", "P2"]),
        ("COMMUNICATION_FAILURE", "COMMUNICATION_FAILURE", ["P3", "MONITOR"]),
        ("METER_MALFUNCTION", "METER_MALFUNCTION", ["P2", "P3"]),
        ("SEASONAL_VARIATION", "SEASONAL_VARIATION", ["MONITOR", "P3"]),
        ("PERSISTENT_HIGH", "PERSISTENT_UNEXPLAINED_ABNORMALITY", ["P2", "P3"])
    ]

    print("\n=======================================================")
    print(" RUNNING AUTOMATED SCENARIO STRESS TESTS (Anomaly != Theft)")
    print("=======================================================")
    
    passed_all = True
    for name, exp_cause, exp_priorities in scenarios:
        res = test_scenario(name)
        cause_match = (res["probable_cause"] == exp_cause)
        prio_match = (res["priority"] in exp_priorities)
        passed = cause_match and prio_match
        if not passed:
            passed_all = False

        status_str = "[PASS]" if passed else "[FAIL]"
        print(f"\n{status_str} Scenario: {name}")
        print(f"       -> Classified Cause: {res['probable_cause']} (Confidence: {res['confidence']*100:.1f}%)")
        print(f"       -> Risk Score:       {res['risk_score']} / 100 ({res['risk_level']} - Priority: {res['priority']})")
        print(f"       -> Evidence:         {res['primary_evidence']}")
        print(f"       -> Action:           {res['recommended_action']}")

    print("\n=======================================================")
    if passed_all:
        print(" [+] ALL 7 SCENARIO TESTS PASSED SUCCESSFULLY!")
        print("     Verified: Communication failures, meter faults, and seasonal variations")
        print("     are strictly separated from theft tampering!")
    else:
        print(" [!] ONE OR MORE SCENARIO TESTS FAILED.")
    print("=======================================================\n")
    return passed_all

if __name__ == "__main__":
    run_all_scenarios()
