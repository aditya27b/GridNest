"""
End-to-End Pipeline & Zero-Leakage Unit Tests.
"""
import numpy as np
import pandas as pd
from ml.feature_engineering import extract_features_at_timestep
from ml.predict import predict_single_payload
from simulator.scenarios import run_all_scenarios

def test_feature_extraction_no_nans():
    mat = np.random.uniform(5, 20, size=(10, 60))
    dates = [pd.Timestamp("2026-01-01") + pd.Timedelta(days=i) for i in range(60)]
    feats = extract_features_at_timestep(mat, 45, dates)
    assert feats.isna().sum().sum() == 0, "Feature matrix contains NaNs!"
    assert not np.isinf(feats.values).any(), "Feature matrix contains Infs!"
    print("[PASS] Feature extraction produces clean, finite values.")

def test_zero_temporal_leakage():
    """
    CRITICAL: Verifies that altering future readings at t+1 ... T
    strictly DOES NOT change features computed at time t.
    """
    mat_orig = np.random.uniform(5, 20, size=(5, 60))
    dates = [pd.Timestamp("2026-01-01") + pd.Timedelta(days=i) for i in range(60)]
    t = 40

    feats_before = extract_features_at_timestep(mat_orig, t, dates)

    # Mutate future days (days 41 to 59)
    mat_mutated = mat_orig.copy()
    mat_mutated[:, 41:] = 9999.0  # extreme future corruption

    feats_after = extract_features_at_timestep(mat_mutated, t, dates)

    # Assert exact match
    diff = np.abs(feats_before.values - feats_after.values).max()
    assert diff == 0.0, f"TEMPORAL LEAKAGE DETECTED! Max diff: {diff}"
    print("[PASS] Zero Temporal Leakage Verified: Future values do not affect historical features.")

def test_single_prediction_payload():
    payload = {
        "consumer_id": "C999",
        "energy_kwh": 4.1,
        "recent_history_30d": [12.0] * 30
    }
    res = predict_single_payload(payload)
    assert res["consumer_id"] == "C999"
    assert "anomaly_score" in res
    assert "probable_cause" in res
    assert "priority" in res
    assert "evidence" in res
    print(f"[PASS] Single prediction successful: Cause={res['probable_cause']}, Priority={res['priority']}")

def test_scenario_suite():
    passed = run_all_scenarios()
    assert passed, "Scenario suite failed!"
    print("[PASS] All 7 operational scenario stress tests passed.")

if __name__ == "__main__":
    print("\n=======================================================")
    print(" RUNNING INTEGRATION & ZERO-LEAKAGE TEST SUITE")
    print("=======================================================")
    test_feature_extraction_no_nans()
    test_zero_temporal_leakage()
    test_single_prediction_payload()
    test_scenario_suite()
    print("\n[+] ALL UNIT & INTEGRATION TESTS COMPLETED SUCCESSFULLY!\n")
