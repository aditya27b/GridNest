"""
Inference & Prediction Pipeline.
Applies trained Isolation Forest, XGBoost Classifier, Cause Engine, and Risk Engine.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd

from ml.config import ROOT, FEATURE_COLUMNS
from ml.feature_engineering import extract_features_at_timestep
from ml.cause_engine import determine_probable_cause
from ml.risk_engine import calculate_risk_score

MODELS_DIR = ROOT / "models"

def load_models():
    iso_path = MODELS_DIR / "isolation_forest.joblib"
    clf_path = MODELS_DIR / "cause_classifier.joblib"

    if not iso_path.exists() or not clf_path.exists():
        raise FileNotFoundError("Trained models not found in 'models/'. Run 'python -m ml.train' first.")

    iso = joblib.load(iso_path)
    clf_data = joblib.load(clf_path)
    return iso, clf_data["model"], clf_data.get("threshold", 0.50)

def predict_consumer_state(
    matrix: np.ndarray,
    dates: list,
    consumer_idx: int = 0,
    consumer_id: str = "C0001",
    t: Optional[int] = None
) -> Dict[str, Any]:
    """
    Evaluates a consumer's time-series up to day index t.
    """
    iso, clf, threshold = load_models()
    num_days = matrix.shape[1]
    if t is None:
        t = num_days - 1

    # Extract time-aware feature row
    feats_df = extract_features_at_timestep(matrix, t, dates)
    feats = feats_df.iloc[consumer_idx].to_dict()
    row_df = feats_df.iloc[[consumer_idx]]

    # 1. Unsupervised Anomaly Score (0 - 100)
    anomaly_score = float(np.round(iso.score(row_df)[0], 2))

    # 2. Supervised Theft Probability (0.0 - 1.0)
    theft_prob = float(np.round(clf.predict_proba(row_df)[0], 4))

    # 3. Probable Cause & Human-Readable Evidence
    cause, conf, evidence, rec_action = determine_probable_cause(feats, theft_prob, anomaly_score)

    # 4. Risk Score & Priority
    risk_score, risk_lvl, priority = calculate_risk_score(feats, theft_prob, anomaly_score, cause)

    curr_val = float(matrix[consumer_idx, t])
    ts_str = dates[t].strftime("%Y-%m-%d")

    return {
        "consumer_id": str(consumer_id),
        "timestamp": ts_str,
        "energy_kwh": round(curr_val, 2) if not np.isnan(curr_val) else None,
        "anomaly": bool(anomaly_score >= 50.0 or theft_prob >= 0.40 or risk_score >= 50.0),
        "anomaly_score": anomaly_score,
        "theft_probability": theft_prob,
        "probable_cause": cause,
        "confidence": conf,
        "risk_score": risk_score,
        "risk_level": risk_lvl,
        "priority": priority,
        "evidence": evidence,
        "recommended_action": rec_action,
        "features": {k: round(float(v), 3) for k, v in feats.items()}
    }

def predict_single_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inference endpoint for single meter reading with historical context.
    """
    c_curr = float(payload.get("consumption_kwh", payload.get("energy_kwh", 12.0)))
    history = payload.get("recent_history_30d", [c_curr] * 30)
    if len(history) < 30:
        history = [c_curr] * (30 - len(history)) + list(history)

    full_series = np.array(history + [c_curr], dtype=float).reshape(1, -1)
    dates = [pd.Timestamp("2026-01-01") + pd.Timedelta(days=i) for i in range(len(full_series[0]))]

    return predict_consumer_state(
        matrix=full_series,
        dates=dates,
        consumer_idx=0,
        consumer_id=str(payload.get("consumer_id", "C0001")),
        t=len(full_series[0]) - 1
    )
