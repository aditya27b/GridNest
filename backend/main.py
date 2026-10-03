"""
FastAPI Backend Application for Energy Intelligence.
Provides endpoints for single reading inference, batch scoring, inspection queues,
and model performance benchmarks.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import io
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ml.config import ROOT
from ml.predict import predict_single_payload, predict_consumer_state, load_models
from ml.feature_engineering import extract_features_at_timestep
from ml.dataset_adapter import load_clean_timeseries
from simulator.scenarios import run_all_scenarios, test_scenario

app = FastAPI(
    title="Codeutsav Energy Intelligence API",
    description="ML-first intelligent electricity theft, tampering, and smart-meter anomaly detection.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SingleReadingPayload(BaseModel):
    consumer_id: str = Field(..., example="C0001")
    meter_id: Optional[str] = Field("M0001", example="M0001")
    energy_kwh: float = Field(..., example=4.2)
    voltage: Optional[float] = Field(230.0, example=230.0)
    current: Optional[float] = Field(2.5, example=2.5)
    power_kw: Optional[float] = Field(0.55, example=0.55)
    meter_status: Optional[str] = Field("NORMAL", example="NORMAL")
    communication_status: Optional[str] = Field("NORMAL", example="NORMAL")
    recent_history_30d: Optional[List[float]] = Field(default_factory=list)

@app.get("/health")
def health():
    models_ready = (ROOT / "models" / "isolation_forest.joblib").exists()
    return {
        "status": "healthy",
        "models_ready": models_ready,
        "pipeline_version": "2.0.0"
    }

@app.post("/predict")
def predict_single(payload: SingleReadingPayload):
    try:
        result = predict_single_payload(payload.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-comparison")
def model_comparison():
    rep_path = ROOT / "reports" / "model_comparison.csv"
    if not rep_path.exists():
        raise HTTPException(status_code=404, detail="Run 'python -m ml.train' or 'python -m ml.model_selection' first.")
    df = pd.read_csv(rep_path)
    return df.to_dict(orient="records")

@app.get("/scenarios/test")
def run_scenarios():
    scenarios = [
        "NORMAL", "1_DAY_SPIKE", "PERSISTENT_LOW_THEFT",
        "COMMUNICATION_FAILURE", "METER_MALFUNCTION",
        "SEASONAL_VARIATION", "PERSISTENT_HIGH"
    ]
    results = [test_scenario(sc) for sc in scenarios]
    return {"status": "success", "results": results}

@app.get("/anomalies")
def anomalies(limit: int = 50):
    return inspection_queue(limit=limit)

@app.get("/inspection-queue")
def inspection_queue(limit: int = 50):
    """
    Fast vectorized evaluation of real consumers from the raw dataset.
    """
    data_path = ROOT / "data" / "Electricity_Theft_Data.csv"
    if not data_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found at data/Electricity_Theft_Data.csv")

    cids, dates, matrix, labels, _ = load_clean_timeseries(str(data_path), max_consumers=200)
    eval_t = matrix.shape[1] - 1
    
    iso, clf, threshold = load_models()
    feats_df = extract_features_at_timestep(matrix, eval_t, dates)
    anomaly_scores = iso.score(feats_df)
    theft_probs = clf.predict_proba(feats_df)

    from ml.cause_engine import determine_probable_cause
    from ml.risk_engine import calculate_risk_score

    queue = []
    for i in range(len(cids)):
        feats = feats_df.iloc[i].to_dict()
        asc = float(np.round(anomaly_scores[i], 2))
        tp = float(np.round(theft_probs[i], 4))
        cause, conf, ev, act = determine_probable_cause(feats, tp, asc)
        risk, rlvl, prio = calculate_risk_score(feats, tp, asc, cause)
        curr_val = float(matrix[i, eval_t])
        queue.append({
            "consumer_id": str(cids[i]),
            "timestamp": dates[eval_t].strftime("%Y-%m-%d"),
            "energy_kwh": round(curr_val, 2) if not np.isnan(curr_val) else None,
            "anomaly": bool(asc >= 50.0 or tp >= 0.40 or risk >= 50.0),
            "anomaly_score": asc,
            "theft_probability": tp,
            "probable_cause": cause,
            "confidence": conf,
            "risk_score": risk,
            "risk_level": rlvl,
            "priority": prio,
            "evidence": ev,
            "recommended_action": act,
            "features": {k: round(float(v), 3) for k, v in feats.items()}
        })

    # Sort by risk score descending
    queue = sorted(queue, key=lambda x: x["risk_score"], reverse=True)[:limit]
    for idx, item in enumerate(queue):
        item["rank"] = idx + 1
    return queue

@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Accepts CSV upload of meter readings and returns predictions with risk classification.
    """
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
        # Process preview of first 50 rows
        results = []
        for _, row in df.head(50).iterrows():
            payload = {
                "consumer_id": str(row.get("CONS_NO", row.get("consumer_id", "C_UNKNOWN"))),
                "energy_kwh": float(row.iloc[1]) if len(row) > 1 else 10.0,
                "recent_history_30d": [float(x) for x in row.iloc[1:31].dropna()] if len(row) > 31 else []
            }
            results.append(predict_single_payload(payload))
        return {
            "total_evaluated": len(results),
            "predictions": results
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process CSV: {str(e)}")
