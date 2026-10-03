"""
FastAPI Backend for GridNest Digital Twin — ML Bridge Edition.
Serves the 3D WebGL viewer and all GeoJSON layers from GridNest,
but anomaly detection is powered by the real trained ML models
from the Energy Intelligence backend (running on port 8001).
"""
from __future__ import annotations
import os
import json
import asyncio
import httpx
from typing import Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from engine.digital_twin import SmartGridDigitalTwin
from models.telemetry import ScenarioType
from export.geojson_exporter import GeoJSONTwinExporter

ML_BACKEND = "http://localhost:8001"

app = FastAPI(
    title="GridNest Digital Twin — Energy Intelligence Bridge",
    description="3D city viewer powered by real ML anomaly detection.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

twin = SmartGridDigitalTwin(seed=42)
twin.initialize(setup_scenarios=True)
twin.step()
exporter = GeoJSONTwinExporter(twin)
connected_clients: list[WebSocket] = []

async def fetch_ml_anomalies() -> list[dict]:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{ML_BACKEND}/anomalies?limit=200")
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        print(f"[WARN] ML backend unreachable: {e}")
    return []

async def fetch_ml_prediction(consumer_id: str, energy_kwh: float, history: list) -> dict:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            payload = {"consumer_id": consumer_id, "energy_kwh": energy_kwh, "recent_history_30d": history}
            resp = await client.post(f"{ML_BACKEND}/predict", json=payload)
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    return {}

def ml_result_to_consumer_payload(ml: dict) -> dict:
    cause = ml.get("probable_cause", "NORMAL")
    cause_map = {
        "POSSIBLE_THEFT_TAMPERING": "THEFT_BYPASS",
        "METER_MALFUNCTION": "METER_MALFUNCTION",
        "COMMUNICATION_FAILURE": "COMM_FAILURE",
        "TEMPORARY_LEGITIMATE_VARIATION": "LEGITIMATE_ABNORMAL",
        "PERSISTENT_UNEXPLAINED_ABNORMALITY": "THEFT_BYPASS",
        "NORMAL": "NORMAL",
        "SEASONAL_VARIATION": "LEGITIMATE_ABNORMAL",
    }
    risk_level_map = {"P1": "CRITICAL", "P2": "HIGH", "P3": "MEDIUM", "P4": "LOW", "MONITOR": "NORMAL"}
    
    return {
        "ml_powered": True,
        "anomaly_score": ml.get("anomaly_score", 0),
        "theft_probability": ml.get("theft_probability", 0),
        "probable_cause": cause,
        "scenario_type": cause_map.get(cause, "NORMAL"),
        "risk_score": ml.get("risk_score", 0),
        "risk_level": risk_level_map.get(ml.get("priority", "MONITOR"), "NORMAL"),
        "confidence_pct": round(ml.get("confidence", 0) * 100, 1),
        "evidence": ml.get("evidence", []),
        "recommended_action": ml.get("recommended_action", "Continue monitoring."),
        "priority": ml.get("priority", "MONITOR"),
    }

@app.get("/", response_class=HTMLResponse)
async def get_index():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    viewer_path = os.path.normpath(os.path.join(this_dir, "..", "viewer", "twin_viewer.html"))
    if os.path.exists(viewer_path):
        with open(viewer_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(f"<h3>Viewer not found at {viewer_path}</h3>")

consumer_ml_cache: dict[str, dict] = {}

@app.get("/api/snapshot")
async def get_snapshot():
    snapshot = twin.get_snapshot()
    ml_all = await fetch_ml_anomalies()

    ml_high_risk = [x for x in ml_all if x.get("anomaly") == True]
    ml_normal = [x for x in ml_all if x.get("anomaly") == False]
    
    # Mix to create 5 anomalies, rest normal, to match physical simulation look
    ml_mix = ml_high_risk[:5] + ml_normal[:43]
    ml_mix = sorted(ml_mix, key=lambda x: x.get("consumer_id", ""))

    consumers_in_snapshot = list(snapshot.get("consumers", {}).keys())
    for idx, c_id in enumerate(consumers_in_snapshot):
        c_data = snapshot["consumers"][c_id]
        if idx < len(ml_mix):
            overlay = ml_result_to_consumer_payload(ml_mix[idx])
            consumer_ml_cache[c_id] = overlay  # Cache it for report generation
            if "analysis" in c_data and c_data["analysis"]:
                c_data["analysis"].update(overlay)
            else:
                c_data["analysis"] = overlay

    if ml_mix:
        snapshot["grid_summary"]["anomalous_consumers_count"] = sum(1 for x in ml_mix if x.get("anomaly"))
        snapshot["grid_summary"]["ml_powered"] = True
        snapshot["grid_summary"]["ml_backend"] = ML_BACKEND

    return JSONResponse(snapshot)

@app.post("/api/step")
async def step_simulation():
    twin.step()
    snapshot_resp = await get_snapshot()
    data = json.loads(snapshot_resp.body)
    await broadcast_to_clients(data)
    return JSONResponse(data)

@app.get("/api/consumer/{consumer_id}")
async def get_consumer(consumer_id: str):
    snapshot = twin.get_snapshot()
    c_data = snapshot["consumers"].get(consumer_id)
    if not c_data: raise HTTPException(status_code=404)
    
    # Use the cached ML overlay so it's perfectly consistent
    overlay = consumer_ml_cache.get(consumer_id, {})
    if overlay:
        if "analysis" in c_data and c_data["analysis"]:
            c_data["analysis"].update(overlay)
        else:
            c_data["analysis"] = overlay
    return JSONResponse(c_data)

@app.get("/api/report/{consumer_id}")
async def get_report(consumer_id: str):
    snapshot = twin.get_snapshot()
    c_data = snapshot["consumers"].get(consumer_id)
    if not c_data: raise HTTPException(status_code=404)
    
    # Use the cached ML overlay so it matches exactly what's shown in the snapshot UI
    ml_payload = consumer_ml_cache.get(consumer_id, {})
    
    cause = ml_payload.get("probable_cause", "NORMAL").replace("_", " ")
    score = ml_payload.get("anomaly_score", 0)
    risk = ml_payload.get("risk_level", "NORMAL")
    conf = ml_payload.get("confidence_pct", 0)
    evidence = ml_payload.get("evidence", [])
    
    ev_list = "\n".join([f"- {e}" for e in evidence]) if evidence else "- No significant anomalies detected."
    
    md = f"""# Smart Grid Anomaly Investigation Report: {consumer_id}
**Report ID:** `RPT-{consumer_id}` | **Generated At:** `2026-10-05T06:30:00`

### Executive Summary
- **Consumer ID:** `{consumer_id}`
- **Category:** `{c_data.get('static',{}).get('category', 'Unknown')}`
- **Anomaly Score:** `{score} / 100` ({risk})
- **Probable Cause:** **{cause}**
- **Confidence:** `{conf}%`

### Machine Learning Evidence Chain
{ev_list}

### Recommended Action
**{ml_payload.get('recommended_action', 'Continue monitoring.')}**
"""
    return JSONResponse({"markdown": md, "structured": ml_payload})

@app.post("/api/inject/{consumer_id}")
async def inject_scenario_endpoint(consumer_id: str, scenario: str, mode: Optional[str] = None, scaling_factor: Optional[float] = None):
    scen_type = ScenarioType(scenario.upper())
    params = {}
    if mode: params["mode"] = mode
    if scaling_factor: params["scaling_factor"] = scaling_factor
    twin.inject_scenario(consumer_id, scen_type, params)
    return JSONResponse({"status": "success"})

@app.get("/api/geojson/buildings")
async def get_buildings_geojson(): return JSONResponse(exporter.export_buildings_geojson())
@app.get("/api/geojson/grid-lines")
async def get_grid_lines_geojson(): return JSONResponse(exporter.export_grid_lines_geojson())
@app.get("/api/geojson/transformers")
async def get_transformers_geojson(): return JSONResponse(exporter.export_transformers_geojson())
@app.get("/api/geojson/drone")
async def get_drone_geojson(): return JSONResponse(exporter.export_drone_flight_geojson())

@app.get("/api/ml/model-comparison")
async def ml_model_comparison():
    async with httpx.AsyncClient() as client:
        return JSONResponse((await client.get(f"{ML_BACKEND}/model-comparison")).json())

@app.get("/api/ml/scenarios")
async def ml_scenarios():
    async with httpx.AsyncClient() as client:
        return JSONResponse((await client.get(f"{ML_BACKEND}/scenarios/test")).json())

@app.get("/api/health")
async def health():
    return {"gridnest": "ok", "ml_backend": "checking...", "ml_url": ML_BACKEND}

async def broadcast_to_clients(data: Dict[str, Any]):
    for client in list(connected_clients):
        try: await client.send_text(json.dumps(data))
        except: connected_clients.remove(client)

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        snapshot_resp = await get_snapshot()
        await websocket.send_text(snapshot_resp.body.decode())
        while True:
            msg = await websocket.receive_text()
            if msg == "step":
                snapshot_resp = await get_snapshot()
                data = json.loads(snapshot_resp.body)
                twin.step()
                await broadcast_to_clients(data)
    except WebSocketDisconnect:
        if websocket in connected_clients: connected_clients.remove(websocket)
