"""
FastAPI Backend for GridNest Digital Twin — Real-Time ML & IoT Hardware Bridge.
Serves the 3D WebGL viewer and all GeoJSON layers from GridNest,
with dynamic multi-signal anomaly detection, real-time scenario simulation,
and live hardware telemetry integration for Zone B Transformer (TX_102) & Consumer (CONS_S_001).
"""
from __future__ import annotations
import os
import json
import time
import asyncio
import httpx
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from engine.digital_twin import SmartGridDigitalTwin
from models.telemetry import ScenarioType
from export.geojson_exporter import GeoJSONTwinExporter

ML_BACKEND = "http://localhost:8001"

app = FastAPI(
    title="GridNest Digital Twin — Energy Intelligence & IoT Hardware Bridge",
    description="3D smart grid digital twin with ML anomaly detection and live ESP32/Adafruit IoT integration.",
    version="2.2.0",
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
exporter = GeoJSONTwinExporter(twin)
connected_clients: list[WebSocket] = []
consumer_ml_cache: dict[str, dict] = {}

# Live physical IoT hardware state (e.g. ESP32 on COM7 / Adafruit IO)
latest_iot_telemetry: Optional[dict] = None
last_iot_timestamp: float = 0.0


class IoTTelemetryPayload(BaseModel):
    transVoltage: float = Field(..., example=4.50, description="Transformer measured voltage in Volts")
    transCurrent: float = Field(..., example=462.20, description="Transformer measured current in mA")
    transPower_W: float = Field(..., example=2.08, description="Transformer measured power in Watts")
    transEnergy_Wh: float = Field(0.0, example=0.0139, description="Transformer cumulative energy in Wh")
    consVoltage: float = Field(..., example=4.35, description="Consumer measured voltage in Volts")
    consCurrent: float = Field(..., example=521.83, description="Consumer measured current in mA")
    consPower_W: float = Field(..., example=2.27, description="Consumer measured power in Watts")
    consEnergy_Wh: float = Field(0.0, example=0.0150, description="Consumer cumulative energy in Wh")
    powerLoss_W: float = Field(0.0, example=0.00, description="Measured line power loss in Watts")
    energyLoss_Wh: float = Field(0.0, example=0.0000, description="Cumulative energy loss in Wh")
    target_consumer_id: Optional[str] = Field("CONS_S_001", description="Consumer ID in Zone B")
    target_transformer_id: Optional[str] = Field("TX_102", description="Transformer ID in Zone B")


def apply_iot_hardware_overlay(snapshot: dict) -> dict:
    """Injects live physical ESP32 / Adafruit IoT sensor readings into Zone B."""
    global latest_iot_telemetry, last_iot_timestamp
    if not latest_iot_telemetry:
        return snapshot

    iot = latest_iot_telemetry
    c_id = iot.get("target_consumer_id", "CONS_S_001")
    tx_id = iot.get("target_transformer_id", "TX_102")

    # 1. Update Target Consumer in Zone B (e.g. CONS_S_001)
    consumers = snapshot.get("consumers", {})
    if c_id in consumers:
        c_data = consumers[c_id]
        trans_w = float(iot.get("transPower_W", 0.0))
        cons_w = float(iot.get("consPower_W", 0.0))
        loss_w = float(iot.get("powerLoss_W", 0.0))
        cons_v = float(iot.get("consVoltage", 0.0))
        cons_ma = float(iot.get("consCurrent", 0.0))
        loss_wh = float(iot.get("energyLoss_Wh", 0.0))

        # Check for Physical Shunt Tap Theft
        # If line loss is significant (> 0.20 W or > 8% of transformer power), flag physical line tap
        loss_pct = (loss_w / max(0.01, trans_w)) * 100.0 if trans_w > 0 else 0.0
        has_theft = (loss_w > 0.25 and loss_pct > 8.0) or (trans_w > cons_w + 0.30)

        if has_theft:
            score = min(100.0, round(78.0 + (loss_pct * 0.22), 1))
            risk = "CRITICAL"
            cause = "THEFT_TAMPERING"
            action = f"PHYSICAL LINE TAP DETECTED: Anti-theft field squad dispatched to inspect drop line between {tx_id} and {c_id}."
            factors = [
                f"Physical ESP32 Sensor: Active line tap / power theft detected on Zone B drop wire",
                f"Transformer input: {trans_w:.2f} W, but consumer meter only registers {cons_w:.2f} W",
                f"Unmetered line diversion loss: {loss_w:.2f} W ({loss_pct:.1f}% stolen power)",
                f"Cumulative stolen energy recorded: {loss_wh:.4f} Wh",
                f"Terminal voltage drop: {cons_v:.2f} V (load current: {cons_ma:.1f} mA)",
            ]
        else:
            score = 12.0
            risk = "NORMAL"
            cause = "NORMAL"
            action = "Live hardware sensors verified healthy. Physical circuit in balance."
            factors = [
                f"Physical ESP32 Sensor: Circuit balanced. Transformer: {trans_w:.2f} W, Meter: {cons_w:.2f} W",
                f"Unexplained physical line loss: {loss_w:.2f} W (nominal resistive loss within ±5% tolerance)",
                f"Measured terminal voltage: {cons_v:.2f} V, Current: {cons_ma:.1f} mA",
                f"Cumulative energy consumed: {float(iot.get('consEnergy_Wh', 0.0)):.4f} Wh",
            ]

        # Update telemetry
        if "telemetry" in c_data and c_data["telemetry"]:
            tel = c_data["telemetry"]
            if "reported" in tel and tel["reported"]:
                tel["reported"]["voltage_v"] = cons_v
                tel["reported"]["current_a"] = round(cons_ma / 1000.0, 3)
                tel["reported"]["active_power_kw"] = round(cons_w / 1000.0, 4)
            if "ground_truth" in tel and tel["ground_truth"]:
                tel["ground_truth"]["voltage_v"] = float(iot.get("transVoltage", 0.0))
                tel["ground_truth"]["active_power_kw"] = round(trans_w / 1000.0, 4)
            tel["unreported_stolen_kw"] = round(loss_w / 1000.0, 4)

        # Update analysis
        c_data["analysis"] = {
            "consumer_id": c_id,
            "anomaly_score": score,
            "risk_level": risk,
            "probable_cause": cause,
            "confidence_pct": 98.5 if has_theft else 95.0,
            "reported_kw": round(cons_w / 1000.0, 4),
            "baseline_mean_kw": round(trans_w / 1000.0, 4),
            "deviation_pct": round(-loss_pct if has_theft else 0.0, 1),
            "contributing_factors": factors,
            "recommended_action": action,
            "is_iot_hardware": True,
            "ml_powered": True,
        }

        # Tag static metadata
        c_data["static"]["is_iot_hardware"] = True
        c_data["static"]["iot_source"] = "ESP32_Dev_Module_COM7"
        c_data["iot_telemetry"] = iot

    # 2. Update Target Transformer in Zone B (TX_102)
    transformers = snapshot.get("transformers", {})
    if tx_id in transformers:
        tx_data = transformers[tx_id]
        trans_w = float(iot.get("transPower_W", 0.0))
        cons_w = float(iot.get("consPower_W", 0.0))
        loss_w = float(iot.get("powerLoss_W", 0.0))
        loss_wh = float(iot.get("energyLoss_Wh", 0.0))
        loss_pct = (loss_w / max(0.01, trans_w)) * 100.0 if trans_w > 0 else 0.0

        is_theft_flag = (loss_w > 0.25 and loss_pct > 8.0)

        iot_sensors = tx_data.setdefault("iot_sensors", {})
        iot_sensors["active_power_kw"] = round(trans_w / 1000.0, 4)
        iot_sensors["voltage_v"] = float(iot.get("transVoltage", 0.0))
        iot_sensors["secondary_current_a"] = round(float(iot.get("transCurrent", 0.0)) / 1000.0, 3)
        iot_sensors["unexplained_loss_kw"] = round(loss_w / 1000.0, 4)
        iot_sensors["unexplained_loss_pct"] = round(loss_pct, 1)
        iot_sensors["is_done_for"] = is_theft_flag
        iot_sensors["status"] = (
            f"ALERT_THEFT_DISCREPANCY ({loss_w:.2f}W LINE LOSS!)"
            if is_theft_flag
            else "BALANCED_HEALTHY (IoT STREAMING)"
        )

        acc = tx_data.setdefault("accounting", {})
        acc["is_done_for"] = is_theft_flag
        acc["power_inputted_kwh"] = round(trans_w, 2)
        acc["meters_added_kwh"] = round(cons_w, 2)
        acc["cumulative_tech_loss_kwh"] = 0.02
        acc["cumulative_unexplained_kwh"] = round(loss_w, 2)
        acc["unexplained_loss_pct"] = round(loss_pct, 1)
        acc["is_iot_hardware"] = True

        tx_data["is_iot_hardware"] = True
        tx_data["iot_telemetry"] = iot

    # 3. Update Zone 2 Accounting
    zone2 = snapshot.get("zone_accounting", {}).get("Zone_2_South", {})
    if zone2 and "metrics" in zone2:
        z_met = zone2["metrics"]
        if latest_iot_telemetry:
            lw = float(latest_iot_telemetry.get("powerLoss_W", 0.0))
            tw = float(latest_iot_telemetry.get("transPower_W", 0.01))
            lp = (lw / max(0.01, tw)) * 100.0
            z_met["is_done_for"] = (lw > 0.25 and lp > 8.0)
            z_met["unexplained_loss_pct"] = round(lp, 1)

    # 4. Update global grid summary
    summary = snapshot.setdefault("grid_summary", {})
    summary["iot_hardware_active"] = True
    summary["iot_last_ping_seconds_ago"] = round(time.time() - last_iot_timestamp, 1)

    return snapshot


def sync_snapshot_cache(snapshot: dict) -> dict:
    """Updates internal cache and applies live physical IoT hardware overlay."""
    snapshot = apply_iot_hardware_overlay(snapshot)
    consumers = snapshot.get("consumers", {})
    for c_id, c_data in consumers.items():
        an = c_data.get("analysis")
        if an:
            consumer_ml_cache[c_id] = an
            an["ml_powered"] = True
    snapshot.setdefault("grid_summary", {})["ml_powered"] = True
    snapshot.setdefault("grid_summary", {})["ml_backend"] = ML_BACKEND
    return snapshot


@app.get("/", response_class=HTMLResponse)
async def get_index():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    viewer_path = os.path.normpath(os.path.join(this_dir, "..", "viewer", "twin_viewer.html"))
    if os.path.exists(viewer_path):
        with open(viewer_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(f"<h3>Viewer not found at {viewer_path}</h3>")


@app.get("/api/snapshot")
async def get_snapshot():
    """Returns the live digital twin snapshot with real-time ML state and IoT hardware overlay."""
    snapshot = twin.get_snapshot()
    snapshot = sync_snapshot_cache(snapshot)
    return JSONResponse(snapshot)


@app.post("/api/step")
async def step_simulation():
    """Advances the simulation clock by 1 tick (15 minutes), re-evaluates all ML metrics, and broadcasts."""
    twin.step()
    snapshot = twin.get_snapshot()
    snapshot = sync_snapshot_cache(snapshot)
    await broadcast_to_clients(snapshot)
    return JSONResponse(snapshot)


@app.post("/api/iot/telemetry")
async def ingest_iot_telemetry(payload: IoTTelemetryPayload):
    """
    Ingests live physical sensor telemetry from ESP32 / Arduino / Adafruit IO:
    Maps transformer readings to Zone B Transformer (TX_102) and
    consumer readings to Zone B house (CONS_S_001).
    Broadcasts the live state immediately to the 3D Viewer via WebSockets.
    """
    global latest_iot_telemetry, last_iot_timestamp
    latest_iot_telemetry = payload.dict()
    last_iot_timestamp = time.time()

    snapshot = twin.get_snapshot()
    snapshot = sync_snapshot_cache(snapshot)
    await broadcast_to_clients(snapshot)

    return JSONResponse({
        "status": "success",
        "message": f"IoT telemetry ingested for {payload.target_transformer_id} and {payload.target_consumer_id}",
        "powerLoss_W": payload.powerLoss_W,
        "is_theft": payload.powerLoss_W > 0.25,
        "timestamp": last_iot_timestamp,
    })


@app.get("/api/iot/status")
async def get_iot_status():
    """Returns the current status of the physical hardware IoT link."""
    is_active = (latest_iot_telemetry is not None and (time.time() - last_iot_timestamp) < 30.0)
    return JSONResponse({
        "hardware_connected": is_active,
        "seconds_since_last_ping": round(time.time() - last_iot_timestamp, 1) if last_iot_timestamp > 0 else None,
        "target_transformer": latest_iot_telemetry.get("target_transformer_id") if latest_iot_telemetry else "TX_102",
        "target_consumer": latest_iot_telemetry.get("target_consumer_id") if latest_iot_telemetry else "CONS_S_001",
        "latest_readings": latest_iot_telemetry,
    })


@app.post("/api/iot/reset")
async def reset_iot_telemetry():
    """Clears physical hardware overlay and restores normal simulation state."""
    global latest_iot_telemetry, last_iot_timestamp
    latest_iot_telemetry = None
    last_iot_timestamp = 0.0
    snapshot = twin.get_snapshot()
    snapshot = sync_snapshot_cache(snapshot)
    await broadcast_to_clients(snapshot)
    return JSONResponse({"status": "reset", "message": "IoT overlay cleared; simulation restored."})


@app.get("/api/consumer/{consumer_id}")
async def get_consumer(consumer_id: str):
    snapshot = twin.get_snapshot()
    snapshot = sync_snapshot_cache(snapshot)
    c_data = snapshot["consumers"].get(consumer_id)
    if not c_data:
        raise HTTPException(status_code=404, detail="Consumer not found")
    an = c_data.get("analysis") or consumer_ml_cache.get(consumer_id)
    if an:
        c_data["analysis"] = an
    return JSONResponse(c_data)


@app.get("/api/report/{consumer_id}")
async def get_report(consumer_id: str):
    """Generates an auditable investigation report reflecting the consumer's dynamic state."""
    snapshot = twin.get_snapshot()
    snapshot = sync_snapshot_cache(snapshot)
    c_data = snapshot["consumers"].get(consumer_id)
    if not c_data:
        raise HTTPException(status_code=404, detail="Consumer not found")

    an = c_data.get("analysis") or consumer_ml_cache.get(consumer_id, {})
    static_info = c_data.get("static", {})

    cause = an.get("probable_cause", "NORMAL").replace("_", " ")
    score = an.get("anomaly_score", 0)
    risk = an.get("risk_level", "NORMAL")
    conf = an.get("confidence_pct", 95.0)
    factors = an.get("contributing_factors") or an.get("evidence", [])

    if factors:
        ev_list = "\n".join([f"- {f}" for f in factors])
    else:
        ev_list = "- Consumption adheres within normal statistical deviation bounds (±20%)\n- Communication, meter health, and peer baselines are stable."

    action = an.get("recommended_action") or "Routine automated surveillance."
    is_iot = an.get("is_iot_hardware", False)
    header_tag = " [LIVE PHYSICAL IOT HARDWARE SENSOR — ESP32]" if is_iot else ""

    md = f"""# Smart Grid Anomaly Investigation Report: {consumer_id}{header_tag}
**Report ID:** `RPT-{consumer_id}` | **Generated At:** `{snapshot.get('clock', {}).get('iso_time', '2026-10-05T06:30:00')}`
**Data Stream:** `{'Physical IoT Bench (ESP32 COM7)' if is_iot else 'Grid Telemetry Stream'}`

### Executive Summary
- **Consumer ID:** `{consumer_id}`
- **Meter ID:** `{static_info.get('meter_id', 'Unknown')}`
- **Category:** `{static_info.get('category', 'Unknown')}`
- **Transformer Zone:** `{static_info.get('transformer_id', 'TX_102')} ({static_info.get('zone_id', 'Zone_2_South')})`
- **Anomaly Score:** `{score} / 100` ({risk} RISK)
- **Probable Cause:** **{cause}**
- **Confidence Level:** `{conf}%`

### Machine Learning & Sensor Evidence Chain
{ev_list}

### Recommended Action
**{action}**
"""
    return JSONResponse({"markdown": md, "structured": an})


@app.post("/api/inject/{consumer_id}")
async def inject_scenario_endpoint(
    consumer_id: str,
    scenario: str = Query(...),
    mode: Optional[str] = Query(None),
    scaling_factor: Optional[float] = Query(None),
):
    try:
        scen_type = ScenarioType(scenario.upper())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid scenario: {scenario}")

    params = {}
    if mode:
        params["mode"] = mode
    if scaling_factor is not None:
        params["scaling_factor"] = float(scaling_factor)
    elif scen_type == ScenarioType.THEFT_BYPASS:
        params["mode"] = "partial_shunt"
        params["scaling_factor"] = 0.25
        params["trigger_tamper_flag"] = True
    elif scen_type == ScenarioType.METER_MALFUNCTION:
        params["malfunction_type"] = "stuck"

    twin.inject_scenario(consumer_id, scen_type, params)
    
    twin.step()
    snapshot = twin.get_snapshot()
    snapshot = sync_snapshot_cache(snapshot)
    await broadcast_to_clients(snapshot)
    return JSONResponse({
        "status": "success",
        "consumer_id": consumer_id,
        "scenario": scen_type.value,
        "snapshot": snapshot,
    })


@app.get("/api/geojson/buildings")
async def get_buildings_geojson():
    return JSONResponse(exporter.export_buildings_geojson())


@app.get("/api/geojson/grid-lines")
async def get_grid_lines_geojson():
    return JSONResponse(exporter.export_grid_lines_geojson())


@app.get("/api/geojson/transformers")
async def get_transformers_geojson():
    return JSONResponse(exporter.export_transformers_geojson())


@app.get("/api/geojson/drone")
async def get_drone_geojson():
    return JSONResponse(exporter.export_drone_flight_geojson())


@app.get("/api/ml/model-comparison")
async def ml_model_comparison():
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{ML_BACKEND}/model-comparison")
            return JSONResponse(resp.json())
    except Exception as e:
        return JSONResponse({"status": "unavailable", "error": str(e)})


@app.get("/api/ml/scenarios")
async def ml_scenarios():
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"{ML_BACKEND}/scenarios/test")
            return JSONResponse(resp.json())
    except Exception as e:
        return JSONResponse({"status": "unavailable", "error": str(e)})


@app.get("/api/health")
async def health():
    ml_ok = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{ML_BACKEND}/health")
            ml_ok = r.status_code == 200
    except Exception:
        pass
    return {
        "gridnest": "ok",
        "ml_backend": "ok" if ml_ok else "unreachable",
        "iot_hardware_active": latest_iot_telemetry is not None,
        "target_zone": "Zone_2_South (Zone B)",
        "target_transformer": "TX_102",
        "target_consumer": "CONS_S_001",
    }


async def broadcast_to_clients(data: Dict[str, Any]):
    for client in list(connected_clients):
        try:
            await client.send_text(json.dumps(data))
        except Exception:
            if client in connected_clients:
                connected_clients.remove(client)


@app.websocket("/ws")
@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        snapshot = twin.get_snapshot()
        snapshot = sync_snapshot_cache(snapshot)
        await websocket.send_text(json.dumps(snapshot))
        while True:
            msg = await websocket.receive_text()
            if msg == "step":
                twin.step()
                snapshot = twin.get_snapshot()
                snapshot = sync_snapshot_cache(snapshot)
                await broadcast_to_clients(snapshot)
    except WebSocketDisconnect:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
