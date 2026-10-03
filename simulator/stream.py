"""
Real-Time Telemetry Stream Simulator.
Streams simulated meter readings with historical context to the prediction pipeline or console.
"""
import argparse
import time
import json
import random
import requests
from simulator.scenarios import build_scenario_matrix
from ml.predict import predict_single_payload

def stream_telemetry(
    scenario: str = "PERSISTENT_LOW_THEFT",
    consumer_id: str = "C0001",
    api_url: str = None,
    delay: float = 1.0,
    steps: int = 15
):
    print(f"\n[*] Starting real-time stream simulation for consumer '{consumer_id}'...")
    print(f"[*] Scenario: {scenario}")
    print(f"[*] Target Endpoint: {api_url or 'Local Python Pipeline'}\n")

    matrix, dates = build_scenario_matrix(scenario, num_days=90)
    series = matrix[0]

    for step in range(steps):
        day_idx = 75 + step
        c_val = float(series[day_idx])
        history = [float(x) for x in series[day_idx-30:day_idx]]

        payload = {
            "consumer_id": consumer_id,
            "meter_id": f"M_{consumer_id}",
            "energy_kwh": c_val,
            "timestamp": dates[day_idx].strftime("%Y-%m-%d"),
            "recent_history_30d": history
        }

        if api_url:
            try:
                res = requests.post(f"{api_url}/predict", json=payload, timeout=5)
                output = res.json()
            except Exception as e:
                output = {"error": str(e)}
        else:
            output = predict_single_payload(payload)

        print(f"[{payload['timestamp']}] Reading: {payload['energy_kwh']:.2f} kWh | Cause: {output.get('probable_cause')} | Risk: {output.get('risk_score')}/100 ({output.get('priority')})")
        time.sleep(delay)

    print("\n[+] Streaming simulation completed.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream simulated smart-meter telemetry.")
    parser.add_argument("--scenario", type=str, default="PERSISTENT_LOW_THEFT", help="Scenario name")
    parser.add_argument("--consumer", type=str, default="C0001", help="Consumer ID")
    parser.add_argument("--api", type=str, default=None, help="Optional API URL (e.g. http://localhost:8000)")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay in seconds between readings")
    parser.add_argument("--steps", type=int, default=10, help="Number of readings to stream")
    args = parser.parse_args()

    stream_telemetry(
        scenario=args.scenario,
        consumer_id=args.consumer,
        api_url=args.api,
        delay=args.delay,
        steps=args.steps
    )
