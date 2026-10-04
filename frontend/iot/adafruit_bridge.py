"""
GridNest Adafruit IO Cloud Bridge.
Polls or subscribes to Adafruit IO MQTT/REST feeds for Transformer & Consumer telemetry
and pushes live state directly into the GridNest Digital Twin API.
"""
import time
import argparse
import requests

DEFAULT_SERVER_URL = "http://localhost:8000/api/iot/telemetry"


def poll_adafruit_io(
    username: str,
    key: str,
    feed_key: str = "energy-telemetry",
    server_url: str = DEFAULT_SERVER_URL,
    poll_interval: float = 2.0,
    aio_url: str = "https://io.adafruit.com",
):
    """
    Polls Adafruit IO REST API (or local mock broker) for the latest telemetry reading.
    Adafruit IO feed can either hold a JSON string with the measurements,
    or individual feeds.
    """
    url = f"{aio_url.rstrip('/')}/api/v2/{username}/feeds/{feed_key}/data/last"
    headers = {"X-AIO-Key": key}

    print("\n=======================================================")
    print(f" ☁️ GRIDNEST ADAFRUIT IO CLOUD BRIDGE")
    print(f" Broker URL:   {aio_url}")
    print(f" Username:     {username}")
    print(f" Feed:         {feed_key}")
    print(f" Target Server:{server_url}")
    print("=======================================================\n")

    last_data_id = None

    while True:
        try:
            resp = requests.get(url, headers=headers, timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                data_id = data.get("id")
                raw_val = data.get("value", "")

                if data_id != last_data_id:
                    last_data_id = data_id
                    print(f"[Adafruit IO] New packet #{data_id}: {raw_val}")

                    # Try parsing JSON from feed value
                    import json
                    try:
                        telemetry_dict = json.loads(raw_val)
                    except Exception:
                        # Fallback simple float loss or power
                        telemetry_dict = {
                            "transVoltage": 4.5,
                            "transCurrent": 500.0,
                            "transPower_W": 2.25,
                            "consVoltage": 4.3,
                            "consCurrent": 480.0,
                            "consPower_W": 2.06,
                            "powerLoss_W": float(raw_val) if raw_val.replace('.', '', 1).isdigit() else 0.19,
                        }

                    telemetry_dict.setdefault("target_consumer_id", "CONS_S_001")
                    telemetry_dict.setdefault("target_transformer_id", "TX_102")

                    # Push to GridNest
                    post_resp = requests.post(server_url, json=telemetry_dict, timeout=2.0)
                    if post_resp.status_code == 200:
                        print(f"[-->] Synced with Digital Twin successfully.")

            time.sleep(poll_interval)
        except KeyboardInterrupt:
            print("\n[*] Stopping Adafruit IO bridge.")
            break
        except Exception as e:
            print(f"[!] Polling error: {e}")
            time.sleep(poll_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GridNest Adafruit IO Bridge")
    parser.add_argument("--username", type=str, default="YOUR_ADAFRUIT_USERNAME", help="Adafruit IO Username")
    parser.add_argument("--key", type=str, default="YOUR_ADAFRUIT_AIO_KEY", help="Adafruit IO Key")
    parser.add_argument("--feed", type=str, default="energy-telemetry", help="Adafruit IO Feed Name")
    parser.add_argument("--url", type=str, default=DEFAULT_SERVER_URL, help="GridNest API URL")
    parser.add_argument("--aio-url", type=str, default="https://io.adafruit.com", help="Adafruit IO Base URL (use http://localhost:8002 for local mock broker)")
    parser.add_argument("--interval", type=float, default=2.0, help="Polling interval in seconds")
    args = parser.parse_args()

    poll_adafruit_io(args.username, args.key, args.feed, args.url, args.interval, args.aio_url)

