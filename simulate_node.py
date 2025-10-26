# node_simulator.py
import requests
import time
import random
from datetime import datetime

# === CONFIG ===
API_BASE = "https://polygraph-w.ddns.net/api"  # URL zu deinem FastAPI Server
TOKEN = "API_TOKEN"           # Token aus Streamlit UI

CONFIG_POLL_INTERVAL = 10  # s, wie oft neue Config geladen wird

# === HELPERS ===
def fetch_config():
    try:
        r = requests.get(f"{API_BASE}/config", params={"token": TOKEN}, timeout=5)
        r.raise_for_status()
        cfg = r.json()
        return cfg
    except Exception as e:
        print(f"[{datetime.utcnow()}] Fehler beim Laden der Config: {e}")
        return None

def generate_measurement(cfg):
    gsr = round(random.uniform(*cfg["gsr_range"]), 3)
    pulse = round(random.uniform(*cfg["pulse_range"]))
    # Humidity & Pressure werden im Streamlit eingegeben, Node sendet None
    data = {
        "token": TOKEN,
        "device_id": cfg["device_id"],
        "gsr": gsr,
        "pulse": pulse,
        "humidity": cfg.get("humidity"),
        "pressure": cfg.get("pressure"),
        "metadata": {"simulated": True}
    }
    return data

def send_measurement(data):
    try:
        r = requests.post(f"{API_BASE}/upload", json=data, timeout=5)
        r.raise_for_status()
        print(f"[{datetime.utcnow()}] Upload OK: GSR={data['gsr']}, Pulse={data['pulse']}, Humidity={data['humidity']}, Pressure={data['pressure']}")
    except Exception as e:
        print(f"[{datetime.utcnow()}] Upload failed: {e}")

# === MAIN LOOP ===
def main():
    last_cfg_fetch = 0
    cfg = fetch_config()
    if cfg is None:
        print("Keine Config, Exit")
        return

    while True:
        now = time.time()
        # Config alle CONFIG_POLL_INTERVAL Sekunden neu laden
        if now - last_cfg_fetch > CONFIG_POLL_INTERVAL:
            new_cfg = fetch_config()
            if new_cfg:
                cfg = new_cfg
                print(f"[{datetime.utcnow()}] Config updated: {cfg}")
            last_cfg_fetch = now

        # Messung generieren und senden
        data = generate_measurement(cfg)
        send_measurement(data)

        # Intervall aus Config
        interval = cfg.get("interval", 2)
        time.sleep(interval)

if __name__ == "__main__":
    main()
