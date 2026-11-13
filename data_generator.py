import json
import random
import pandas as pd

with open("network_config.json", "r") as f:
    config = json.load(f)

roads = config["roads"]
rows = []

for _ in range(6000):
    road = random.choice(roads)
    hour = random.randint(0, 23)
    is_rain = random.choices([0, 1], weights=[0.75, 0.25])[0]
    has_incident = random.choices([0, 1], weights=[0.88, 0.12])[0]
    is_peak = 1 if (8 <= hour <= 11 or 17 <= hour <= 20) else 0

    base_load = random.uniform(0.2, 0.45)
    if is_peak: base_load += random.uniform(0.25, 0.45)
    if is_rain: base_load += random.uniform(0.10, 0.20)
    if has_incident: base_load += random.uniform(0.30, 0.50)

    load_ratio = min(1.35, base_load)
    vehicle_count = int(road["capacity_pcu"] * load_ratio)
    queue_length = max(0, int((load_ratio - 0.5) * 45)) if load_ratio > 0.5 else random.randint(0, 4)

    speed_factor = max(0.15, 1.0 - (load_ratio * 0.65) - (0.15 if is_rain else 0))
    current_speed = round(road["max_speed_kmh"] * speed_factor, 1)

    future_load = min(1.4, load_ratio + (0.12 if is_peak and hour in [8, 9, 17, 18] else -0.05) + (0.15 if has_incident else 0))
    future_speed = round(max(5.0, road["max_speed_kmh"] * max(0.12, 1.0 - (future_load * 0.65) - (0.15 if is_rain else 0))), 1)

    if future_speed < road["max_speed_kmh"] * 0.40:
        risk_label = "HIGH"
    elif future_speed < road["max_speed_kmh"] * 0.70:
        risk_label = "MEDIUM"
    else:
        risk_label = "LOW"

    rows.append({
        "road_id": road["id"],
        "hour": hour,
        "is_peak": is_peak,
        "is_rain": is_rain,
        "has_incident": has_incident,
        "vehicle_count": vehicle_count,
        "queue_length": queue_length,
        "current_speed": current_speed,
        "max_speed": road["max_speed_kmh"],
        "future_speed_30m": future_speed,
        "risk_label": risk_label
    })

df = pd.DataFrame(rows)
df.to_csv("traffic_history.csv", index=False)
print("✅ Generated traffic_history.csv with", len(df), "rows!")