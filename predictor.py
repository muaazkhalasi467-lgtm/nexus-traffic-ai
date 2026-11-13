import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score

df = pd.read_csv("traffic_history.csv")
df["road_num"] = df["road_id"].str.replace("R", "").astype(int)

features = ["road_num", "hour", "is_peak", "is_rain", "has_incident", "vehicle_count", "queue_length", "current_speed", "max_speed"]
X = df[features]
y_speed = df["future_speed_30m"]
y_risk = df["risk_label"]

X_train, X_test, ys_train, ys_test, yr_train, yr_test = train_test_split(X, y_speed, y_risk, test_size=0.2, random_state=42)

speed_model = RandomForestRegressor(n_estimators=80, random_state=42)
speed_model.fit(X_train, ys_train)

risk_model = RandomForestClassifier(n_estimators=80, random_state=42)
risk_model.fit(X_train, yr_train)

mae = mean_absolute_error(ys_test, speed_model.predict(X_test))
acc = accuracy_score(yr_test, risk_model.predict(X_test))

joblib.dump({"speed_model": speed_model, "risk_model": risk_model, "mae": round(mae, 2), "accuracy": round(acc * 100, 1)}, "congestion_model.pkl")
print(f"✅ Model Trained! 30-Min Speed MAE: {mae:.2f} km/h | Risk Accuracy: {acc*100:.1f}%")