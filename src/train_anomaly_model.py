"""Train an Isolation Forest on rolling-window features from healthy motor data.

The model learns what normal vibration and sound look like. A window that is
unusual for that baseline gets a low score. The model and the scaler are saved
to models/ and used by live_monitor.py.

Usage:
    python src/train_anomaly_model.py [path/to/healthy_recording.csv]
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from config import ANOMALY_CONTAMINATION, ANOMALY_WINDOW, MODEL_DIR, PROCESSED_DIR

SIGNALS = ["sound", "ax", "ay"]

# Order matters: the live monitor builds its features in the same order
ANOMALY_FEATURES = [f"{s}_std" for s in SIGNALS] + [f"{s}_rms" for s in SIGNALS]


def rolling_features(df, window=ANOMALY_WINDOW):
    """Rolling standard deviation and RMS of each signal."""
    features = {}
    for col in SIGNALS:
        features[f"{col}_std"] = df[col].rolling(window).std()
    for col in SIGNALS:
        features[f"{col}_rms"] = np.sqrt((df[col] ** 2).rolling(window).mean())
    return pd.DataFrame(features, columns=ANOMALY_FEATURES).dropna()


def main(input_path):
    df = pd.read_csv(input_path)
    X = rolling_features(df)
    print(f"Feature matrix: {X.shape[0]} windows x {X.shape[1]} features")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=100, contamination=ANOMALY_CONTAMINATION, random_state=42
    )
    model.fit(X_scaled)

    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / "anomaly_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    preds = model.predict(X_scaled)
    flagged = int((preds == -1).sum())
    print(f"Flagged as anomalies in the training data: {flagged} ({flagged / len(preds):.2%})")
    print(f"Saved model and scaler to {MODEL_DIR}")


if __name__ == "__main__":
    default_input = PROCESSED_DIR / "sensor_data_healthy.csv"
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_input
    main(input_path)
