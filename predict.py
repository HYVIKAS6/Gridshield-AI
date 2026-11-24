"""
Live Prediction CLI and Standalone Inference Engine for GridShield AI.
Allows single-observation or sequence inference with trained models.

Usage:
    python predict.py --model random_forest --voltage 230.1 --current 10.0 --load 2.301 --frequency 50.0
    python predict.py --model isolation_forest --preset voltage_spike
    python predict.py --model lstm --preset normal
"""

import argparse
import os
import sys
import time
import joblib
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_models.random_forest import GridRandomForestModel
from ml_models.xgboost_model import GridXGBoostModel, HAS_XGBOOST
from ml_models.isolation_forest import GridIsolationForestModel
from ml_models.mlp_model import GridMLPModel
from ml_models.lstm_model import GridLSTMModel, HAS_TORCH

PRESETS = {
    "normal": {
        "voltage": 230.1,
        "current": 9.9,
        "load": 2.278,
        "frequency": 50.01,
        "description": "Nominal operational telemetry (P = V*I/1000 compliant)",
    },
    "voltage_spike": {
        "voltage": 252.4,
        "current": 10.1,
        "load": 2.323,
        "frequency": 49.98,
        "description": "Severe bus overvoltage (+22.4V transient spike)",
    },
    "frequency_drop": {
        "voltage": 229.8,
        "current": 9.8,
        "load": 2.252,
        "frequency": 48.95,
        "description": "Severe frequency sag (-1.05 Hz trip threshold excursion)",
    },
    "load_drift_fdia": {
        "voltage": 230.0,
        "current": 10.0,
        "load": 4.850,
        "frequency": 50.00,
        "description": "Uncoordinated false data injection breaking Ohm active power relationship",
    },
    "sensor_blackout": {
        "voltage": 0.0,
        "current": 0.0,
        "load": 0.0,
        "frequency": 0.0,
        "description": "Complete communication blackout / telemetry dropout",
    },
}


MODEL_FILES = {
    "random_forest": ("random_forest.joblib", GridRandomForestModel),
    "xgboost": ("xgboost_model.joblib", GridXGBoostModel),
    "isolation_forest": ("isolation_forest.joblib", GridIsolationForestModel),
    "mlp": ("mlp_model.joblib", GridMLPModel),
    "lstm": ("lstm_model.pt", GridLSTMModel),
}


def load_trained_model(model_name: str, saved_dir: str):
    if model_name not in MODEL_FILES:
        raise ValueError(f"Unknown model name: {model_name}. Choose from {list(MODEL_FILES.keys())}")

    filename, model_cls = MODEL_FILES[model_name]
    model_path = os.path.join(saved_dir, filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Run 'python train.py' first.")

    return model_cls.load(model_path)


def predict_observation(
    model_name: str,
    voltage: float,
    current: float,
    load: float,
    frequency: float,
    saved_dir: str = os.path.join(PROJECT_ROOT, "models", "saved"),
):
    model = load_trained_model(model_name, saved_dir)
    pipeline = joblib.load(os.path.join(saved_dir, "feature_pipeline.joblib"))
    return predict_with_model(model, pipeline, voltage, current, load, frequency)


def predict_with_model(model, pipeline, voltage, current, load, frequency):
    raw_df = pd.DataFrame([{
        "voltage": float(voltage),
        "current": float(current),
        "load": float(load),
        "frequency": float(frequency),
    }])

    start_t = time.perf_counter()
    X_scaled = pipeline.transform(raw_df)
    pred = model.predict(X_scaled)[0]
    prob_mat = model.predict_proba(X_scaled)
    prob = prob_mat[0, 1] if prob_mat.shape[1] > 1 else prob_mat[0, 0]
    latency_ms = (time.perf_counter() - start_t) * 1000.0

    return {
        "model_name": model.name,
        "prediction": "ATTACK / ANOMALY" if pred == 1 else "NORMAL",
        "is_attack": bool(pred == 1),
        "anomaly_score_or_prob": float(prob),
        "latency_ms": float(latency_ms),
        "inputs": {
            "voltage": voltage,
            "current": current,
            "load": load,
            "frequency": frequency,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="GridShield AI Live Prediction Engine")
    parser.add_argument("--model", type=str, default="random_forest", choices=["random_forest", "xgboost", "isolation_forest", "mlp", "lstm"])
    parser.add_argument("--preset", type=str, default=None, choices=list(PRESETS.keys()))
    parser.add_argument("--voltage", type=float, default=None)
    parser.add_argument("--current", type=float, default=None)
    parser.add_argument("--load", type=float, default=None)
    parser.add_argument("--frequency", type=float, default=None)
    args = parser.parse_args()

    if args.preset:
        preset_info = PRESETS[args.preset]
        v = preset_info["voltage"]
        i = preset_info["current"]
        p = preset_info["load"]
        f = preset_info["frequency"]
        print(f"\n[Preset Selected: {args.preset.upper()}] - {preset_info['description']}")
    else:
        v = args.voltage if args.voltage is not None else 230.0
        i = args.current if args.current is not None else 10.0
        p = args.load if args.load is not None else 2.30
        f = args.frequency if args.frequency is not None else 50.0

    res = predict_observation(args.model, voltage=v, current=i, load=p, frequency=f)

    print("\n" + "=" * 65)
    print(" ⚡ GRIDSHIELD AI — LIVE INFERENCE REPORT")
    print("=" * 65)
    print(f"Model Architecture:     {res['model_name']}")
    print(f"Telemetry Input:        V={v:.2f}V, I={i:.2f}A, P={p:.3f}kW, f={f:.2f}Hz")
    print(f"Diagnosis Verdict:      {'🚨 ' if res['is_attack'] else '🟢 '}{res['prediction']}")
    print(f"Confidence / Score:     {res['anomaly_score_or_prob'] * 100:.2f}%")
    print(f"Inference Latency:      {res['latency_ms']:.3f} ms")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
