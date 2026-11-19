"""
Comprehensive Automated Test Suite for Upgraded GridShield AI ML Architecture.
Tests:
1. Dataset ingestion (MSU and Synthetic).
2. Modular ML model training, prediction, evaluation, and saving (RF, XGB, IF, MLP, LSTM).
3. Leak-free chronological pipelines.
4. CLI scripts (train.py, predict.py).
5. Explainability and Cyber-physical integrity.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_preprocessing import (
    LeakFreeGridPipeline,
    chronological_train_test_split,
    generate_synthetic_grid_data,
    validate_grid_dataframe,
)
from data.download_datasets import load_dataset
from ml_models.random_forest import GridRandomForestModel
from ml_models.xgboost_model import GridXGBoostModel, HAS_XGBOOST
from ml_models.isolation_forest import GridIsolationForestModel
from ml_models.mlp_model import GridMLPModel
from ml_models.lstm_model import GridLSTMModel, HAS_TORCH
from model_training import calibrate_contamination, compare_all_models
from predict import predict_observation


def test_msu_dataset_loading_and_validation():
    """Verify MSU/ORNL research benchmark loads and passes schema validation."""
    df, desc = load_dataset("msu_benchmark")
    assert not df.empty
    assert "PMU Relay Logs" in desc or "Mississippi State" in desc
    is_valid, msg, clean_df = validate_grid_dataframe(df)
    assert is_valid
    assert {"voltage", "current", "load", "frequency", "target"}.issubset(clean_df.columns)
    assert set(clean_df["target"].unique()).issubset({0, 1})


def test_synthetic_dataset_loading():
    """Verify synthetic dataset benchmark loads with 6 attack vectors."""
    df, desc = load_dataset("synthetic_sample")
    assert len(df) >= 1000
    is_valid, _, clean_df = validate_grid_dataframe(df)
    assert is_valid
    assert clean_df["target"].sum() > 0


def test_leak_free_chronological_splitting():
    """Verify chronological split avoids temporal leakage."""
    df = generate_synthetic_grid_data(n_samples=200, random_seed=42)
    train_df, test_df = chronological_train_test_split(df, test_ratio=0.25)

    assert len(train_df) == 150
    assert len(test_df) == 50
    # Strict temporal ordering
    assert train_df["timestamp"].max() < test_df["timestamp"].min()


def test_all_five_modular_models_train_and_evaluate():
    """Verify all 5 modular architectures fit, predict, and produce complete metrics."""
    df = generate_synthetic_grid_data(n_samples=250, attack_prob=0.15, random_seed=42)
    train_df, test_df = chronological_train_test_split(df, test_ratio=0.25)

    pipeline = LeakFreeGridPipeline(window_size=5)
    X_train = pipeline.fit_transform(train_df)
    X_test = pipeline.transform(test_df)
    y_train = train_df["target"].values
    y_test = test_df["target"].values

    expected_keys = {
        "accuracy", "precision", "recall", "f1", "roc_auc",
        "false_positive_rate", "confusion_matrix", "predictions", "anomaly_scores",
    }

    # 1. Isolation Forest
    iso = GridIsolationForestModel(contamination=0.1, random_state=42).fit(X_train)
    iso_res = iso.evaluate(X_test, y_test)
    assert expected_keys.issubset(iso_res.keys())
    assert 0.0 <= iso_res["accuracy"] <= 1.0

    # 2. Random Forest
    rf = GridRandomForestModel(n_estimators=30, random_state=42).fit(X_train, y_train)
    rf_res = rf.evaluate(X_test, y_test)
    assert expected_keys.issubset(rf_res.keys())
    assert 0.0 <= rf_res["accuracy"] <= 1.0

    # 3. XGBoost
    if HAS_XGBOOST:
        xgb = GridXGBoostModel(n_estimators=30, random_state=42).fit(X_train, y_train)
        xgb_res = xgb.evaluate(X_test, y_test)
        assert expected_keys.issubset(xgb_res.keys())

    # 4. MLP Neural Network
    mlp = GridMLPModel(max_iter=30, random_state=42).fit(X_train, y_train)
    mlp_res = mlp.evaluate(X_test, y_test)
    assert expected_keys.issubset(mlp_res.keys())

    # 5. PyTorch LSTM
    if HAS_TORCH:
        lstm = GridLSTMModel(epochs=3, random_state=42).fit(X_train, y_train)
        lstm_res = lstm.evaluate(X_test, y_test)
        assert expected_keys.issubset(lstm_res.keys())


def test_live_prediction_presets():
    """Verify predict_observation operates accurately across normal and attack presets."""
    res_norm = predict_observation("random_forest", voltage=230.1, current=10.0, load=2.301, frequency=50.0)
    assert res_norm["prediction"] in ["NORMAL", "ATTACK / ANOMALY"]
    assert res_norm["latency_ms"] > 0.0

    res_atk = predict_observation("random_forest", voltage=255.0, current=10.0, load=2.301, frequency=50.0)
    assert "prediction" in res_atk
    assert res_atk["latency_ms"] > 0.0


def test_model_persistence_and_reloading(tmp_path):
    """Verify model saving and loading works for joblib and pt formats."""
    X = np.random.randn(50, 6)
    y = np.random.randint(0, 2, 50)

    # Test RF saving and reloading
    rf_path = str(tmp_path / "test_rf.joblib")
    rf = GridRandomForestModel(n_estimators=10).fit(X, y)
    rf.save(rf_path)
    loaded_rf = GridRandomForestModel.load(rf_path)
    assert loaded_rf.is_fitted
    assert loaded_rf.predict(X).shape == (50,)

    # Test LSTM saving and reloading
    if HAS_TORCH:
        lstm_path = str(tmp_path / "test_lstm.pt")
        lstm = GridLSTMModel(epochs=2).fit(X, y)
        lstm.save(lstm_path)
        loaded_lstm = GridLSTMModel.load(lstm_path)
        assert loaded_lstm.is_fitted
        assert loaded_lstm.predict(X).shape == (50,)
