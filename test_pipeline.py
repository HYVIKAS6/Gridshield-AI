import numpy as np
import pandas as pd
import pytest

from data_preprocessing import (
    LeakFreeGridPipeline,
    chronological_train_test_split,
    generate_synthetic_grid_data,
    inject_attack_at_indices,
    inject_dynamic_attack_scenario,
    validate_grid_dataframe,
)
from model_training import (
    calibrate_contamination,
    evaluate_isolation_forest,
    evaluate_mlp_model,
    train_isolation_forest,
    train_mlp_model,
)


def test_synthetic_data_generation_reproducibility():
    """Verify reproducible generation when seed is provided, and variation when changed."""
    df1 = generate_synthetic_grid_data(n_samples=200, attack_prob=0.1, random_seed=42)
    df2 = generate_synthetic_grid_data(n_samples=200, attack_prob=0.1, random_seed=42)
    df3 = generate_synthetic_grid_data(n_samples=200, attack_prob=0.1, random_seed=99)

    pd.testing.assert_frame_equal(df1, df2)
    assert not df1["voltage"].equals(df3["voltage"])
    assert len(df1) == 200
    assert "target" in df1.columns
    assert "attack_type" in df1.columns


def test_attack_injection_actually_modifies_telemetry():
    """Verify that attack injection changes telemetry values and flags targets."""
    df_clean = generate_synthetic_grid_data(n_samples=100, attack_prob=0.0, random_seed=123)
    assert (df_clean["target"] == 0).all()

    target_indices = [10, 25, 50]
    original_voltages = df_clean.loc[target_indices, "voltage"].copy()

    df_attacked = inject_attack_at_indices(
        df_clean,
        indices=target_indices,
        attack_type="voltage_spike",
        magnitude=12.0,
    )

    # 1. Telemetry values must change
    for idx in target_indices:
        assert df_attacked.loc[idx, "voltage"] == pytest.approx(original_voltages.loc[idx] + 12.0)
        assert df_attacked.loc[idx, "target"] == 1
        assert df_attacked.loc[idx, "attack_type"] == "voltage_spike"

    # 2. Non-attacked rows must remain unchanged
    unaffected = [i for i in range(100) if i not in target_indices]
    pd.testing.assert_frame_equal(df_clean.loc[unaffected], df_attacked.loc[unaffected])


def test_dynamic_attack_scenario_duration():
    """Verify dynamic attack injection over a time duration."""
    df_clean = generate_synthetic_grid_data(n_samples=100, attack_prob=0.0, random_seed=7)
    df_mod, mod_indices = inject_dynamic_attack_scenario(
        df_clean,
        attack_type="frequency_shift",
        start_idx=80,
        duration=10,
        magnitude=0.8,
    )
    assert len(mod_indices) == 10
    assert mod_indices == list(range(80, 90))
    assert (df_mod.loc[mod_indices, "target"] == 1).all()
    assert (df_mod.loc[:79, "target"] == 0).all()


def test_chronological_train_test_split():
    """Verify strict chronological splitting without shuffling."""
    df = generate_synthetic_grid_data(n_samples=100, attack_prob=0.1, random_seed=42)
    train_df, test_df = chronological_train_test_split(df, test_ratio=0.2)

    assert len(train_df) == 80
    assert len(test_df) == 20
    # Strict temporal boundary check: max train timestamp < min test timestamp
    assert train_df["timestamp"].max() < test_df["timestamp"].min()


def test_pipeline_leak_free_scaler_fit():
    """Verify that scaler is fitted strictly on training data."""
    train_df = generate_synthetic_grid_data(n_samples=100, attack_prob=0.0, random_seed=1)
    test_df = generate_synthetic_grid_data(n_samples=50, attack_prob=0.0, random_seed=2)

    pipeline = LeakFreeGridPipeline(window_size=5)
    pipeline.fit(train_df)

    assert pipeline.is_fitted
    scaler_mean_train = pipeline.scaler.mean_.copy()

    # Transforming test data must NOT alter the fitted training scaler parameters
    X_test_scaled = pipeline.transform(test_df)
    np.testing.assert_array_equal(pipeline.scaler.mean_, scaler_mean_train)
    assert X_test_scaled.shape == (50, len(LeakFreeGridPipeline.FEATURE_COLS))


def test_model_training_and_evaluation_on_unseen_test():
    """Verify models train and compute comprehensive evaluation metrics on unseen test data."""
    df = generate_synthetic_grid_data(n_samples=300, attack_prob=0.15, random_seed=42)
    train_df, test_df = chronological_train_test_split(df, test_ratio=0.25)

    pipeline = LeakFreeGridPipeline()
    X_train = pipeline.fit_transform(train_df)
    X_test = pipeline.transform(test_df)

    y_train = train_df["target"].values
    y_test = test_df["target"].values

    # Isolation Forest
    contam = calibrate_contamination(y_train)
    if_model = train_isolation_forest(X_train, contamination=contam)
    if_results = evaluate_isolation_forest(if_model, X_test, y_test)

    # Required metric keys
    expected_keys = {
        "accuracy", "precision", "recall", "f1", "roc_auc",
        "false_positive_rate", "confusion_matrix", "predictions", "anomaly_scores",
    }
    assert expected_keys.issubset(if_results.keys())
    assert 0.0 <= if_results["accuracy"] <= 1.0
    assert 0.0 <= if_results["roc_auc"] <= 1.0
    assert len(if_results["predictions"]) == len(y_test)

    # MLP Classifier
    mlp_model = train_mlp_model(X_train, y_train)
    mlp_results = evaluate_mlp_model(mlp_model, X_test, y_test)

    assert expected_keys.issubset(mlp_results.keys())
    assert 0.0 <= mlp_results["accuracy"] <= 1.0
    assert len(mlp_results["predictions"]) == len(y_test)


def test_dataframe_validation_error_handling():
    """Verify validation handles corrupted and empty DataFrames gracefully."""
    is_valid, msg, _ = validate_grid_dataframe(pd.DataFrame())
    assert not is_valid

    # Missing essential column
    corrupt_df = pd.DataFrame({"voltage": [230.0], "current": [10.0]})
    is_valid, msg, _ = validate_grid_dataframe(corrupt_df)
    assert not is_valid
    assert "Missing required" in msg

    # Valid DataFrame
    clean_df = generate_synthetic_grid_data(n_samples=10)
    is_valid, msg, df_out = validate_grid_dataframe(clean_df)
    assert is_valid
    assert len(df_out) == 10
