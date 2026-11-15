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
    compare_all_models,
    evaluate_isolation_forest,
    evaluate_mlp_model,
    evaluate_random_forest,
    evaluate_xgboost,
    train_and_evaluate_all_models,
    train_isolation_forest,
    train_mlp_model,
    train_random_forest,
    train_xgboost,
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

    # Random Forest
    rf_model = train_random_forest(X_train, y_train, n_estimators=50)
    rf_results = evaluate_random_forest(rf_model, X_test, y_test)
    assert expected_keys.issubset(rf_results.keys())
    assert 0.0 <= rf_results["accuracy"] <= 1.0

    # XGBoost
    xgb_model = train_xgboost(X_train, y_train, n_estimators=50)
    xgb_results = evaluate_xgboost(xgb_model, X_test, y_test)
    assert xgb_results is not None
    assert expected_keys.issubset(xgb_results.keys())
    assert 0.0 <= xgb_results["accuracy"] <= 1.0

    # Multi-model comparison helper
    models_dict = {
        "Isolation Forest": if_model,
        "Random Forest": rf_model,
        "XGBoost": xgb_model,
        "MLP Neural Network": mlp_model,
    }
    comp_df, detailed = compare_all_models(models_dict, X_test, y_test)
    assert len(comp_df) == 4
    assert {"Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "False Positive Rate", "TP", "FP", "TN", "FN"}.issubset(comp_df.columns)
    assert len(detailed) == 4


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


def test_telemetry_dropout_and_ramping_fdi():
    """Verify newly introduced telemetry_dropout and ramping_fdi scenarios."""
    df_clean = generate_synthetic_grid_data(n_samples=60, attack_prob=0.0, random_seed=42)
    
    # 1. Telemetry dropout - all sensors
    df_drop, drop_idx = inject_dynamic_attack_scenario(
        df_clean,
        attack_type="telemetry_dropout",
        start_idx=10,
        duration=5,
        target_sensor="all",
        dropout_mode="zero",
    )
    for idx in drop_idx:
        assert df_drop.loc[idx, "voltage"] == 0.0
        assert df_drop.loc[idx, "current"] == 0.0
        assert df_drop.loc[idx, "load"] == 0.0
        assert df_drop.loc[idx, "frequency"] == 0.0
        assert df_drop.loc[idx, "target"] == 1
        assert df_drop.loc[idx, "attack_type"] == "telemetry_dropout"

    # 2. Telemetry dropout - single sensor flatline
    df_flat, flat_idx = inject_dynamic_attack_scenario(
        df_clean,
        attack_type="telemetry_dropout",
        start_idx=20,
        duration=5,
        target_sensor="voltage",
        dropout_mode="flatline",
    )
    for idx in flat_idx:
        assert df_flat.loc[idx, "voltage"] == df_clean.loc[19, "voltage"]
        assert df_flat.loc[idx, "target"] == 1
        assert df_flat.loc[idx, "attack_type"] == "telemetry_dropout"

    # 3. Ramping FDI
    df_ramp, ramp_idx = inject_dynamic_attack_scenario(
        df_clean,
        attack_type="ramping_fdi",
        start_idx=30,
        duration=10,
        magnitude=15.0,
        target_sensor="voltage",
        ramp_shape="linear",
    )
    assert len(ramp_idx) == 10
    assert (df_ramp.loc[ramp_idx, "target"] == 1).all()
    assert (df_ramp.loc[ramp_idx, "attack_type"] == "ramping_fdi").all()
    # Linear ramp ends at peak delta
    assert df_ramp.loc[ramp_idx[-1], "voltage"] == pytest.approx(df_clean.loc[ramp_idx[-1], "voltage"] + 15.0)
    assert df_ramp.loc[ramp_idx[0], "voltage"] < df_ramp.loc[ramp_idx[-1], "voltage"]

    # 4. Pipeline handles dropouts gracefully without NaN leakage
    pipe = LeakFreeGridPipeline()
    X = pipe.fit_transform(df_drop)
    assert not np.isnan(X).any()
    assert X.shape == (60, 6)


def test_incident_report_generation():
    """Verify HTML incident report generation with actual application data."""
    import report_generator

    telemetry_summary = {
        "total_samples": 1200,
        "suspicious_readings": 45,
        "suspicious_pct": 3.75,
        "mean_voltage": 229.8,
        "mean_frequency": 50.002,
        "mean_current": 9.85,
        "mean_load": 2.26,
    }
    attack_history = [
        {"type": "telemetry_dropout", "target_sensor": "frequency", "magnitude": "zero", "duration": 8, "indices": list(range(8))},
        {"type": "ramping_fdi", "target_sensor": "voltage", "magnitude": 15.0, "duration": 12, "indices": list(range(12))},
    ]
    model_df = pd.DataFrame([
        {"Model": "Random Forest", "Accuracy": 0.98, "Precision": 0.94, "Recall": 0.91, "F1-score": 0.925, "ROC-AUC": 0.985, "False Positive Rate": 0.01, "TP": 35, "FP": 2, "TN": 180, "FN": 3},
        {"Model": "XGBoost", "Accuracy": 0.975, "Precision": 0.93, "Recall": 0.89, "F1-score": 0.910, "ROC-AUC": 0.980, "False Positive Rate": 0.015, "TP": 34, "FP": 3, "TN": 179, "FN": 4},
    ])
    mitigation_actions = [
        "Trigger Fast Frequency Response (FFR) to stabilize secondary grid bus.",
        "Re-authenticate PMU sensors via zero-trust challenge-response.",
    ]

    html_out = report_generator.generate_incident_report_html(
        project_name="GridShield AI",
        telemetry_summary=telemetry_summary,
        attack_history=attack_history,
        model_comparison_df=model_df,
        mitigation_actions=mitigation_actions,
        active_threat_level="CRITICAL",
    )

    assert "<!DOCTYPE html>" in html_out
    assert "GridShield AI" in html_out
    assert "telemetry_dropout" in html_out
    assert "ramping_fdi" in html_out
    assert "Random Forest" in html_out
    assert "XGBoost" in html_out
    assert "Fast Frequency Response" in html_out
    assert "CRITICAL" in html_out


def test_cyber_physical_consistency_engine():
    """Verify physical Ohm/power consistency, RoCoF computation, and grid health index."""
    from cyber_physical import check_physical_power_consistency, compute_rocof, evaluate_grid_physics, get_cyber_physical_summary

    # Normal physics: V=230, I=10, pf=0.95 -> Expected P ~ 2.185 kW
    v = np.array([230.0, 230.0])
    i = np.array([10.0, 10.0])
    p_consistent = np.array([2.185, 2.185])
    p_inconsistent = np.array([2.185, 5.000])

    is_viol_ok, res_ok = check_physical_power_consistency(v, i, p_consistent)
    assert (is_viol_ok == 0).all()

    is_viol_bad, res_bad = check_physical_power_consistency(v, i, p_inconsistent)
    assert is_viol_bad[0] == 0
    assert is_viol_bad[1] == 1  # 5.0 kW vs ~2.185 kW is an electrical violation

    # RoCoF calculation: delta of 0.8 Hz in 1 second
    freqs = np.array([50.0, 50.8])
    rocof = compute_rocof(freqs, 1.0)
    assert rocof[1] == pytest.approx(0.8, rel=1e-2)

    # DataFrame evaluation
    df_sample = generate_synthetic_grid_data(n_samples=50, attack_prob=0.1, random_seed=42)
    df_audited = evaluate_grid_physics(df_sample)
    assert "cyber_physical_anomaly_score" in df_audited.columns
    assert "physics_power_violation" in df_audited.columns

    summary = get_cyber_physical_summary(df_audited)
    assert "grid_health_index" in summary
    assert 0.0 <= summary["grid_health_index"] <= 100.0


def test_pandapower_smart_grid_simulation():
    """Verify IEEE power network loading and power-flow execution."""
    from power_simulation import HAS_PANDAPOWER, load_network_case, run_power_flow_simulation

    if not HAS_PANDAPOWER:
        pytest.skip("pandapower not installed")

    net = load_network_case("IEEE 14-Bus System")
    res = run_power_flow_simulation(net, attack_bus_id=3, attack_type="load_fdi", magnitude=1.5)
    assert res["success"] is True
    assert res["n_buses"] == 14
    assert res["attack_applied"] is True
    assert "bus_df" in res
    assert "line_df" in res


def test_lstm_model_training_and_evaluation():
    """Verify PyTorch LSTM sequential anomaly detector train and evaluation."""
    from model_training import HAS_TORCH, train_lstm_model, evaluate_lstm_model

    if not HAS_TORCH:
        pytest.skip("PyTorch not installed")

    rng = np.random.default_rng(42)
    X_train = rng.normal(0, 1, size=(60, 6)).astype(np.float32)
    y_train = np.array([0] * 50 + [1] * 10, dtype=int)
    X_test = rng.normal(0, 1, size=(20, 6)).astype(np.float32)
    y_test = np.array([0] * 16 + [1] * 4, dtype=int)

    lstm = train_lstm_model(X_train, y_train, window_size=4, epochs=3, random_state=42)
    assert lstm is not None

    eval_out = evaluate_lstm_model(lstm, X_test, y_test)
    assert "accuracy" in eval_out
    assert "f1" in eval_out
    assert "roc_auc" in eval_out
    assert len(eval_out["predictions"]) == 20


def test_explainability_and_feature_importances():
    """Verify feature importances calculation and single prediction attribution."""
    from sklearn.ensemble import RandomForestClassifier
    from explainability import compute_feature_importances, explain_single_prediction

    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(40, 4))
    y = np.array([0] * 30 + [1] * 10)
    rf = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)

    feat_names = ["voltage", "current", "load", "frequency"]
    imp_df = compute_feature_importances({"Random Forest": rf}, feat_names)
    assert len(imp_df) == 4
    assert "Importance" in imp_df.columns
    assert np.isclose(imp_df["Importance"].sum(), 1.0, atol=1e-2)

    single_exp = explain_single_prediction(rf, X[0], baseline_background=X, feature_names=feat_names)
    assert "contributions" in single_exp
    assert single_exp["top_contributor"] in feat_names
    assert len(single_exp["contributions"]) == 4


def test_offline_sample_dataset_integrity():
    """Verify offline sample dataset is valid, non-empty, and loadable."""
    import os
    sample_file = os.path.join(os.path.dirname(__file__), "data", "sample", "smartgrid_telemetry_sample.csv")
    assert os.path.exists(sample_file)
    sample_df = pd.read_csv(sample_file)
    assert len(sample_df) >= 500
    is_valid, msg, cleaned = validate_grid_dataframe(sample_df)
    assert is_valid is True
    assert "voltage" in cleaned.columns
    assert "target" in cleaned.columns



