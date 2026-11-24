import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest


def test_app_startup_and_initial_state():
    """Verify Streamlit app initializes with default simulation data in session_state."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception
    assert at.session_state["sim_df"] is not None
    assert len(at.session_state["sim_df"]) == 1200
    assert at.session_state["models_trained"] is False


def test_app_attack_injection_persists_in_session_state():
    """Verify that clicking the inject button modifies telemetry and persists across reruns."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception

    initial_targets = int((at.session_state["sim_df"]["target"] == 1).sum())

    # Find the '🔴 INJECT ATTACK' button and click it
    inject_buttons = [btn for btn in at.button if "INJECT ATTACK" in btn.label]
    assert len(inject_buttons) == 1
    inject_buttons[0].click().run()

    assert not at.exception
    new_targets = int((at.session_state["sim_df"]["target"] == 1).sum())
    assert new_targets > initial_targets
    assert len(at.session_state["attack_history"]) == 1


def test_app_train_models_chronological_flow():
    """Verify training models runs leak-free chronological evaluation and populates metrics."""
    at = AppTest.from_file("app.py", default_timeout=45)
    at.run()
    assert not at.exception

    train_buttons = [btn for btn in at.button if "Train & Evaluate Models" in btn.label]
    assert len(train_buttons) == 1
    train_buttons[0].click().run()

    assert not at.exception
    assert at.session_state["models_trained"] is True
    assert "if_results" in at.session_state["train_results"]
    assert "mlp_results" in at.session_state["train_results"]

    # Verify metrics are within valid probability / scoring bounds
    if_res = at.session_state["train_results"]["if_results"]
    mlp_res = at.session_state["train_results"]["mlp_results"]
    assert 0.0 <= if_res["accuracy"] <= 1.0
    assert 0.0 <= mlp_res["accuracy"] <= 1.0
    assert 0.0 <= if_res["roc_auc"] <= 1.0
    assert 0.0 <= mlp_res["roc_auc"] <= 1.0


def test_studio_training_and_live_prediction(monkeypatch):
    """Fresh dashboard sessions can train and predict without CLI checkpoints."""
    import predict

    def unexpected_checkpoint(*args):
        pytest.fail("Dashboard prediction tried to load a disk checkpoint")

    monkeypatch.setattr(predict, "load_trained_model", unexpected_checkpoint)
    at = AppTest.from_file("app.py", default_timeout=60).run()
    at.multiselect(key="studio_models_selected").set_value(["Random Forest (Supervised Bagging)"])
    next(btn for btn in at.button if "Start Live Model Training" in btn.label).click().run()
    assert not at.exception
    assert list(at.session_state["train_results"]["comparison_df"]["Model"]) == ["Random Forest"]

    preset = at.selectbox(key="lab_preset_choice")
    preset.select(next(label for label in preset.options if "Voltage Spike" in label)).run()
    assert at.number_input(key="lab_v").value == 252.4
    next(btn for btn in at.button if "Run Live Model Prediction" in btn.label).click().run()
    assert not at.exception
    assert any("VERDICT" in alert.value for alert in [*at.success, *at.error])
    assert len(at.tabs) == 12
