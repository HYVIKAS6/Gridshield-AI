"""
GridShield AI - Intelligent Cyberattack Detection and Response Simulator for Smart Grids
Refactored Streamlit Web Application (Phase 1 & Phase 2).
"""

from typing import Any, Dict
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from data_preprocessing import (
    LeakFreeGridPipeline,
    chronological_train_test_split,
    generate_synthetic_grid_data,
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

# Page configuration
st.set_page_config(
    page_title="GridShield AI - Smart Grid Cyberattack Simulator",
    layout="wide",
    page_icon="⚡",
)

# CSS for modern professional appearance compatible with dark/light themes
st.markdown("""
<style>
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-normal { background-color: #27ae60; color: white; }
    .status-alert { background-color: #c0392b; color: white; }
    .status-warning { background-color: #d35400; color: white; }
</style>
""", unsafe_allow_html=True)

# App Header
st.title("⚡ GridShield AI")
st.markdown("### Intelligent Cyberattack Detection & Simulation Engine for Smart Grids")
st.caption("Physics-consistent telemetry generation • Leak-free ML evaluation • Dynamic stateful attack injection")

# ---------------------------------------------------------
# Session State Initialization (Stateful Simulation)
# ---------------------------------------------------------
if "sim_df" not in st.session_state:
    st.session_state.sim_df = None
if "sim_config" not in st.session_state:
    st.session_state.sim_config = {}
if "attack_history" not in st.session_state:
    st.session_state.attack_history = []
if "models_trained" not in st.session_state:
    st.session_state.models_trained = False
if "train_results" not in st.session_state:
    st.session_state.train_results = {}
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "active_data_source" not in st.session_state:
    st.session_state.active_data_source = "Synthetic Simulation"

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
st.sidebar.header("🕹️ Simulation Controls")

data_source = st.sidebar.selectbox(
    "Data Source",
    ["Synthetic Simulation", "Upload Dataset (CSV / Excel)"],
    index=0,
)

current_config: Dict[str, Any] = {"data_source": data_source}

if data_source == "Synthetic Simulation":
    n_samples = st.sidebar.slider("Number of Samples (Hours)", min_value=300, max_value=5000, value=1200, step=100)
    base_attack_prob = st.sidebar.slider("Baseline Attack Probability (%)", min_value=0, max_value=30, value=5, step=1) / 100.0
    random_seed = st.sidebar.number_input("Random Seed", value=42, step=1)
    
    current_config.update({
        "n_samples": n_samples,
        "attack_prob": base_attack_prob,
        "seed": int(random_seed),
    })

    # Invalidate models and reset dataset if parameters changed
    if (
        st.session_state.sim_df is None
        or st.session_state.sim_config != current_config
        or st.session_state.active_data_source != data_source
    ):
        fresh_df = generate_synthetic_grid_data(
            n_samples=n_samples,
            attack_prob=base_attack_prob,
            random_seed=int(random_seed),
        )
        st.session_state.sim_df = fresh_df
        st.session_state.sim_config = current_config
        st.session_state.active_data_source = data_source
        st.session_state.attack_history = []
        # Clear stale model results on configuration change
        st.session_state.models_trained = False
        st.session_state.train_results = {}

    if st.sidebar.button("🔄 Regenerate Fresh Baseline"):
        fresh_df = generate_synthetic_grid_data(
            n_samples=n_samples,
            attack_prob=base_attack_prob,
            random_seed=int(random_seed),
        )
        st.session_state.sim_df = fresh_df
        st.session_state.attack_history = []
        st.session_state.models_trained = False
        st.session_state.train_results = {}
        st.sidebar.success("Fresh simulation data generated.")
        st.rerun()

else:
    # Upload mode
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_excel(uploaded_file)
            
            is_valid, val_msg, cleaned_df = validate_grid_dataframe(raw_df)
            if is_valid:
                # Check if new file uploaded
                if st.session_state.sim_config.get("filename") != uploaded_file.name:
                    st.session_state.sim_df = cleaned_df
                    st.session_state.sim_config = {"filename": uploaded_file.name, "rows": len(cleaned_df)}
                    st.session_state.active_data_source = data_source
                    st.session_state.attack_history = []
                    st.session_state.models_trained = False
                    st.session_state.train_results = {}
                    st.sidebar.success(val_msg)
            else:
                st.sidebar.error(val_msg)
        except Exception as e:
            st.sidebar.error(f"Error parsing uploaded file: {str(e)}")
    else:
        st.sidebar.info("Awaiting file upload. You can switch back to 'Synthetic Simulation' anytime.")

df = st.session_state.sim_df

if df is None or df.empty:
    st.warning("⚠️ No active telemetry data found. Select Synthetic Simulation or upload a valid dataset.")
    st.stop()

# Ensure timestamp is datetime
if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
    df["timestamp"] = pd.to_datetime(df["timestamp"])

# ---------------------------------------------------------
# Sidebar: Interactive FDIA Attack Console
# ---------------------------------------------------------
st.sidebar.divider()
st.sidebar.header("🎯 Live Attack Injection Studio")
attack_choice = st.sidebar.selectbox(
    "Attack Type",
    ["voltage_spike", "frequency_shift", "load_drift", "stealth_fdia"],
    format_func=lambda x: {
        "voltage_spike": "Voltage Spike / Surge (+V)",
        "frequency_shift": "Frequency Sag / Shift (±f)",
        "load_drift": "Load Inconsistency / Drift (*P)",
        "stealth_fdia": "Stealth Coordinated FDIA (V, I, P)",
    }.get(x, x),
)

attack_duration = st.sidebar.slider("Attack Duration (Hours)", min_value=1, max_value=min(50, len(df)), value=12)
magnitude_defaults = {
    "voltage_spike": 12.0,
    "frequency_shift": 0.8,
    "load_drift": 1.4,
    "stealth_fdia": 1.25,
}
attack_magnitude = st.sidebar.number_input(
    "Perturbation Magnitude / Scale",
    value=magnitude_defaults.get(attack_choice, 1.0),
    step=0.1,
)

col_atk1, col_atk2 = st.sidebar.columns(2)
with col_atk1:
    inject_btn = st.button("🔴 INJECT ATTACK", use_container_width=True)
with col_atk2:
    clear_btn = st.button("🧹 Clear Attacks", use_container_width=True)

if inject_btn:
    # Manipulate telemetry over the chosen window
    start_point = max(0, len(df) - attack_duration)
    modified_df, mod_indices = inject_dynamic_attack_scenario(
        df,
        attack_type=attack_choice,
        start_idx=start_point,
        duration=attack_duration,
        magnitude=float(attack_magnitude),
    )
    st.session_state.sim_df = modified_df
    st.session_state.attack_history.append({
        "type": attack_choice,
        "duration": attack_duration,
        "start_idx": start_point,
        "indices": mod_indices,
    })
    # Signal model rerun required or re-predict if model is active
    if st.session_state.models_trained:
        st.session_state.models_trained = False
        st.sidebar.warning("Telemetry changed! Please re-train models to update metrics.")
    st.sidebar.success(f"Injected {attack_choice} across {len(mod_indices)} intervals!")
    st.rerun()

if clear_btn:
    # Reset target and attack_type
    df_reset = df.copy()
    df_reset["target"] = 0
    df_reset["attack_type"] = "none"
    # Recalculate physical load
    df_reset["load"] = (df_reset["voltage"] * df_reset["current"]) / 1000.0
    st.session_state.sim_df = df_reset
    st.session_state.attack_history = []
    st.session_state.models_trained = False
    st.session_state.train_results = {}
    st.sidebar.info("All injected attacks cleared.")
    st.rerun()

# ---------------------------------------------------------
# Sidebar: AI Model Training Controls
# ---------------------------------------------------------
st.sidebar.divider()
st.sidebar.header("🧠 Leak-Free ML Engine")
test_split_ratio = st.sidebar.slider("Test Split Ratio (Chronological)", 0.1, 0.4, 0.2, 0.05)
train_models_btn = st.sidebar.button("🚀 Train & Evaluate Models", type="primary", use_container_width=True)

if train_models_btn:
    with st.spinner("Executing leak-free chronological training & evaluation..."):
        # 1. Chronological splitting
        train_df, test_df = chronological_train_test_split(df, test_ratio=test_split_ratio)
        
        # 2. Fit pipeline strictly on train_df
        pipeline = LeakFreeGridPipeline(window_size=5)
        X_train = pipeline.fit_transform(train_df)
        X_test = pipeline.transform(test_df)
        
        y_train = train_df["target"].values
        y_test = test_df["target"].values
        
        # 3. Train Isolation Forest (Dynamic Contamination)
        contam = calibrate_contamination(y_train)
        if_model = train_isolation_forest(X_train, contamination=contam)
        if_results = evaluate_isolation_forest(if_model, X_test, y_test)
        
        # 4. Train Supervised Neural Network (MLP)
        mlp_model = train_mlp_model(X_train, y_train)
        mlp_results = evaluate_mlp_model(mlp_model, X_test, y_test)
        
        # 5. Persist in session state
        st.session_state.pipeline = pipeline
        st.session_state.train_results = {
            "train_df": train_df,
            "test_df": test_df,
            "y_train": y_train,
            "y_test": y_test,
            "if_model": if_model,
            "if_results": if_results,
            "mlp_model": mlp_model,
            "mlp_results": mlp_results,
            "split_ratio": test_split_ratio,
        }
        st.session_state.models_trained = True
        st.success("✅ Models trained using strict leak-free temporal splitting!")

# ---------------------------------------------------------
# Main Panel: Grid Telemetry Visualization
# ---------------------------------------------------------
total_samples = len(df)
attack_samples = int((df["target"] == 1).sum())
attack_pct = (attack_samples / total_samples) * 100 if total_samples > 0 else 0.0

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric("Total Intervals", f"{total_samples:,} hrs")
with col_m2:
    st.metric("Compromised Intervals", f"{attack_samples} ({attack_pct:.1f}%)")
with col_m3:
    avg_v = df["voltage"].mean()
    st.metric("Mean Voltage", f"{avg_v:.2f} V")
with col_m4:
    avg_f = df["frequency"].mean()
    st.metric("Mean Frequency", f"{avg_f:.3f} Hz")

st.subheader("📈 Multi-Sensor Telemetry & Cyberattack Overlay")

# Multi-trace figure with Voltage, Current, Load, and Frequency
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Voltage (V) with Cyber Threats", "System Frequency (Hz) Normal: [49.8 - 50.2]", "Current (A)", "Active Power / Load (kW)"),
    shared_xaxes=True,
    vertical_spacing=0.12,
    horizontal_spacing=0.08,
)

# 1. Voltage
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["voltage"], name="Voltage", line=dict(color="#2980b9", width=1.5)), row=1, col=1)
attacks = df[df["target"] == 1]
if not attacks.empty:
    fig.add_trace(go.Scatter(
        x=attacks["timestamp"], y=attacks["voltage"],
        mode="markers", name="Attack Active",
        marker=dict(color="#e74c3c", size=7, symbol="cross"),
    ), row=1, col=1)

# 2. Frequency with safety corridor
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["frequency"], name="Frequency", line=dict(color="#27ae60", width=1.5)), row=1, col=2)
fig.add_hline(y=50.2, line_dash="dash", line_color="orange", row=1, col=2)
fig.add_hline(y=49.8, line_dash="dash", line_color="orange", row=1, col=2)

# 3. Current
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["current"], name="Current", line=dict(color="#8e44ad", width=1.5)), row=2, col=1)

# 4. Load
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["load"], name="Active Load", line=dict(color="#f39c12", width=1.5)), row=2, col=2)

fig.update_layout(height=520, margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified", showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# Model Performance & Leak-Free Evaluation Section
# ---------------------------------------------------------
if st.session_state.models_trained:
    res = st.session_state.train_results
    if_res = res["if_results"]
    mlp_res = res["mlp_results"]
    test_df = res["test_df"]
    
    st.divider()
    st.subheader("📊 Model Performance on Unseen Chronological Test Data")
    st.caption(f"Evaluated on {len(test_df)} chronologically held-out observations (test split: {res['split_ratio']:.0%}).")

    col_eval1, col_eval2 = st.columns(2)

    with col_eval1:
        st.markdown("#### 🌲 Isolation Forest (Unsupervised)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{if_res['accuracy']:.1%}")
        m2.metric("Precision", f"{if_res['precision']:.2f}")
        m3.metric("Recall", f"{if_res['recall']:.2f}")
        m4.metric("F1 Score", f"{if_res['f1']:.2f}")

        m5, m6 = st.columns(2)
        m5.metric("ROC-AUC", f"{if_res['roc_auc']:.2f}")
        m6.metric("False Positive Rate", f"{if_res['false_positive_rate']:.1%}")

        # Confusion matrix visual
        cm_if = np.array(if_res["confusion_matrix"])
        fig_cm_if = px.imshow(
            cm_if,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Normal", "Attack"],
            y=["Normal", "Attack"],
            title="Isolation Forest: Confusion Matrix",
        )
        fig_cm_if.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_cm_if, use_container_width=True)

    with col_eval2:
        st.markdown("#### 🧠 MLP Neural Network (Supervised)")
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Accuracy", f"{mlp_res['accuracy']:.1%}")
        n2.metric("Precision", f"{mlp_res['precision']:.2f}")
        n3.metric("Recall", f"{mlp_res['recall']:.2f}")
        n4.metric("F1 Score", f"{mlp_res['f1']:.2f}")

        n5, n6 = st.columns(2)
        n5.metric("ROC-AUC", f"{mlp_res['roc_auc']:.2f}")
        n6.metric("False Positive Rate", f"{mlp_res['false_positive_rate']:.1%}")

        # Confusion matrix visual
        cm_mlp = np.array(mlp_res["confusion_matrix"])
        fig_cm_mlp = px.imshow(
            cm_mlp,
            text_auto=True,
            color_continuous_scale="Greens",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Normal", "Attack"],
            y=["Normal", "Attack"],
            title="MLP Classifier: Confusion Matrix",
        )
        fig_cm_mlp.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_cm_mlp, use_container_width=True)

    # ---------------------------------------------------------
    # Intelligent Incident Mitigation (Chronological Slicing)
    # ---------------------------------------------------------
    st.divider()
    st.subheader("🛡️ Real-Time Incident Assessment & Action Advisory")

    # Inspect the most recent chronological predictions from the held-out test sequence
    recent_window_size = min(15, len(mlp_res["predictions"]))
    recent_preds = mlp_res["predictions"][-recent_window_size:]
    recent_attacks_detected = sum(recent_preds)
    recent_test_slice = test_df.iloc[-recent_window_size:]

    if recent_attacks_detected >= 2:
        st.error(f"🚨 ACTIVE CYBER INCIDENT: Detected {recent_attacks_detected} anomalous conditions across the last {recent_window_size} chronological intervals.")
        
        # Determine specific sensor deviations
        freq_dev = abs(recent_test_slice["frequency"].mean() - 50.0)
        volt_dev = abs(recent_test_slice["voltage"].mean() - 230.0)
        
        rec_list = []
        if freq_dev > 0.15:
            rec_list.append("• **Frequency Anomaly:** Trigger Fast Frequency Response (FFR) and dispatch spinning reserves to prevent generator tripping.")
        if volt_dev > 2.0:
            rec_list.append("• **Voltage Anomaly:** Verify tap-changer controllers on substation transformers and re-calibrate PMU bus telemetry.")
        rec_list.append("• **SCADA Re-Authentication:** Enforce zero-trust handshake for affected PMUs and switch secondary bus to local islanded mode.")

        for r in rec_list:
            st.markdown(r)
    else:
        st.success(f"✅ System Telemetry Stable. No active cyber threat pattern detected across recent {recent_window_size} operating intervals.")

else:
    st.info("👈 Click **'Train & Evaluate Models'** in the sidebar to run the leak-free evaluation engine.")

# Raw data expander
with st.expander("🔍 View Raw Grid Telemetry Data"):
    st.dataframe(df.tail(20), use_container_width=True)
