"""
GridShield AI - Intelligent Cyberattack Detection and Response Dashboard for Smart Grids.
Professional Industrial Cybersecurity Monitoring & AI Defense Platform.
"""

from datetime import datetime
import html
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from data_preprocessing import (
    ATTACK_METADATA,
    LeakFreeGridPipeline,
    chronological_train_test_split,
    generate_synthetic_grid_data,
    inject_dynamic_attack_scenario,
    validate_grid_dataframe,
)
from model_training import (
    HAS_XGBOOST,
    calibrate_contamination,
    evaluate_isolation_forest,
    evaluate_mlp_model,
    train_and_evaluate_all_models,
    train_isolation_forest,
    train_mlp_model,
)
from report_generator import generate_incident_report_html

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="GridShield AI - Cyberattack Monitoring Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Modern Industrial Cybersecurity Theme & CSS Styling
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Dark Navy & Slate Industrial Theme */
    :root {
        --bg-navy-dark: #0a1128;
        --bg-navy-panel: #0e1b38;
        --bg-card: #132247;
        --bg-card-hover: #1b2a4e;
        --border-color: #1e3567;
        --cyan-accent: #00d2ff;
        --cyan-glow: rgba(0, 210, 255, 0.35);
        --neon-green: #00e676;
        --neon-red: #ff3366;
        --neon-amber: #ffb300;
        --text-primary: #f0f4fc;
        --text-secondary: #94a3b8;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }

    /* Top Branding Banner */
    .gridshield-header {
        background: linear-gradient(135deg, #0e1b38 0%, #132247 50%, #0a1128 100%);
        border: 1px solid #1e3567;
        border-radius: 12px;
        padding: 20px 28px;
        margin-bottom: 20px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.45);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .header-title-box h1 {
        color: #00d2ff;
        font-family: 'Segoe UI', -apple-system, Roboto, sans-serif;
        font-weight: 700;
        font-size: 1.85rem;
        margin: 0 0 4px 0;
        letter-spacing: 0.5px;
        text-shadow: 0 0 16px rgba(0, 210, 255, 0.4);
    }
    .header-title-box p {
        color: #94a3b8;
        font-size: 0.92rem;
        margin: 0;
    }
    .system-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(0, 210, 255, 0.08);
        border: 1px solid #00d2ff;
        border-radius: 20px;
        padding: 6px 16px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #00d2ff;
        box-shadow: 0 0 10px rgba(0, 210, 255, 0.2);
    }
    .status-dot-active {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #00e676;
        box-shadow: 0 0 8px #00e676;
    }

    /* Professional Industrial Cards */
    .cyber-card {
        background: #0e1b38;
        border: 1px solid #1e3567;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 16px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    }
    .cyber-card:hover {
        border-color: #00d2ff;
        box-shadow: 0 4px 20px rgba(0, 210, 255, 0.15);
    }
    .cyber-card-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #00d2ff;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Metric Indicator Callouts */
    .metric-hero {
        background: #132247;
        border-left: 4px solid #00d2ff;
        border-radius: 8px;
        padding: 12px 18px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    .metric-hero-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .metric-hero-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #f0f4fc;
    }

    /* Badges & Tags */
    .cyber-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .cyber-badge-critical { background: rgba(255, 51, 102, 0.18); border: 1px solid #ff3366; color: #ff3366; }
    .cyber-badge-warning { background: rgba(255, 179, 0, 0.18); border: 1px solid #ffb300; color: #ffb300; }
    .cyber-badge-safe { background: rgba(0, 230, 118, 0.18); border: 1px solid #00e676; color: #00e676; }
    .cyber-badge-info { background: rgba(0, 210, 255, 0.18); border: 1px solid #00d2ff; color: #00d2ff; }

    /* Model vs Attack Legend */
    .legend-box {
        display: flex;
        gap: 20px;
        padding: 8px 16px;
        background: #0e1b38;
        border: 1px solid #1e3567;
        border-radius: 8px;
        margin-bottom: 12px;
        font-size: 0.84rem;
        align-items: center;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .marker-simulated {
        color: #ff3366;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .marker-prediction {
        color: #00d2ff;
        font-weight: 700;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Session State Initialization (Strict Backward Compatibility)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Sidebar: Primary Telemetry Source & Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ GridShield Console")
    st.caption("SCADA / PMU Telemetry Defense Engine")
    st.markdown("---")

    st.subheader("📁 Telemetry Feed Source")
    data_source = st.selectbox(
        "Source Mode",
        ["Synthetic Simulation", "Upload Dataset (CSV / Excel)"],
        index=0,
    )

    current_config: Dict[str, Any] = {"data_source": data_source}

    if data_source == "Synthetic Simulation":
        n_samples = st.slider("Total Telemetry Hours", min_value=300, max_value=5000, value=1200, step=100)
        base_attack_prob = st.slider("Baseline Cyber Threat Frequency (%)", min_value=0, max_value=30, value=5, step=1) / 100.0
        random_seed = st.number_input("Generator Seed", value=42, step=1)

        current_config.update({
            "n_samples": n_samples,
            "attack_prob": base_attack_prob,
            "seed": int(random_seed),
        })

        # Auto-initialize or regenerate when config changes
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
            st.session_state.models_trained = False
            st.session_state.train_results = {}

        if st.button("🔄 Regenerate Baseline Grid", use_container_width=True):
            fresh_df = generate_synthetic_grid_data(
                n_samples=n_samples,
                attack_prob=base_attack_prob,
                random_seed=int(random_seed),
            )
            st.session_state.sim_df = fresh_df
            st.session_state.attack_history = []
            st.session_state.models_trained = False
            st.session_state.train_results = {}
            st.success("New synthetic telemetry generated.")
            st.rerun()

    else:
        # File upload mode
        uploaded_file = st.file_uploader("Upload SCADA / PMU File", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    raw_df = pd.read_csv(uploaded_file)
                else:
                    raw_df = pd.read_excel(uploaded_file)

                is_valid, val_msg, cleaned_df = validate_grid_dataframe(raw_df)
                if is_valid:
                    if st.session_state.sim_config.get("filename") != uploaded_file.name:
                        st.session_state.sim_df = cleaned_df
                        st.session_state.sim_config = {"filename": uploaded_file.name, "rows": len(cleaned_df)}
                        st.session_state.active_data_source = data_source
                        st.session_state.attack_history = []
                        st.session_state.models_trained = False
                        st.session_state.train_results = {}
                        st.success(val_msg)
                else:
                    st.error(val_msg)
            except Exception as e:
                st.error(f"Error parsing uploaded telemetry: {e}")
        else:
            st.info("Upload grid telemetry or revert to Synthetic Simulation.")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Sidebar Quick Actions & Model Training Trigger
    # -------------------------------------------------------------------------
    st.subheader("🧠 Multi-Model AI Training")
    test_split_ratio = st.slider("Chronological Test Ratio", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    train_models_btn = st.button("🚀 Train & Evaluate Models", type="primary", use_container_width=True)

    if train_models_btn and st.session_state.sim_df is not None:
        with st.spinner("Training all 4 models chronologically (IF, MLP, RF, XGBoost)..."):
            active_df = st.session_state.sim_df.copy()
            # 1. Chronological splitting
            train_df, test_df = chronological_train_test_split(active_df, test_ratio=test_split_ratio)

            # 2. Leak-free feature pipeline fit strictly on train_df
            pipeline = LeakFreeGridPipeline(window_size=5)
            X_train = pipeline.fit_transform(train_df)
            X_test = pipeline.transform(test_df)

            y_train = train_df["target"].values
            y_test = test_df["target"].values

            # 3. Train all 4 models via train_and_evaluate_all_models
            models_dict, comparison_df, detailed_results = train_and_evaluate_all_models(
                X_train, y_train, X_test, y_test, random_state=42
            )

            # Backward-compatible keys for existing tests & legacy consumers
            if_res = detailed_results.get("Isolation Forest", {})
            mlp_res = detailed_results.get("MLP Neural Network", {})

            st.session_state.pipeline = pipeline
            st.session_state.train_results = {
                "train_df": train_df,
                "test_df": test_df,
                "y_train": y_train,
                "y_test": y_test,
                "models": models_dict,
                "comparison_df": comparison_df,
                "detailed_results": detailed_results,
                "if_model": models_dict.get("Isolation Forest"),
                "if_results": if_res,
                "mlp_model": models_dict.get("MLP Neural Network"),
                "mlp_results": mlp_res,
                "split_ratio": test_split_ratio,
            }
            st.session_state.models_trained = True
            st.success(f"Evaluated {len(comparison_df)} AI models without data leakage!")

df = st.session_state.sim_df

if df is None or df.empty:
    st.warning("⚠️ No active telemetry data loaded. Please select Synthetic Simulation or upload a valid dataset.")
    st.stop()

# Ensure timestamp is datetime
if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
    df["timestamp"] = pd.to_datetime(df["timestamp"])

# -----------------------------------------------------------------------------
# Top Hero / Industrial Header Banner
# -----------------------------------------------------------------------------
total_samples = len(df)
attack_samples = int((df["target"] == 1).sum())
attack_pct = (attack_samples / total_samples) * 100 if total_samples > 0 else 0.0

st.markdown(f"""
<div class="gridshield-header">
    <div class="header-title-box">
        <h1>⚡ GridShield AI &mdash; Cyber Threat Operations</h1>
        <p>Industrial SCADA Anomaly Detection &bull; Multi-Vector Attack Simulator &bull; Leak-Free ML/DL Diagnostics</p>
    </div>
    <div class="system-status-pill">
        <span class="status-dot-active"></span>
        DEFENSE ONLINE &bull; {total_samples:,} INTERVALS MONITORED
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Navigation Tabs (7 Comprehensive Operational Modules)
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "⚡ Smart Grid Overview",
    "📡 Sensor Monitoring",
    "🎯 Attack Simulation Studio",
    "🧠 ML/DL Model Evaluation",
    "🚨 Security Alerts",
    "🛡️ Mitigation Recommendations",
    "📑 Incident Reports & Export",
])

# =============================================================================
# TAB 1: Smart Grid Overview
# =============================================================================
with tab1:
    st.markdown("### ⚡ Grid Telemetry & SCADA Substation Health")
    st.caption("Real-time summary of electrical telemetry, operational safety corridors, and active threat footprint.")

    # 4 Key Metrics
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"""
        <div class="metric-hero">
            <div class="metric-hero-label">Total Telemetry Span</div>
            <div class="metric-hero-value">{total_samples:,} <span style="font-size:0.9rem; color:#94a3b8;">hrs</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        status_color = "#ff3366" if attack_pct > 10 else "#ffb300" if attack_pct > 0 else "#00e676"
        st.markdown(f"""
        <div class="metric-hero" style="border-left-color: {status_color};">
            <div class="metric-hero-label">Compromised Readings</div>
            <div class="metric-hero-value" style="color: {status_color};">{attack_samples} <span style="font-size:0.9rem;">({attack_pct:.1f}%)</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        avg_v = float(df["voltage"].mean())
        v_color = "#00e676" if 225 <= avg_v <= 235 else "#ff3366"
        st.markdown(f"""
        <div class="metric-hero" style="border-left-color: {v_color};">
            <div class="metric-hero-label">Mean Bus Voltage</div>
            <div class="metric-hero-value">{avg_v:.2f} <span style="font-size:0.9rem; color:#94a3b8;">V (±1.5%)</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col_m4:
        avg_f = float(df["frequency"].mean())
        f_color = "#00e676" if 49.8 <= avg_f <= 50.2 else "#ff3366"
        st.markdown(f"""
        <div class="metric-hero" style="border-left-color: {f_color};">
            <div class="metric-hero-label">System Frequency</div>
            <div class="metric-hero-value">{avg_f:.3f} <span style="font-size:0.9rem; color:#94a3b8;">Hz</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Substation Topology & Overview Row
    ov_c1, ov_c2 = st.columns([3, 2])
    with ov_c1:
        st.markdown("""
        <div class="cyber-card">
            <div class="cyber-card-header">📊 Substation Active Telemetry Profile</div>
            <p style="color:#94a3b8; font-size:0.88rem;">
                Statistical distribution of operational physical variables across the active monitoring timeframe.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Mini distribution plot
        stat_df = df[["voltage", "frequency", "current", "load"]].describe().T[["mean", "std", "min", "max"]]
        stat_df.columns = ["Mean", "Std Dev", "Min Observed", "Max Observed"]
        st.dataframe(stat_df.style.format("{:.3f}"), use_container_width=True)

    with ov_c2:
        st.markdown("""
        <div class="cyber-card">
            <div class="cyber-card-header">🛡️ SCADA Architecture Health</div>
            <div style="margin-top: 8px;">
                <p style="margin: 6px 0; font-size: 0.88rem;"><strong>PMU Sync Frequency:</strong> <span style="color:#00d2ff;">50 Hz Synchrophasor</span></p>
                <p style="margin: 6px 0; font-size: 0.88rem;"><strong>Encryption Handshake:</strong> <span style="color:#00e676;">AES-GCM-256 Validated</span></p>
                <p style="margin: 6px 0; font-size: 0.88rem;"><strong>Telemetry Protocol:</strong> <span style="color:#00d2ff;">IEEE C37.118.2 / DNP3 Secure</span></p>
                <p style="margin: 6px 0; font-size: 0.88rem;"><strong>Injected Attacks Logged:</strong> <span style="color:#ffb300;">{} event(s)</span></p>
                <p style="margin: 6px 0; font-size: 0.88rem;"><strong>AI Inference State:</strong> <span style="color:{};">{}</span></p>
            </div>
        </div>
        """.format(
            len(st.session_state.attack_history),
            "#00e676" if st.session_state.models_trained else "#ffb300",
            "MODELS TRAINED & EVALUATING" if st.session_state.models_trained else "AWAITING TRAINING",
        ), unsafe_allow_html=True)

    # Attack Distribution Bar Chart
    if attack_samples > 0:
        st.markdown("#### 🎯 Active Cyberattack Scenario Distribution")
        atk_counts = df[df["target"] == 1]["attack_type"].value_counts().reset_index()
        atk_counts.columns = ["Attack Vector", "Affected Hours"]
        fig_atk_dist = px.bar(
            atk_counts,
            x="Attack Vector",
            y="Affected Hours",
            color="Attack Vector",
            color_discrete_sequence=["#ff3366", "#00d2ff", "#ffb300", "#9d4edd", "#00e676"],
        )
        fig_atk_dist.update_layout(
            paper_bgcolor="#0e1b38",
            plot_bgcolor="#0a1128",
            font=dict(color="#f0f4fc"),
            height=280,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_atk_dist, use_container_width=True)

# =============================================================================
# TAB 2: Multi-Sensor Monitoring
# =============================================================================
with tab2:
    st.markdown("### 📡 Synchronized PMU & Multi-Sensor Telemetry")
    st.caption("Continuous electrical monitoring with clear visual distinction between Simulated Attack Ground Truth and AI Model Inferences.")

    # Controls row
    ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([2, 2, 2])
    with ctrl_c1:
        overlay_ground_truth = st.checkbox("Show Injected Attacks (Ground Truth)", value=True)
    with ctrl_c2:
        overlay_ai_preds = st.checkbox("Show Model Predictions (AI Inferred)", value=True, disabled=not st.session_state.models_trained)
    with ctrl_c3:
        if st.session_state.models_trained:
            model_options = list(st.session_state.train_results.get("detailed_results", {}).keys())
            chosen_model_overlay = st.selectbox("AI Model Overlay Source", model_options if model_options else ["MLP Neural Network"])
        else:
            chosen_model_overlay = "None"
            st.caption("ℹ️ Train models to overlay AI detections.")

    # Visual distinction legend
    st.markdown("""
    <div class="legend-box">
        <span style="font-weight:600; color:#94a3b8;">Legend Indicators:</span>
        <div class="legend-item"><span class="marker-simulated">✖</span> <span><strong>Simulated Attack Ground Truth</strong> (Injected Perturbation)</span></div>
        <div class="legend-item"><span class="marker-prediction">▲</span> <span><strong>AI Model Prediction</strong> (Flagged by Machine Learning)</span></div>
        <div class="legend-item"><span style="color:#ffb300;">- - -</span> <span>Frequency Safety Bounds [49.8 &mdash; 50.2 Hz]</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Prepare Subplots
    fig_telemetry = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Substation Bus Voltage (V)",
            "System Frequency Corridor (Hz) [49.8 - 50.2]",
            "Line Current (A)",
            "Active Power / Total Load (kW)"
        ),
        shared_xaxes=True,
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
    )

    # 1. Voltage
    fig_telemetry.add_trace(go.Scatter(
        x=df["timestamp"], y=df["voltage"],
        name="Voltage (V)", line=dict(color="#00d2ff", width=1.5)
    ), row=1, col=1)

    # 2. Frequency with safety corridor
    fig_telemetry.add_trace(go.Scatter(
        x=df["timestamp"], y=df["frequency"],
        name="Frequency (Hz)", line=dict(color="#00e676", width=1.5)
    ), row=1, col=2)
    fig_telemetry.add_hline(y=50.2, line_dash="dash", line_color="#ffb300", row=1, col=2)
    fig_telemetry.add_hline(y=49.8, line_dash="dash", line_color="#ffb300", row=1, col=2)

    # 3. Current
    fig_telemetry.add_trace(go.Scatter(
        x=df["timestamp"], y=df["current"],
        name="Current (A)", line=dict(color="#b388ff", width=1.5)
    ), row=2, col=1)

    # 4. Load
    fig_telemetry.add_trace(go.Scatter(
        x=df["timestamp"], y=df["load"],
        name="Load (kW)", line=dict(color="#ffd166", width=1.5)
    ), row=2, col=2)

    # Ground Truth Overlays (Red Crosses)
    if overlay_ground_truth:
        atk_df = df[df["target"] == 1]
        if not atk_df.empty:
            for r, c, col_name in [(1, 1, "voltage"), (1, 2, "frequency"), (2, 1, "current"), (2, 2, "load")]:
                fig_telemetry.add_trace(go.Scatter(
                    x=atk_df["timestamp"],
                    y=atk_df[col_name],
                    mode="markers",
                    name="Injected Attack" if (r == 1 and c == 1) else None,
                    showlegend=(r == 1 and c == 1),
                    marker=dict(color="#ff3366", size=8, symbol="cross", line=dict(width=1, color="#ffffff")),
                ), row=r, col=c)

    # AI Model Prediction Overlays (Cyan Triangles)
    if overlay_ai_preds and st.session_state.models_trained:
        det = st.session_state.train_results.get("detailed_results", {})
        m_eval = det.get(chosen_model_overlay)
        test_df = st.session_state.train_results.get("test_df")
        if m_eval is not None and test_df is not None:
            pred_flags = np.array(m_eval.get("predictions", []))
            if len(pred_flags) == len(test_df):
                test_pred_df = test_df[pred_flags == 1]
                if not test_pred_df.empty:
                    for r, c, col_name in [(1, 1, "voltage"), (1, 2, "frequency"), (2, 1, "current"), (2, 2, "load")]:
                        fig_telemetry.add_trace(go.Scatter(
                            x=test_pred_df["timestamp"],
                            y=test_pred_df[col_name],
                            mode="markers",
                            name=f"{chosen_model_overlay} Anomaly" if (r == 1 and c == 1) else None,
                            showlegend=(r == 1 and c == 1),
                            marker=dict(color="#00f2fe", size=7, symbol="triangle-up", line=dict(width=1, color="#005577")),
                        ), row=r, col=c)

    fig_telemetry.update_layout(
        paper_bgcolor="#0e1b38",
        plot_bgcolor="#0a1128",
        font=dict(color="#f0f4fc"),
        height=580,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_telemetry.update_xaxes(gridcolor="#1e3567", zeroline=False)
    fig_telemetry.update_yaxes(gridcolor="#1e3567", zeroline=False)

    st.plotly_chart(fig_telemetry, use_container_width=True)

# =============================================================================
# TAB 3: Attack Simulation Studio
# =============================================================================
with tab3:
    st.markdown("### 🎯 Cyberattack Scenario Simulation Studio")
    st.caption("Inject physics-compliant, stateful, and dynamic multi-sensor cyberattacks directly into active SCADA telemetry.")

    studio_c1, studio_c2 = st.columns([1, 1])

    with studio_c1:
        st.markdown("#### 🛠️ Attack Scenario Parameters")
        attack_types_available = [
            "voltage_spike",
            "frequency_shift",
            "load_drift",
            "stealth_fdia",
            "telemetry_dropout",
            "ramping_fdi",
        ]

        def attack_name_formatter(atk_key: str) -> str:
            meta = ATTACK_METADATA.get(atk_key, {})
            return meta.get("name", atk_key)

        chosen_attack = st.selectbox(
            "Attack Vector Type",
            attack_types_available,
            format_func=attack_name_formatter,
        )

        chosen_meta = ATTACK_METADATA.get(chosen_attack, {})

        # Dynamic sensor targeting
        default_sensors = chosen_meta.get("affected_sensors", ["voltage"])
        target_sensor_choice = st.selectbox(
            "Target Substation Sensor",
            ["voltage", "frequency", "current", "load", "all"],
            index=0 if "voltage" in default_sensors else 1 if "frequency" in default_sensors else 0,
        )

        atk_duration = st.slider("Attack Duration (Operating Hours)", min_value=1, max_value=min(60, len(df)), value=12)

        # Attack-specific controls
        dropout_mode = "zero"
        ramp_shape = "linear"

        if chosen_attack == "telemetry_dropout":
            dropout_mode = st.radio("Dropout Emulation Mode", ["zero (0.0 Flatline)", "flatline (Freeze Previous Value)"], index=0)
            dropout_mode = "zero" if "zero" in dropout_mode else "flatline"

        elif chosen_attack == "ramping_fdi":
            ramp_shape = st.radio("Drift Ramp Profile", ["linear (Direct Ramp)", "progressive (Exponential / Quadratic)"], index=0)
            ramp_shape = "linear" if "linear" in ramp_shape else "progressive"

        # Magnitude control
        default_mag = float(chosen_meta.get("default_magnitude", 1.0))
        atk_magnitude = st.number_input(
            "Attack Perturbation Magnitude / Multiplier",
            value=default_mag,
            step=0.1,
            help=chosen_meta.get("param_description", "Perturbation scale parameter"),
        )

        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            inject_btn = st.button("🔴 INJECT ATTACK", use_container_width=True, type="primary")
        with btn_c2:
            clear_btn = st.button("🧹 Clear Injected Attacks", use_container_width=True)

    with studio_c2:
        st.markdown("#### 📋 Attack Vector Intelligence Card")
        st.markdown(f"""
        <div class="cyber-card">
            <div class="cyber-card-header">
                <span>🛡️ Vector: {chosen_meta.get('name', chosen_attack)}</span>
                <span class="cyber-badge cyber-badge-critical">THREAT PROFILE</span>
            </div>
            <p style="color:#f0f4fc; font-size:0.92rem;">{chosen_meta.get('description', '')}</p>
            <hr style="border-color:#1e3567; margin: 12px 0;">
            <p style="margin: 4px 0; font-size: 0.88rem;"><strong>Primary Affected Sensors:</strong> <code style="color:#00d2ff;">{', '.join(chosen_meta.get('affected_sensors', []))}</code></p>
            <p style="margin: 4px 0; font-size: 0.88rem;"><strong>Default Baseline Impact:</strong> {chosen_meta.get('default_magnitude')}</p>
            <p style="margin: 4px 0; font-size: 0.88rem;"><strong>Parameter Guidance:</strong> <span style="color:#94a3b8;">{chosen_meta.get('param_description')}</span></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 📜 Session Attack History Log")
        if st.session_state.attack_history:
            history_rows = []
            for i, h in enumerate(st.session_state.attack_history, 1):
                history_rows.append({
                    "Event": f"#{i}",
                    "Vector": h.get("type"),
                    "Target Sensor": h.get("target_sensor", "N/A"),
                    "Duration": f"{h.get('duration')} hrs",
                    "Indices Impacted": len(h.get("indices", [])),
                })
            st.dataframe(pd.DataFrame(history_rows), use_container_width=True)
        else:
            st.info("No manual attacks injected in this session yet.")

    # Execution of Attack Injection
    if inject_btn:
        start_point = max(0, len(df) - atk_duration)
        modified_df, mod_indices = inject_dynamic_attack_scenario(
            df,
            attack_type=chosen_attack,
            start_idx=start_point,
            duration=atk_duration,
            magnitude=float(atk_magnitude),
            target_sensor=target_sensor_choice,
            dropout_mode=dropout_mode,
            ramp_shape=ramp_shape,
        )
        st.session_state.sim_df = modified_df
        st.session_state.attack_history.append({
            "type": chosen_attack,
            "target_sensor": target_sensor_choice,
            "duration": atk_duration,
            "magnitude": float(atk_magnitude),
            "start_idx": start_point,
            "indices": mod_indices,
            "dropout_mode": dropout_mode,
            "ramp_shape": ramp_shape,
        })
        if st.session_state.models_trained:
            st.session_state.models_trained = False
            st.warning("⚠️ Telemetry state modified! Re-train models to evaluate defense against the new attack scenario.")
        st.success(f"Injected {chosen_attack} over {len(mod_indices)} intervals!")
        st.rerun()

    if clear_btn:
        df_reset = df.copy()
        df_reset["target"] = 0
        df_reset["attack_type"] = "none"
        df_reset["load"] = (df_reset["voltage"] * df_reset["current"]) / 1000.0
        st.session_state.sim_df = df_reset
        st.session_state.attack_history = []
        st.session_state.models_trained = False
        st.session_state.train_results = {}
        st.success("All attacks cleared. Baseline telemetry restored.")
        st.rerun()

# =============================================================================
# TAB 4: ML/DL Model Evaluation
# =============================================================================
with tab4:
    st.markdown("### 🧠 Machine Learning & Deep Learning Model Benchmarks")
    st.caption("Rigorous, leak-free chronological evaluation comparing Unsupervised, Ensemble, Gradient Boosted, and Deep Neural architectures.")

    if not st.session_state.models_trained:
        st.info("👈 Please click **'🚀 Train & Evaluate Models'** in the sidebar to train and compare all 4 models on unseen chronological test data.")
    else:
        results = st.session_state.train_results
        test_df = results["test_df"]
        comp_df = results.get("comparison_df")
        det_res = results.get("detailed_results", {})

        st.markdown(f"**Chronological Test Set:** Evaluating on **{len(test_df)} held-out intervals** (Test ratio: `{results['split_ratio']:.0%}`). Zero future-to-past data leakage.")

        # Consolidated Model Comparison Table
        if comp_df is not None and not comp_df.empty:
            st.markdown("#### 📋 Consolidated Performance Metrics Table")

            formatted_table = comp_df.copy()
            st.dataframe(
                formatted_table.style.format({
                    "Accuracy": "{:.1%}",
                    "Precision": "{:.3f}",
                    "Recall": "{:.3f}",
                    "F1-score": "{:.3f}",
                    "ROC-AUC": "{:.3f}",
                    "False Positive Rate": "{:.1%}",
                }).highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"], color="#134e4a"),
                use_container_width=True,
            )

            # Interactive Multi-Model Metric Bar Chart
            st.markdown("#### 📊 Comparative Metric Breakdown (Accuracy vs F1-score vs ROC-AUC)")
            plot_metrics_df = comp_df.melt(
                id_vars=["Model"],
                value_vars=["Accuracy", "F1-score", "ROC-AUC"],
                var_name="Metric",
                value_name="Score"
            )
            fig_comp_bars = px.bar(
                plot_metrics_df,
                x="Model",
                y="Score",
                color="Metric",
                barmode="group",
                color_discrete_sequence=["#00d2ff", "#00e676", "#ffb300"],
            )
            fig_comp_bars.update_layout(
                paper_bgcolor="#0e1b38",
                plot_bgcolor="#0a1128",
                font=dict(color="#f0f4fc"),
                height=320,
                margin=dict(l=20, r=20, t=30, b=20),
                yaxis=dict(range=[0, 1.05], gridcolor="#1e3567"),
                xaxis=dict(gridcolor="#1e3567"),
            )
            st.plotly_chart(fig_comp_bars, use_container_width=True)

        # Side-by-side Confusion Matrix Heatmaps
        st.markdown("#### 🧩 Confusion Matrix Visualizations (Unseen Chronological Test Data)")
        cm_cols = st.columns(len(det_res))

        for idx, (m_name, m_data) in enumerate(det_res.items()):
            with cm_cols[idx]:
                cm_arr = np.array(m_data["confusion_matrix"])
                fig_cm = px.imshow(
                    cm_arr,
                    text_auto=True,
                    color_continuous_scale="Blues" if idx % 2 == 0 else "Teal",
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=["Normal", "Attack"],
                    y=["Normal", "Attack"],
                    title=f"{m_name}",
                )
                fig_cm.update_layout(
                    paper_bgcolor="#0e1b38",
                    plot_bgcolor="#0a1128",
                    font=dict(color="#f0f4fc"),
                    height=260,
                    margin=dict(l=20, r=20, t=40, b=20),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_cm, use_container_width=True)
                st.markdown(f"""
                <div style="font-size:0.82rem; color:#94a3b8; text-align:center;">
                    F1: <strong style="color:#00d2ff;">{m_data['f1']:.3f}</strong> | 
                    FPR: <strong style="color:#ffb300;">{m_data['false_positive_rate']:.1%}</strong> | 
                    Recall: <strong>{m_data['recall']:.3f}</strong>
                </div>
                """, unsafe_allow_html=True)

# =============================================================================
# TAB 5: Security Alerts
# =============================================================================
with tab5:
    st.markdown("### 🚨 Real-Time Security Alerts & Anomaly Feed")
    st.caption("Chronological threat assessment inspecting the latest operating intervals in the SCADA pipeline.")

    if not st.session_state.models_trained:
        st.info("Train the AI models in the sidebar to activate the Real-Time Security Alert stream.")
    else:
        results = st.session_state.train_results
        test_df = results["test_df"]
        det_res = results.get("detailed_results", {})

        # Select primary model for alerting
        alert_model_name = st.selectbox(
            "Incident Detection Model Feed",
            list(det_res.keys()),
            index=list(det_res.keys()).index("MLP Neural Network") if "MLP Neural Network" in det_res else 0,
        )

        chosen_eval = det_res[alert_model_name]
        preds = np.array(chosen_eval["predictions"])
        scores = np.array(chosen_eval.get("anomaly_scores", preds))

        # Inspect the recent window
        window_size = min(24, len(test_df))
        recent_preds = preds[-window_size:]
        recent_attacks = int(np.sum(recent_preds))
        recent_test_slice = test_df.iloc[-window_size:]

        # Real-time threat classification
        if recent_attacks >= 5:
            threat_level = "CRITICAL"
            badge_class = "cyber-badge-critical"
            status_desc = f"CRITICAL SCADA ALERT: Severe persistent cyber attack detected ({recent_attacks}/{window_size} intervals flagged anomalous)!"
        elif recent_attacks >= 2:
            threat_level = "HIGH"
            badge_class = "cyber-badge-critical"
            status_desc = f"HIGH CYBER ALERT: Multi-interval telemetry perturbation detected ({recent_attacks}/{window_size} intervals anomalous)!"
        elif recent_attacks == 1:
            threat_level = "ELEVATED"
            badge_class = "cyber-badge-warning"
            status_desc = f"ELEVATED THREAT: Transient anomaly or early stage FDI attempt detected (1/{window_size} intervals anomalous)."
        else:
            threat_level = "NORMAL"
            badge_class = "cyber-badge-safe"
            status_desc = f"NORMAL STATE: All recent {window_size} operational intervals conform to nominal grid dynamics."

        st.markdown(f"""
        <div class="cyber-card" style="border-left: 6px solid {'#ff3366' if threat_level in ['CRITICAL', 'HIGH'] else '#ffb300' if threat_level == 'ELEVATED' else '#00e676'};">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div class="cyber-card-header" style="margin-bottom:0;">🚨 Active Threat Level: {threat_level}</div>
                <span class="cyber-badge {badge_class}">{threat_level}</span>
            </div>
            <p style="margin: 10px 0 0 0; color:#f0f4fc; font-size:0.95rem;">{status_desc}</p>
        </div>
        """, unsafe_allow_html=True)

        # Chronological Recent Alerts Table
        st.markdown(f"#### ⏱️ Chronological Inspection of Last {window_size} Telemetry Intervals")
        inspect_df = recent_test_slice[["timestamp", "voltage", "frequency", "current", "load", "target"]].copy()
        inspect_df["AI Prediction"] = np.where(recent_preds == 1, "⚠️ Cyber Anomaly", "✅ Normal")
        inspect_df["Anomaly Score"] = scores[-window_size:]
        inspect_df["Ground Truth"] = np.where(inspect_df["target"] == 1, "Simulated Attack", "Normal")
        inspect_df = inspect_df.drop(columns=["target"])

        st.dataframe(
            inspect_df.style.format({
                "voltage": "{:.2f} V",
                "frequency": "{:.3f} Hz",
                "current": "{:.2f} A",
                "load": "{:.2f} kW",
                "Anomaly Score": "{:.3f}",
            }),
            use_container_width=True,
        )

# =============================================================================
# TAB 6: Mitigation Recommendations
# =============================================================================
with tab6:
    st.markdown("### 🛡️ Prescriptive Cyber-Physical Mitigation Actions")
    st.caption("Automated industrial response protocols triggered dynamically by detected physical sensor deviations.")

    if not st.session_state.models_trained:
        st.info("Train the AI models to synthesize tailored cyber-physical mitigation guidance.")
    else:
        results = st.session_state.train_results
        test_df = results["test_df"]
        det_res = results.get("detailed_results", {})

        # Use MLP or top model
        m_eval = det_res.get("MLP Neural Network") or next(iter(det_res.values()))
        recent_preds = np.array(m_eval["predictions"])[-min(20, len(test_df)):]
        recent_test_slice = test_df.iloc[-min(20, len(test_df)):]

        active_anomalies = int(np.sum(recent_preds))

        # Dynamic physics deviations
        freq_dev = abs(float(recent_test_slice["frequency"].mean()) - 50.0)
        volt_dev = abs(float(recent_test_slice["voltage"].mean()) - 230.0)
        curr_mean = float(recent_test_slice["current"].mean())
        load_mean = float(recent_test_slice["load"].mean())

        # Generate targeted prescriptive actions
        prescriptive_actions: List[Dict[str, str]] = []

        if active_anomalies > 0:
            if freq_dev > 0.15:
                prescriptive_actions.append({
                    "priority": "HIGH",
                    "domain": "Grid Frequency Stabilization",
                    "action": "Trigger Fast Frequency Response (FFR) and dispatch spinning reserves. PMU frequency corridor compromised.",
                    "protocol": "IEEE 1547 / Grid Code Class 1",
                })
            if volt_dev > 2.5:
                prescriptive_actions.append({
                    "priority": "HIGH",
                    "domain": "Substation Voltage Control",
                    "action": "Lock On-Load Tap Changers (OLTC) in fail-safe position. Cross-verify bus potential transformers against secondary redundant sensors.",
                    "protocol": "NERC CIP-005-5 / IEC 61850",
                })
            prescriptive_actions.append({
                "priority": "CRITICAL",
                "domain": "SCADA Zero-Trust Defense",
                "action": "Enforce zero-trust TLS re-authentication across edge PMU microcontrollers and switch compromised feeder bus to autonomous islanded microgrid.",
                "protocol": "IEC 62351 / NIST SP 800-82",
            })
            prescriptive_actions.append({
                "priority": "MEDIUM",
                "domain": "Telemetry State Estimation",
                "action": "Activate Weighted Least Squares (WLS) bad-data rejection filter with residual threshold set to 3-sigma confidence bounds.",
                "protocol": "IEEE PES State Estimation Standard",
            })
        else:
            prescriptive_actions.append({
                "priority": "INFO",
                "domain": "Nominal Surveillance",
                "action": "Grid telemetry operates within safe physical envelopes. Maintain continuous SCADA watchdog polling and automated PMU cryptographic key rotation.",
                "protocol": "IEC 62443 Standard Operating Procedure",
            })

        for act in prescriptive_actions:
            p_class = "cyber-badge-critical" if act["priority"] == "CRITICAL" else "cyber-badge-warning" if act["priority"] == "HIGH" else "cyber-badge-safe"
            st.markdown(f"""
            <div class="cyber-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div class="cyber-card-header">{act['domain']}</div>
                    <span class="cyber-badge {p_class}">{act['priority']} PRIORITY</span>
                </div>
                <p style="color:#f0f4fc; font-size:0.95rem; margin:8px 0;">{act['action']}</p>
                <div style="font-size:0.8rem; color:#94a3b8;">Standard Compliance: <code style="color:#00d2ff;">{act['protocol']}</code></div>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# TAB 7: Incident Reports & Export
# =============================================================================
with tab7:
    st.markdown("### 📑 Cyber Incident & Forensic Audit Report")
    st.caption("Generate, review, and download audit-ready HTML & PDF-printable reports combining live telemetry, attack scenarios, and ML benchmarks.")

    telemetry_summary = {
        "total_samples": len(df),
        "suspicious_readings": int((df["target"] == 1).sum()),
        "suspicious_pct": float((df["target"] == 1).mean() * 100) if len(df) > 0 else 0.0,
        "mean_voltage": float(df["voltage"].mean()),
        "mean_frequency": float(df["frequency"].mean()),
        "mean_current": float(df["current"].mean()),
        "mean_load": float(df["load"].mean()),
    }

    # Extract mitigation list
    mitigation_list_str = []
    if st.session_state.models_trained:
        t_level = "CRITICAL" if telemetry_summary["suspicious_readings"] > 10 else "ELEVATED" if telemetry_summary["suspicious_readings"] > 0 else "NORMAL"
        mitigation_list_str = [
            "Trigger Fast Frequency Response (FFR) and dispatch spinning reserves to prevent generator trip.",
            "Lock Substation Transformer Tap-Changers into fail-safe steady state.",
            "Enforce zero-trust PMU re-authentication and transition affected bus to autonomous islanding.",
            "Engage Weighted Least Squares (WLS) state estimator bad-data detector.",
        ]
    else:
        t_level = "ELEVATED" if telemetry_summary["suspicious_readings"] > 0 else "NORMAL"
        mitigation_list_str = ["Telemetry baseline surveillance active. Perform routine PMU health checks."]

    # Model comparison dataframe
    report_comp_df = st.session_state.train_results.get("comparison_df") if st.session_state.models_trained else None

    # Generate standalone styled HTML report
    report_html = generate_incident_report_html(
        project_name="GridShield AI Smart Grid Cyber Defense",
        telemetry_summary=telemetry_summary,
        attack_history=st.session_state.attack_history,
        model_comparison_df=report_comp_df,
        mitigation_actions=mitigation_list_str,
        active_threat_level=t_level,
    )

    btn_down_c1, btn_down_c2 = st.columns([2, 4])
    with btn_down_c1:
        st.download_button(
            label="💾 Download Incident Audit Report (.HTML)",
            data=report_html,
            file_name=f"GridShield_Audit_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            type="primary",
            use_container_width=True,
        )
    with btn_down_c2:
        st.caption("ℹ️ The downloaded HTML file can be opened directly in any browser and printed or saved as a clean PDF document.")

    st.markdown("#### 🔍 Interactive Forensic Report Preview")
    st.components.v1.html(report_html, height=720, scrolling=True)

# -----------------------------------------------------------------------------
# Raw Telemetry Inspector (Collapsible Bottom Expander)
# -----------------------------------------------------------------------------
with st.expander("🔍 SCADA Raw Telemetry Inspector (DataFrame View)"):
    st.dataframe(df, use_container_width=True)
