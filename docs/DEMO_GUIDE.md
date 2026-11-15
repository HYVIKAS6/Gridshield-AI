# GridShield AI — College Lecturer Demonstration Guide 🎓

**Document Version:** 2.0  
**Target Audience:** Academic Examiners, College Lecturers, and Cybersecurity Evaluators  
**Execution Time:** ~10-15 minutes  

---

## 1. Demonstration Overview

This guide provides a step-by-step walkthrough to showcase **GridShield AI** in an academic evaluation or classroom demonstration. The workflow proves that the application executes real machine learning models, domain physics equations, and power-flow simulations with zero hardcoding.

---

## 2. Step-by-Step Live Demonstration Sequence

### Step 1: Launch Application
Open PowerShell or Terminal and execute:

```powershell
cd "C:\Users\koush\Desktop\Gridshield-AI-main (1)\Gridshield-AI-main"
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501` (or `8502`).

---

### Step 2: Telemetry Ingestion (Sidebar)
1. Point out the sidebar **Telemetry Feed Source** selector.
2. Select **Offline Academic Sample Dataset** (demonstrates zero-internet resilience).
3. Point out the loaded 1,200 telemetry records and 7 operational columns.

---

### Step 3: Inspect Telemetry & Corridors (Tabs 1 & 2)
1. Navigate to **⚡ Command Overview** (Tab 1): Highlight the Mean Bus Voltage ($230.1\text{ V}$) and Frequency ($50.0\text{ Hz}$).
2. Navigate to **📡 Telemetry & Sensors** (Tab 2): Point out the real-time Plotly interactive curves for Voltage, Frequency, Current, and Active Power with statutory bounds.

---

### Step 4: Inject Cyberattack Scenario (Tab 3)
1. Click **🎯 Attack Simulation Studio** (Tab 3).
2. Choose **Stealth Coordinated FDIA** or **Frequency Sag / Shift**.
3. Set Attack Duration to **12 hours** and Magnitude to **1.35x**.
4. Click **🔥 Inject Attack Scenario into Live Telemetry**.
5. Observe the red compromise markers appearing on the telemetry waveforms.

---

### Step 5: Multi-Model AI Training (Sidebar & Tab 4)
1. In the sidebar under **Multi-Model AI Training**, set Chronological Test Ratio to **0.25** ($25\%$).
2. Click **🚀 Train & Evaluate Models**.
3. Navigate to **🧠 Multi-Model AI Intelligence** (Tab 4):
   - Review the Comparative Performance Table showing **Isolation Forest**, **Random Forest**, **XGBoost**, **MLP**, and **PyTorch LSTM**.
   - Show the side-by-side Confusion Matrix heatmaps proving that real TP, FP, TN, and FN are calculated on unseen test data.

---

### Step 6: Explainable AI & Feature Attribution (Tab 5)
1. Navigate to **💡 Explainable AI (XAI)** (Tab 5).
2. Show the **Global Feature Importance** bar chart (Random Forest & XGBoost).
3. Select an attack observation in the test slider to show the **SHAP attribution waterfall** decoding why that specific telemetry instance was flagged.
4. Click **📦 Export Models to Registry** to demonstrate model persistence into `models/saved/`.

---

### Step 7: Cyber-Physical Law Tracking (Tab 6)
1. Navigate to **🔬 Cyber-Physical Integrity** (Tab 6).
2. Show the **$P = V \cdot I$ Power Law Tracking** graph. Show how physical discrepancy triggers when active load does not match theoretical power.
3. Show the **Rate of Change of Frequency (RoCoF)** tracking against the $0.50\text{ Hz/s}$ emergency trip limit.

---

### Step 8: Multi-Bus IEEE Power-Flow Simulation (Tab 7)
1. Navigate to **🌐 IEEE Grid Topology & Power Flow** (Tab 7).
2. Select **IEEE 14-Bus System**, set Target Bus to **Bus 3**, and select **Load FDI** attack ($2.0\times$).
3. Click **⚡ Execute AC Newton-Raphson Power Flow**.
4. Show the solved bus voltage profile ($V_m\text{ p.u.}$) and transmission line thermal loading percentages calculated dynamically via Pandapower.

---

### Step 9: SOC Incident Alerts & Mitigation Directives (Tabs 8 & 9)
1. Navigate to **🚨 Incident Alerts & SOC** (Tab 8): Show the rolling threat level (**CRITICAL** / **HIGH**).
2. Navigate to **🛡️ Mitigation Directives** (Tab 9): Highlight prescriptive response workflows mapped to **NERC CIP-005** and **IEC 61850**.

---

### Step 10: Export Forensic Audit Report (Tab 10)
1. Navigate to **📑 Forensics & Export Reports** (Tab 10).
2. Click **💾 Download Incident Audit Report (.HTML)**.
3. Open the downloaded HTML file in the browser to demonstrate the complete, audit-ready incident forensic brief with printable CSS.
