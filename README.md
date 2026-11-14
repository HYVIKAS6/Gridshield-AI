# GridShield AI ⚡
### Industrial Intelligent Cyberattack Detection and Simulation Engine for Smart Grids

**GridShield AI** is an industrial-grade simulation and cyber defense platform designed for power systems operators, cybersecurity researchers, and AI engineers. It simulates, detects, evaluates, and advises mitigation strategies against sophisticated cyber-physical threats targeting critical smart grid infrastructure:
- **False Data Injection Attacks (FDIA)** (covert and uncoordinated)
- **Frequency Sags / Swells**
- **Voltage Surges / Sags**
- **Telemetry Dropout & Communication Outages**
- **Ramping / Slow-Drift FDI** (stealth attacks evading threshold relays)

---

## 🚀 Key Upgrades & Capabilities

### 1. Multi-Model Leak-Free AI Engine
- **Unsupervised Anomaly Detection:**
  - **Isolation Forest:** Auto-calibrated contamination hyperparameter with continuous decision score distributions.
- **Supervised ML & Neural Baselines:**
  - **Random Forest Classifier:** Multi-tree bagging model capturing non-linear sensor correlations.
  - **XGBoost Classifier:** Extreme Gradient Boosted Trees for high-precision cyberattack discrimination.
  - **Multi-Layer Perceptron (MLP):** Multi-layer neural network with adaptive early stopping.
- **Chronological Time-Series Validation:**
  - Temporal forward-looking splits without lookahead data leakage.
  - Rolling window averages and lag features fitted strictly on training distributions.
- **Unified Empirical Evaluation Metrics:**
  - Evaluated on actual unseen test predictions (Accuracy, Precision, Recall, F1-Score, ROC-AUC, False Positive Rate, and Confusion Matrix breakdown).

### 2. Comprehensive Cyberattack Simulator
- **Voltage Spike / Surge (+V):** Transient overvoltage excursions.
- **Frequency Sag / Shift (±f):** Violation of grid frequency stability corridors ($[49.8 - 50.2]\text{ Hz}$).
- **Load Inconsistency / Drift (*P):** Sensor desynchronization violating physical Ohm's and active power relationships ($P = V \cdot I$).
- **Stealth Coordinated FDIA:** Simultaneous algebraic perturbation across voltage, current, and active power.
- **Telemetry Dropout / Loss:** Simulates packet drop, RTU communication outages, or dead sensors (zero-out or flatline freeze).
- **Ramping / Slow-Drift FDI:** Progressive linear or exponential parameter tampering designed to evade instantaneous trip thresholds.
- **Visual Distinction:** Distinct UI markers and legends differentiating injected ground-truth attacks (red crosses) from AI model predictions (cyan triangles).

### 3. Industrial Cybersecurity Dashboard
A redesigned dark navy (`#0a1128`) and electric cyan interface structured into 7 dedicated operational views:
1. ⚡ **Smart Grid Overview:** High-level metrics, sensor health indicators, and baseline statistics.
2. 📡 **Sensor Monitoring:** Real-time multi-sensor Plotly telemetry for Voltage, Frequency corridor, Current, and Active Power.
3. 🎯 **Attack Simulation Studio:** Live scenario configuration, duration and magnitude controls, and attack parameter guidance cards.
4. 🧠 **ML/DL Model Evaluation:** Comparative performance tables, multi-model F1/Accuracy bar charts, and side-by-side Confusion Matrix heatmaps.
5. 🚨 **Security Alerts:** Chronological inspection of recent test windows with real-time threat levels (CRITICAL, HIGH, ELEVATED, NORMAL).
6. 🛡️ **Mitigation Recommendations:** Actionable cyber-physical response protocols adhering to standards (IEEE 1547, NERC CIP, IEC 61850).
7. 📑 **Incident Reports & Export:** One-click exportable HTML audit reports (printable/savable to PDF).

---

## 📂 Project Architecture

```text
.
├── app.py                  # Redesigned industrial Streamlit dashboard & stateful navigation
├── data_preprocessing.py   # Physics telemetry generation, 6 attack vectors, leak-free pipeline
├── model_training.py       # Isolation Forest, Random Forest, XGBoost, MLP & benchmark helpers
├── report_generator.py     # HTML & PDF-printable incident report generation engine
├── test_pipeline.py        # Automated test suite for data pipeline, attacks, ML models, and reports
├── test_streamlit_app.py   # Streamlit AppTest suite verifying session state and UI workflows
├── requirements.txt        # Pinned dependencies (including xgboost and pytest)
├── pytest.ini              # Pytest runner configuration
├── backup_pre_upgrade/     # Complete untouched backup of pre-upgrade repository files
└── README.md               # Architecture documentation and execution guide
```

---

## 🛠️ Windows Execution Instructions

### 1. Activate the Virtual Environment
Open PowerShell in the project directory:
```powershell
.\.venv\Scripts\activate
```

### 2. Verify / Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Run Automated Pytest Suite
```powershell
pytest -v
```
*(All 12 automated unit and integration tests will execute and pass)*.

### 4. Launch the Web Application
```powershell
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## 📊 Empirical Model Benchmark Results
Evaluated strictly on held-out chronological test data ($N=300$ unseen intervals from $N=1500$ observations):

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | False Positive Rate | True Positives | False Positives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier** | **99.0%** | **1.000** | **0.893** | **0.943** | **1.000** | **0.0%** | 25 | 0 |
| **Random Forest** | 98.7% | 1.000 | 0.857 | 0.923 | 1.000 | 0.0% | 24 | 0 |
| **Isolation Forest (Unsupervised)** | 91.0% | 0.516 | 0.571 | 0.542 | 0.912 | 5.5% | 16 | 15 |
| **MLP Neural Network** | 92.0% | 1.000 | 0.143 | 0.250 | 0.618 | 0.0% | 4 | 0 |

---

## ⚠️ Known Limitations & Scope Boundaries
- **Telemetry Boundaries:** Generates physics-consistent single-bus aggregated telemetry ($V, I, P = V \cdot I, f$). It does not perform multi-bus IEEE power flow solves (e.g., IEEE 14/33-bus AC power flow), which is a future research direction.
- **Simulation Disclaimer:** This software is a cyber-physical simulator for testing detection models and training operators; simulated monitoring is not connected to physical grid hardware or live SCADA networks.
