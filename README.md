# GridShield AI ⚡
### Industrial Cyber-Physical Smart Grid Intrusion Detection & Anomaly Monitoring Platform

**GridShield AI** is an advanced cyber-physical simulation, intrusion detection, and power systems defense platform designed for utility operators, cybersecurity researchers, and AI engineers. It combines physics-informed electrical models with deep learning and machine learning algorithms to detect, isolate, and advise mitigations against sophisticated cyber-physical attacks targeting critical infrastructure:

- **False Data Injection Attacks (FDIA)** (stealth coordinated and uncoordinated)
- **Voltage Surges & Sags**
- **Frequency Deviations & RoCoF Spikes**
- **Load Inconsistencies & Telemetry Dropouts**
- **Ramping / Slow-Drift FDI** (stealth parameter tampering evading threshold relays)
- **Multi-Bus Substation & Line Trip Scenarios** (IEEE 9-Bus & IEEE 14-Bus)

---

## 🚀 Core Capabilities & Architecture

```
                                      +------------------------------------+
                                      |   SCADA / PMU Telemetry Sources    |
                                      | (Synthetic, IEEE Bus, Research CSV)|
                                      +-----------------+------------------+
                                                        |
                                                        v
                                      +------------------------------------+
                                      | Data Engineering & Schema Pipeline |
                                      |  (Leak-Free Scaling, Adapters)     |
                                      +--------+------------------+--------+
                                               |                  |
                       +-----------------------+                  +-----------------------+
                       v                                                                  v
+---------------------------------------------+                    +---------------------------------------------+
|        Cyber-Physical Diagnostic Engine     |                    |       Multi-Model AI/DL Anomaly Engine      |
|  - Active Power Law (P = V * I * pf / 1000) |                    |  - Isolation Forest (Unsupervised)          |
|  - Voltage Stability Corridors (0.90 - 1.10)|                    |  - Random Forest Classifier                 |
|  - Frequency Corridors & RoCoF (|df/dt|)    |                    |  - XGBoost Gradient Boosted Trees           |
|  - Sensor Flatline & Telemetry Loss Checks  |                    |  - Deep MLP Neural Network                  |
+----------------------+----------------------+                    |  - Temporal PyTorch LSTM Sequence Model     |
                       |                                           +----------------------+----------------------+
                       |                                                                  |
                       +-----------------------+                  +-----------------------+
                                               |                  |
                                               v                  v
                                      +------------------------------------+
                                      |   IEEE Grid Power Flow Simulation  |
                                      | (Pandapower AC Newton-Raphson NR)  |
                                      +-----------------+------------------+
                                                        |
                                                        v
                                      +------------------------------------+
                                      | Industrial Streamlit Command Center|
                                      |  - 9 Structured Monitoring Modules |
                                      |  - Real-Time Security Alerts & SOC |
                                      |  - Forensic Audit Report Generator |
                                      +------------------------------------+
```

### 1. Multi-Model AI & Deep Learning Intelligence
- **Isolation Forest:** Unsupervised outlier detection with dynamic contamination calibration.
- **Random Forest:** Non-linear decision trees capturing sensor interactions.
- **XGBoost:** Gradient boosted decision trees for robust boundary classification.
- **MLP Classifier:** Multi-layer feedforward network with adaptive learning rate and early stopping.
- **PyTorch LSTM Neural Network:** Sequential temporal deep learning model tracking sliding time windows without data leakage.

### 2. Physics-Informed Cyber-Physical Validation
- **Power Conservation:** Validates single-phase & three-phase active power $P = \frac{V \cdot I \cdot \cos\phi}{1000}$ with residual tracking.
- **Voltage Stability:** Statutory operating envelopes ($[0.90 - 1.10]\text{ p.u.}$).
- **Rate of Change of Frequency (RoCoF):** Instantaneous $|df/dt|$ tracking with trip alarms above $0.50\text{ Hz/s}$.
- **Telemetry Integrity:** Detects frozen sensors, packet drops, and zero-out outages.

### 3. Multi-Bus Smart Grid Simulation (Pandapower)
- Supports standard **IEEE 14-Bus**, **IEEE 9-Bus**, and **CIGRE Medium Voltage** networks.
- Full **AC Newton-Raphson power-flow solver** calculating bus voltage magnitudes ($V_m\text{ p.u.}$) and transmission line thermal loading percentages.
- Simulates targeted bus attacks including **Load FDI**, **Voltage Setpoint Tampering**, and **Feeder Line Outages**.

### 4. Authentic Research Dataset Adapters
- Ingestion adapters for:
  - **SmartGrid CyberPhysical Attack Dataset** (DOI: 10.21227/symr-bz19)
  - **Data4Cyber Dataset** (Zenodo: 20004612)
  - **Power System Intrusion Dataset** (PMU / Relay telemetry)
- Automatic column schema detection, duplicate elimination, and data quality reporting.

---

## 📊 AI Benchmark Performance (Verified Chronological Test Set)

| Model Name | Type | Accuracy | Precision | Recall | F1-Score | ROC-AUC | False Positive Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | Supervised Bagging | **93.3%** | **0.800** | **0.800** | **0.800** | **0.955** | **4.2%** |
| **XGBoost** | Gradient Boosted Trees | **93.3%** | **0.800** | **0.800** | **0.800** | **0.882** | **4.2%** |
| **Isolation Forest** | Unsupervised Trees | **83.3%** | **0.571** | **0.800** | **0.667** | **0.806** | **12.5%** |
| **PyTorch LSTM** | Deep Sequential Recurrent | **80.0%** | **0.500** | **0.400** | **0.444** | **0.597** | **8.3%** |
| **MLP Neural Net** | Deep Feedforward | **83.3%** | **0.333** | **0.250** | **0.286** | **0.542** | **12.5%** |

*Note: All models are evaluated on the exact same unseen chronological test partition with zero lookahead feature leakage.*

---

## 🛠️ Installation & Execution

### 1. Requirements
Ensure Python 3.10+ is installed. Install all project dependencies:

```bash
pip install -r requirements.txt
```

### 2. Running Automated Tests
Execute the standalone test suite verifying data pipelines, physical consistency equations, pandapower AC simulation, and AI model evaluation:

```bash
python -m pytest test_pipeline.py
```
*(Or run via python directly: `python -c "import test_pipeline; [getattr(test_pipeline, f)() for f in dir(test_pipeline) if f.startswith('test_')]"`)*

### 3. Launching the Command Center
Start the interactive Streamlit command dashboard:

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```text
.
├── app.py                  # 9-module Streamlit Cyber Command Center dashboard
├── cyber_physical.py       # Physics checks: P = V*I, voltage corridors, RoCoF, sensor health
├── power_simulation.py     # Pandapower AC power flow & IEEE 9/14 multi-bus simulation
├── data_preprocessing.py   # Research adapters, 6 attack vectors, leak-free forward pipelines
├── model_training.py       # Multi-model AI/DL training (IF, RF, XGB, MLP, PyTorch LSTM)
├── report_generator.py     # HTML & PDF-printable cyber incident forensic report engine
├── test_pipeline.py        # Complete automated test suite covering all modules (12 tests)
├── requirements.txt        # Pinned dependency requirements
└── README.md               # Architecture documentation & execution guide
```
