# GridShield AI — System Architecture & Implementation Guide ⚡

**Document Version:** 2.0  
**Project Title:** Machine Learning-Based Cyberattack Detection and Anomaly Monitoring for Smart Grid Systems  
**Status:** IMPLEMENTED AND TESTED  

---

## 1. High-Level System Architecture

GridShield AI is organized as a layered, modular cyber-physical monitoring platform:

```text
                           +-----------------------------------------------+
                           |        Smart Grid Telemetry Ingestion         |
                           |   (Offline Sample, Research CSV, Synthetic)   |
                           +-----------------------+-----------------------+
                                                   |
                                                   v
                           +-----------------------------------------------+
                           |      Schema Validator & Sanitizer Engine      |
                           |   (V1, I1, Freq, P, Label PMU Normalization)  |
                           +-----------------------+-----------------------+
                                                   |
                                                   v
                           +-----------------------------------------------+
                           |      Chronological Split & Pipeline Eng.      |
                           |    (Leak-Free Scaling, Window & Lag States)   |
                           +-----------+-----------------------+-----------+
                                       |                       |
               +-----------------------+                       +-----------------------+
               v                                                                       v
+---------------------------------------------+                         +---------------------------------------------+
|    Cyber-Physical Diagnostic Validation     |                         |      Multi-Model AI / Deep Learning Engine  |
|  - Active Power Law: P = V * I * pf / 1000  |                         |  - Isolation Forest (Unsupervised Anomaly)  |
|  - Voltage Corridors: [0.90, 1.10] p.u.     |                         |  - Random Forest Classifier (Bagging)       |
|  - Frequency Corridors & RoCoF (|df/dt|)    |                         |  - XGBoost (Gradient Boosted Decision Trees)|
|  - Sensor Flatline & Packet Drop Detection  |                         |  - MLP Neural Network (Deep Baseline)       |
+----------------------+----------------------+                         |  - PyTorch LSTM (Recurrent Sequence Model)  |
                       |                                                +----------------------+----------------------+
                       |                                                                       |
                       +-----------------------+                       +-----------------------+
                                               |                       |
                                               v                       v
                               +-----------------------------------------------+
                               |          Explainable AI (XAI) Engine          |
                               |    (SHAP TreeExplainer & Feature Rankings)    |
                               +-----------------------+-----------------------+
                                                       |
                                                       v
                               +-----------------------------------------------+
                               |       IEEE Multi-Bus Simulation Engine        |
                               |   (Pandapower AC Newton-Raphson Solver)       |
                               +-----------------------+-----------------------+
                                                       |
                                                       v
                               +-----------------------------------------------+
                               |    Streamlit Industrial SOC Command Center    |
                               |   (10 Operational Monitoring & Audit Modules) |
                               +-----------------------------------------------+
```

---

## 2. Core Modules & Responsibilities

1. **`data_preprocessing.py`**:
   - Ingestion adapters for research datasets (DOI: `10.21227/symr-bz19` and Zenodo: `20004612`).
   - `LeakFreeGridPipeline`: Scikit-learn transformer ensuring scaling and rolling windows fit strictly on training partitions.
   - 6 configurable dynamic attack vectors (voltage surge, frequency sag, load drift, stealth FDIA, telemetry dropout, ramping FDI).

2. **`model_training.py`**:
   - Implementation and benchmark orchestration for **Isolation Forest**, **Random Forest**, **XGBoost**, **MLP**, and **PyTorch LSTM**.
   - Strict time-series validation with confusion matrix, F1-score, ROC-AUC, and False Positive Rate metrics.

3. **`cyber_physical.py`**:
   - Physics-informed rule engine calculating active power discrepancies, voltage bounds, RoCoF rate-of-change, and sensor dropouts.

4. **`explainability.py`**:
   - SHAP TreeExplainer integration and heuristic deviation models attributing attack predictions to specific physical sensors.

5. **`model_registry.py`**:
   - Disk serialization of trained models (`.joblib` and `.pt`) and timestamped evaluation metadata in `models/saved/` and `models/metadata/`.

6. **`power_simulation.py`**:
   - Pandapower AC power flow calculations on IEEE 14-bus and IEEE 9-bus networks with bus-level attack injection.

7. **`report_generator.py`**:
   - Forensic HTML audit report generator with printable CSS formatting and MITRE ATT&CK mapping.

8. **`app.py`**:
   - Streamlit 10-module cybersecurity monitoring dashboard.
