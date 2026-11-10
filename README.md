# GridShield AI ⚡
### Intelligent Cyberattack Detection and Simulation Engine for Smart Grids

GridShield AI is an interactive platform designed for power systems researchers and AI engineers to simulate, detect, and respond to cyber threats in smart grids. It targets **False Data Injection Attacks (FDIA)**, **Frequency Deviation/Sag Attacks**, and **Coordinated Sensor Tampering** in critical energy infrastructure.

---

## 🚀 Features

- **Physics-Consistent Telemetry Generation:** Generates synthetic smart grid measurements ($V$, $I$, $P = V \cdot I$, $f$) with configurable random seeds and baseline noise.
- **Dynamic Stateful Attack Studio:** Interactively inject cyberattacks into live telemetry streams with persistence across Streamlit reruns:
  - **Voltage Spike / Surge (+V)**: High-magnitude transient excursions.
  - **Frequency Sag / Shift (±f)**: Frequency manipulation violating grid stability bounds ($[49.8 - 50.2]\text{ Hz}$).
  - **Load Inconsistency / Drift (*P)**: Sensor desynchronization violating Ohm's and Kirchhoff's laws.
  - **Stealth Coordinated FDIA**: Simultaneous current and active power alterations to bypass scalar checks.
- **Leak-Free Machine Learning Pipeline:**
  - **Chronological Time-Series Splitting**: Strict forward-looking validation without temporal lookahead or data shuffling leakage.
  - **Isolated Scaling & Feature Engineering**: `StandardScaler`, rolling window averages, and lag features fitted strictly on training data.
- **Dual-Model Anomaly Detection:**
  - **Isolation Forest (Unsupervised)**: Calibrated contamination based on baseline training rates with score-based ROC-AUC and confusion matrices.
  - **Multi-Layer Perceptron (Supervised NN)**: Evaluates classification performance (Accuracy, Precision, Recall, F1, ROC-AUC, FPR).
- **Interactive Multi-Sensor Visualization**: Plotly multi-panel dashboards displaying Voltage, Frequency corridor, Current, and Active Load with cyberattack event markers.
- **Context-Aware Mitigation Advisory**: Real-time recommendations derived from chronological anomaly sequences (e.g. Fast Frequency Response, PMU re-authentication, tap-changer regulation).

---

## 📂 Project Structure

```text
.
├── app.py                  # Streamlit dashboard and stateful UI logic
├── data_preprocessing.py   # Physics-consistent generation, dynamic attack injection, leak-free pipeline
├── model_training.py       # Isolation Forest & MLP training, dynamic contamination, multi-metric evaluation
├── test_pipeline.py        # Automated test suite for preprocessing, attack injection, and leakage prevention
├── test_streamlit_app.py   # Streamlit AppTest suite verifying session state persistence and UI flow
├── requirements.txt        # Pinned project dependencies (including openpyxl and pytest)
└── README.md               # Architecture documentation and usage guide
```

---

## 🛠️ Installation & Local Setup

1. **Activate the Virtual Environment:**
   ```bash
   .\.venv\Scripts\activate
   ```
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Automated Test Suite:**
   ```bash
   pytest -v
   ```
4. **Launch the Dashboard:**
   ```bash
   streamlit run app.py
   ```

---

## 🧠 ML & Evaluation Architecture

1. **Data Splitting**: Chronological time-series split (`test_ratio=0.2`) ensures the models are tested strictly on unseen future intervals.
2. **Feature Engineering**: Features include primary telemetry (`voltage`, `current`, `load`, `frequency`), 5-hour rolling voltage mean, and 1-hour lag load. Rolling features for test sets strictly ingest training boundary context.
3. **Evaluation Metrics**:
   - Classification Accuracy, Precision, Recall, and F1 Score
   - Receiver Operating Characteristic Area Under Curve (ROC-AUC)
   - False Positive Rate (FPR)
   - Real-time 2x2 Confusion Matrix Heatmaps
