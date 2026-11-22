# GridShield AI — ML-Based Smart Grid Cyberattack Detection & Simulation System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework: Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Deep Learning: PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-EE4C2C.svg)](https://pytorch.org/)
[![Tests: Pytest Passing](https://img.shields.io/badge/Tests-23%20Passed-brightgreen.svg)]()
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-00D4FF?style=for-the-badge&logo=streamlit)](https://gridshield-ai-9rfgltanyslf9tuihswpd2.streamlit.app/)

> **🌐 Live Interactive Application:** [https://gridshield-ai-9rfgltanyslf9tuihswpd2.streamlit.app/](https://gridshield-ai-9rfgltanyslf9tuihswpd2.streamlit.app/)  
> **Final Year B.E. Engineering Project**  
> An academic, technically defensible, and interactive machine learning platform combining electrical power domain physics, empirical smart-grid PMU datasets, multi-algorithm AI benchmarking, and real-time cyber-incident forensics.

---

## ⚡ Key Highlights & Architecture Upgrades

1. **Authentic Empirical Research Dataset:**
   - Ingests the **Mississippi State University (MSU) & Oak Ridge National Laboratory (ORNL) Power System Attack Dataset** (Phasor Measurement Unit logs: $V_a, I_a, P, f$ under natural events vs. cyberattacks).
   - Preserves offline synthetic domain telemetry (1,200 records) with 6 reproducible cyberattack vectors.
2. **Modular Machine Learning & Deep Learning Engine (`ml_models/`):**
   - **Isolation Forest** (`ml_models/isolation_forest.py`): Unsupervised anomaly detection for zero-day threats.
   - **Random Forest** (`ml_models/random_forest.py`): Supervised ensemble bagging with Gini feature importances.
   - **XGBoost Classifier** (`ml_models/xgboost_model.py`): Regularized gradient boosting for tabular telemetry.
   - **MLP Neural Network** (`ml_models/mlp_model.py`): Multi-layer perceptron feedforward architecture.
   - **PyTorch LSTM** (`ml_models/lstm_model.py`): 2-layer temporal recurrent neural network evaluating sliding time sequences ($k=5$).
3. **Leak-Free Chronological Splitting & Forward Pipelines:**
   - Evaluates strictly on forward holdout intervals ($75\%$ train, $25\%$ test).
   - Rolling means and scalers fit only on past training context, eliminating lookahead data leakage.
4. **Interactive ML Training Studio (Tab 4):**
   - Configurable hyperparameter sliders (trees, learning rates, epochs).
   - Real-time training progress bar, elapsed training timer, and timestamped operational logs.
5. **Live ML Prediction Lab (Tab 5):**
   - Instant single-observation and preset telemetry inferencing.
   - Live prediction confidence score, execution latency in milliseconds, and physical Ohm's Law consistency verification ($P = V \cdot I / 1000$).
6. **Dedicated Classroom Jupyter Notebook (`notebooks/GridShield_ML_Training.ipynb`):**
   - 17 fully documented sections covering EDA, leak-free pipeline, model training, ROC curves, confusion matrices, and model serialization.
7. **Lecturer Defense & Demonstration Guide (`docs/TEACHER_DEMO_GUIDE.md`):**
   - Concise 7-minute presentation workflow, architecture diagrams, and comprehensive Q&A preparation.

---

## 📁 Repository Structure

```text
Gridshield-AI-main/
├── app.py                     # Streamlit Industrial Command Center (12 Modules)
├── train.py                   # Standalone CLI training engine (multi-model)
├── predict.py                 # Standalone CLI prediction engine (presets & custom telemetry)
├── requirements.txt           # Project dependencies
├── pytest.ini                 # Pytest configuration
├── test_pipeline.py           # Core pipeline & pandapower unit tests
├── test_streamlit_app.py      # Streamlit AppTest integration tests
├── data_preprocessing.py      # Leak-free forward pipeline & 6 attack vectors
├── model_training.py          # Unified model training & evaluation adapter
├── cyber_physical.py          # Physics consistency: P = V*I, voltage corridor, RoCoF
├── power_simulation.py        # AC Newton-Raphson power flow on IEEE 9 & 14-bus networks
├── explainability.py          # SHAP & Gini feature importance attribution
├── report_generator.py        # Printable HTML & forensic cyber incident reports
├── data/
│   ├── download_datasets.py   # Automated reproducible dataset acquisition script
│   ├── DATASET_DOCUMENTATION.md # Academic dataset references & schemas
│   ├── raw/                   # Raw empirical PMU logs (MSU/ORNL data1.csv)
│   ├── processed/             # Balanced benchmark datasets (msu_pmu_benchmark.csv)
│   └── sample/                # Offline synthetic telemetry (smartgrid_telemetry_sample.csv)
├── ml_models/                 # Dedicated modular model classes
│   ├── base_model.py          # Abstract base class & standardized evaluation protocol
│   ├── random_forest.py       # Scikit-learn Random Forest bagging classifier
│   ├── xgboost_model.py       # XGBoost gradient boosted decision tree classifier
│   ├── isolation_forest.py    # Unsupervised Isolation Forest anomaly detector
│   ├── mlp_model.py           # Deep Multi-Layer Perceptron neural network
│   └── lstm_model.py          # PyTorch Long Short-Term Memory recurrent network
├── models/
│   ├── saved/                 # Serialized model checkpoints (.joblib and .pt)
│   └── metadata/              # Training run metadata and evaluation metrics
├── notebooks/
│   └── GridShield_ML_Training.ipynb # Step-by-step academic training notebook
├── tests/
│   └── test_ml_suite.py       # Comprehensive modular ML and dataset test suite
└── docs/
    └── TEACHER_DEMO_GUIDE.md  # 7-Minute presentation script & viva voce Q&A
```

---

## 🚀 Quick Start Guide

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone <repository-url>
cd Gridshield-AI-main
pip install -r requirements.txt
```

### 2. Standalone Model Training
Train all 5 architectures on the authentic MSU research dataset or synthetic benchmark:
```bash
# Train on MSU / ORNL Research Dataset
python train.py --dataset msu_benchmark --epochs 5

# Or train on Synthetic Simulation Benchmark
python train.py --dataset synthetic_sample --epochs 10
```

### 3. Live Prediction CLI
Run live inference using presets or custom telemetry inputs:
```bash
# Evaluate an overvoltage attack
python predict.py --model random_forest --preset voltage_spike

# Evaluate nominal normal operation
python predict.py --model random_forest --preset normal

# Evaluate custom sensor inputs
python predict.py --model random_forest --voltage 230.1 --current 10.0 --load 2.301 --frequency 50.0
```

### 4. Launching the Interactive Dashboard
Launch the Streamlit Industrial SOC Command Center:
```bash
python -m streamlit run app.py
```
Open `http://localhost:8501` in your web browser.

### 5. Running Automated Tests
Execute the comprehensive automated test suite (23 passing tests):
```bash
pytest
```

---

## 🎓 Academic Demonstration Sequence for Lecturers

1. **Step 1: Dataset Inspection** — Open `data/processed/msu_pmu_benchmark.csv` and show real PMU synchrophasor columns (`voltage`, `current`, `frequency`, `target`).
2. **Step 2: CLI Training** — Execute `python train.py --dataset msu_benchmark --epochs 5` to show live leak-free evaluation.
3. **Step 3: Interactive Dashboard** — Launch `app.py` and open **Tab 4 (ML Training Studio)**. Train individual models and observe real-time progress.
4. **Step 4: Live Prediction Lab** — Switch to **Tab 5 (Live ML Prediction Lab)**. Test normal vs. attack presets live.
5. **Step 5: Physics & Forensics** — Review physical power consistency ($P = V \cdot I$) in **Tab 8** and export the audit HTML report in **Tab 12**.

---

## 📜 Citation & Credits
- **MSU / ORNL Dataset:** Tommy Morris, Uttam Adhikari, Shengyi Pan (Mississippi State University & Oak Ridge National Laboratory).
- **Power Simulation:** Pandapower AC Newton-Raphson load-flow engine (Fraunhofer IEE / University of Kassel).
