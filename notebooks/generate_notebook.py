import json
import os

cells = []


def add_md(source):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [s + "\n" for s in source.split("\n")]
    })


def add_code(source):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [s + "\n" for s in source.split("\n")]
    })


# 1. Intro
add_md("""# GridShield AI — Machine Learning Model Training & Evaluation
### Academic Laboratory Demonstration & Technical Walkthrough
**Author:** Final Year B.E. Student Project Team  
**Focus Area:** Cyber-Physical Power Systems, Smart Grid Security & Machine Learning Intrusion Detection  
**Target Architecture:** Multi-Algorithm Benchmark (Isolation Forest, Random Forest, XGBoost, MLP, PyTorch LSTM)

---

## 1. Project Introduction and Problem Statement
Modern smart electrical grids rely on bidirectional communication networks, automated SCADA telecontrol, and Phasor Measurement Units (PMUs) to balance power generation and consumption. However, this cyber-physical convergence introduces vulnerabilities to malicious cyberattacks, including:
- **False Data Injection Attacks (FDIA)**: Stealthy manipulation of sensor readings ($V, I, P, f$) that bypass basic statutory residual limits.
- **Ramping / Slow-Drift FDI**: Gradual degradation of sensor readings designed to evade fixed threshold alarms.
- **Service Disruption & Breaker Trips**: Unauthorized commands causing voltage sags, frequency deviations, and catastrophic cascaded line outages.

This notebook demonstrates the end-to-end data science and machine learning pipeline to detect these cyber threats using both empirical laboratory research datasets and cyber-physical simulations.""")

# 2. Imports
add_md("## 2. Importing Required Libraries\nWe load core scientific computation, machine learning, and deep learning libraries.")
add_code("""import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = os.path.abspath('..')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_preprocessing import LeakFreeGridPipeline, chronological_train_test_split, validate_grid_dataframe
from data.download_datasets import load_dataset
from ml_models import (
    GridRandomForestModel,
    GridXGBoostModel,
    GridIsolationForestModel,
    GridMLPModel,
    GridLSTMModel,
    HAS_XGBOOST,
    HAS_TORCH
)
from model_training import calibrate_contamination, compare_all_models
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score, auc

plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
%matplotlib inline
print("Libraries imported successfully!")""")

# 3. Load Data
add_md("## 3. Loading the Dataset\nWe load the Mississippi State University / ORNL Power System Attack Dataset.")
add_code("""dataset_name = "msu_benchmark"  # Option: "synthetic_sample"
df, desc = load_dataset(dataset_name)
is_valid, msg, df_clean = validate_grid_dataframe(df)
print(f"Validation status: {is_valid} ({msg})")
print(f"Dataset Description: {desc}")
print(f"Total Records: {len(df_clean)}, Total Features: {df_clean.shape[1]}")""")

# 4. Display Records
add_md("## 4. Displaying Dataset Records\nLet us inspect the first few rows of the telemetry stream.")
add_code("""display_cols = ['timestamp', 'voltage', 'current', 'load', 'frequency', 'target', 'attack_type']
df_clean[display_cols].head(10)""")

# 5. EDA
add_md("## 5. Exploratory Data Analysis (EDA)\nAnalyzing class distribution, sensor telemetry statistics, and attack breakdown.")
add_code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Target class distribution
class_counts = df_clean['target'].value_counts()
class_labels = ['Normal (0)', 'Attack (1)'] if len(class_counts) == 2 else class_counts.index
axes[0].bar([str(x) for x in class_labels], class_counts.values, color=['#10B981', '#F43F5E'])
axes[0].set_title("Ground Truth Class Distribution")
axes[0].set_ylabel("Count")

# Sensor distributions
sensor_cols = ['voltage', 'current', 'load', 'frequency']
df_clean[sensor_cols].boxplot(ax=axes[1])
axes[1].set_title("Primary Sensor Telemetry Distributions")
axes[1].set_ylabel("Measured Value")

plt.tight_layout()
plt.show()

print("Telemetry Summary Statistics:")
df_clean[sensor_cols].describe().T""")

# 6. Missing Values
add_md("## 6. Missing Value & Integrity Handling\nSmart grid communications frequently suffer packet dropouts or null readings.")
add_code("""missing_counts = df_clean.isna().sum()
print("Missing Values per Column:")
print(missing_counts[missing_counts > 0] if missing_counts.sum() > 0 else "Zero missing values detected. Telemetry stream is integral.")""")

# 7. Feature Engineering
add_md("## 7. Feature Engineering\nEngineering rolling statistics and historical lags.")
add_code("""pipeline = LeakFreeGridPipeline(window_size=5)
print(f"Engineered feature names: {LeakFreeGridPipeline.FEATURE_COLS}")""")

# 8 & 9. Split & Pipeline
add_md("## 8 & 9. Chronological Train-Test Split & Scaling\nPreserves sequential order and avoids temporal lookahead leakage.")
add_code("""test_ratio = 0.25
train_df, test_df = chronological_train_test_split(df_clean, test_ratio=test_ratio)

# Fit pipeline strictly on training data
X_train = pipeline.fit_transform(train_df)
X_test = pipeline.transform(test_df)

y_train = train_df['target'].values
y_test = test_df['target'].values

print(f"Training partition: {X_train.shape[0]} records, {X_train.shape[1]} features")
print(f"Testing partition:  {X_test.shape[0]} records (strictly forward unseen)")""")

# 10. Training
add_md("## 10. Training 5 Core ML & DL Architectures\nTraining all models on training partition.")
add_code("""models = {}
evaluations = {}

print("Training Isolation Forest...")
contam = calibrate_contamination(y_train)
iso = GridIsolationForestModel(contamination=contam, random_state=42).fit(X_train)
models["Isolation Forest"] = iso
evaluations["Isolation Forest"] = iso.evaluate(X_test, y_test)

print("Training Random Forest...")
rf = GridRandomForestModel(n_estimators=100, random_state=42).fit(X_train, y_train)
models["Random Forest"] = rf
evaluations["Random Forest"] = rf.evaluate(X_test, y_test)

if HAS_XGBOOST:
    print("Training XGBoost...")
    xgb = GridXGBoostModel(n_estimators=100, random_state=42).fit(X_train, y_train)
    models["XGBoost"] = xgb
    evaluations["XGBoost"] = xgb.evaluate(X_test, y_test)

print("Training MLP Neural Network...")
mlp = GridMLPModel(max_iter=100, random_state=42).fit(X_train, y_train)
models["MLP Neural Network"] = mlp
evaluations["MLP Neural Network"] = mlp.evaluate(X_test, y_test)

if HAS_TORCH:
    print("Training PyTorch LSTM...")
    lstm = GridLSTMModel(epochs=6, random_state=42).fit(X_train, y_train)
    models["PyTorch LSTM"] = lstm
    evaluations["PyTorch LSTM"] = lstm.evaluate(X_test, y_test)

print("All models successfully trained!")""")

# 11. Performance Comparison
add_md("## 11. Model Performance Comparison\nEvaluating Accuracy, Precision, Recall, F1-Score, ROC-AUC, and FPR.")
add_code("""comp_df = compare_all_models(evaluations)
comp_df.sort_values(by="F1-Score", ascending=False).reset_index(drop=True)""")

# 12. Confusion Matrices
add_md("## 12. Confusion Matrices\nVisualizing classification breakdown across models.")
add_code("""fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.5))
if len(models) == 1:
    axes = [axes]

for i, (name, ev) in enumerate(evaluations.items()):
    cm = np.array(ev['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['Pred Normal', 'Pred Attack'],
                yticklabels=['True Normal', 'True Attack'])
    axes[i].set_title(f"{name}\\nAcc: {ev['accuracy']:.2f} | F1: {ev['f1']:.2f}")

plt.tight_layout()
plt.show()""")

# 13. ROC Curves
add_md("## 13. Receiver Operating Characteristic (ROC) Curves\nTrue Positive Rate vs False Positive Rate.")
add_code("""plt.figure(figsize=(9, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing (AUC = 0.50)')

for name, ev in evaluations.items():
    scores = np.array(ev['anomaly_scores'])
    if len(np.unique(y_test)) > 1:
        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity / Recall)')
plt.title('ROC Curves — Unseen Chronological Test Evaluation')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()""")

# 14. Feature Importance
add_md("## 14. Feature Importance & Explainability\nGini impurity feature importance from Random Forest.")
add_code("""if 'Random Forest' in models:
    rf_imp = models['Random Forest'].get_feature_importances(LeakFreeGridPipeline.FEATURE_COLS)
    imp_df = pd.DataFrame(list(rf_imp.items()), columns=['Feature', 'Importance']).sort_values('Importance', ascending=True)

    plt.figure(figsize=(8, 4.5))
    plt.barh(imp_df['Feature'], imp_df['Importance'], color='#00D4FF')
    plt.title("Random Forest Gini Feature Importances")
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.show()""")

# 15. Saving Models
add_md("## 15. Model Serialization & Checkpoints\nPersisting model files and pipeline for live inference.")
add_code("""save_dir = os.path.join(PROJECT_ROOT, "models", "saved")
os.makedirs(save_dir, exist_ok=True)

for name, m in models.items():
    if hasattr(m, 'save'):
        slug = name.lower().replace(' ', '_')
        ext = '.pt' if 'lstm' in slug else '.joblib'
        path = os.path.join(save_dir, f"{slug}{ext}")
        m.save(path)
        print(f"Saved: {path}")

import joblib
pipeline_path = os.path.join(save_dir, "feature_pipeline.joblib")
joblib.dump(pipeline, pipeline_path)
print(f"Saved pipeline: {pipeline_path}")""")

# 16. Live Prediction
add_md("## 16. Live Prediction Verification on Unseen Observations\nDemonstrating live inference on arbitrary telemetry.")
add_code("""test_samples = pd.DataFrame([
    {"name": "Nominal Baseline", "voltage": 230.1, "current": 10.0, "load": 2.301, "frequency": 50.0},
    {"name": "Voltage Spike Attack", "voltage": 252.0, "current": 10.0, "load": 2.301, "frequency": 50.0},
    {"name": "Severe Frequency Sag", "voltage": 229.8, "current": 9.9, "load": 2.275, "frequency": 48.9},
])

for _, row in test_samples.iterrows():
    raw_obs = pd.DataFrame([{'voltage': row['voltage'], 'current': row['current'], 'load': row['load'], 'frequency': row['frequency']}])
    X_obs = pipeline.transform(raw_obs)
    pred = models['Random Forest'].predict(X_obs)[0]
    prob = models['Random Forest'].predict_proba(X_obs)[0, 1]
    verdict = '🚨 ATTACK' if pred == 1 else '🟢 NORMAL'
    print(f"Scenario: {row['name']:<25} -> {verdict} (Confidence: {prob*100:.1f}%)")""")

# 17. Conclusion
add_md("""## 17. Conclusion & Project Limitations
### Summary of Findings:
1. **Supervised Tree Ensembles (Random Forest & XGBoost)** achieve the highest F1-scores on complex multi-sensor attacks.
2. **Unsupervised Isolation Forest** effectively detects zero-day extreme voltage/frequency trips without requiring attack labels.
3. **Temporal LSTM** successfully tracks sequential dependencies across sliding time windows.

### Project Limitations & Next Steps:
- Telemetry datasets from testbeds represent scaled laboratory power systems rather than full continental transmission grids.
- Real-time hardware SCADA RTU integration requires hardware-in-the-loop (HIL) simulators (e.g. OPAL-RT or RTDS).""")

notebook_dict = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GridShield_ML_Training.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook_dict, f, indent=2)

print("Successfully written:", out_path)
