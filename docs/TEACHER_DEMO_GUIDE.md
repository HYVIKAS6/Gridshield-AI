# GridShield AI — Lecturer Live Demonstration & Defense Guide

**Academic Program:** Bachelor of Engineering (B.E.)  
**Project Title:** GridShield AI — Machine Learning-Based Smart Grid Cyberattack Detection and Simulation System  
**Prepared For:** Engineering Project Evaluation & Viva Voce Defense  
**Location:** Department of Computer Science / Electrical & Electronics Engineering  

---

## 1. Project Abstract
GridShield AI is an enterprise-grade cyber-physical anomaly detection and simulation platform tailored for modern electrical smart grids. As power networks integrate internet-connected Phasor Measurement Units (PMUs), SCADA remote telemetry units (RTUs), and automated industrial control systems, they become vulnerable to stealthy cyberattacks such as False Data Injection Attacks (FDIA), slow-ramping manipulation, and coordinated sensor dropouts.

GridShield AI addresses this threat by combining **domain electrical physics** (Ohm's Law active power consistency $P = V \cdot I$, statutory voltage corridors $[0.90, 1.10]\text{ p.u.}$, and Rate of Change of Frequency $|\text{RoCoF}| \le 0.5\text{ Hz/s}$) with **five state-of-the-art machine learning & deep learning architectures**:
1. **Isolation Forest** (Unsupervised Anomaly Detection)
2. **Random Forest** (Supervised Ensemble Bagging)
3. **XGBoost** (Extreme Gradient Boosted Trees)
4. **MLP Neural Network** (Feedforward Deep Learning)
5. **PyTorch LSTM** (Temporal Recurrent Deep Learning)

All pipelines enforce **strict chronological train-test splitting and leak-free forward preprocessing**, guaranteeing authentic, mathematically sound evaluation without lookahead bias.

---

## 2. Problem Statement & Research Objectives
### The Problem:
- Traditional substation security relies on static threshold alarms (e.g. overvoltage trips).
- Advanced adversaries craft **Stealth False Data Injection Attacks (FDIA)** that satisfy algebraic power-flow constraints while misguiding automatic generation control (AGC) or state estimation algorithms.
- Conventional ML projects frequently suffer from **data leakage** (e.g. fitting scalers across the full dataset or random train-test shuffling on time-series telemetry).

### Core Objectives:
1. Ingest both **authentic empirical research datasets** (Mississippi State University & ORNL Power System Attack Dataset) and high-fidelity **physics-constrained synthetic datasets**.
2. Build standalone, modular Python classes for each ML architecture adhering to scikit-learn and PyTorch conventions.
3. Eliminate data leakage through chronological splitting and rolling-window history context.
4. Provide an interactive **ML Training Studio** and **Live ML Prediction Lab** for live demonstration.
5. Couple machine learning decisions with Explainable AI (SHAP and Gini feature importances) and prescriptive mitigation guidance (IEEE 1547, IEC 61850, IEC 62351).

---

## 3. Dataset Architecture & Empirical Verification

| Dataset Name | Type | Source & Authors | Records | Key Features | Target Labels |
| :--- | :---: | :--- | :---: | :--- | :--- |
| **MSU / ORNL Power System Attack Dataset** | Empirical Research Benchmark | Dr. Tommy Morris, Dr. Uttam Adhikari, Dr. Shengyi Pan (Mississippi State Univ. & ORNL) | 2,200 | Voltage (`R1-PA1:VH`), Current (`R1-PA4:IH`), Active Load ($P = V \cdot I / 1000$), Frequency (`R1:F`) | 0 = Natural / Normal<br>1 = Cyber Attack |
| **Synthetic Smart Grid Telemetry** | High-Fidelity Physics Simulation | GridShield AI Domain Engine | 1,200 | $V$ (230V $\pm 0.5$V), $I$ (10A $\pm 2$A diurnal), $P$ ($V \cdot I/1000$), $f$ (50Hz $\pm 0.02$Hz) | 0 = Nominal<br>1 = 6 Attack Vectors |

### The 6 Simulated Cyberattack Vectors:
1. `voltage_spike`: Transient bus overvoltage ($+12\text{V}$ to $+25\text{V}$).
2. `frequency_shift`: Generator trip / PMU frequency spoofing ($\pm 0.8\text{Hz}$).
3. `load_drift`: Multiplicative scaling ($1.4\times$) violating physical $P = V \cdot I$ relation.
4. `stealth_fdia`: Coordinated multi-sensor scaling with added Gaussian noise.
5. `telemetry_dropout`: Communication blackout / sensor flatline to $0.0$.
6. `ramping_fdi`: Gradual slow-drift attack evading static thresholds over time.

---

## 4. Machine Learning Algorithms Explained

### 1. Isolation Forest (Unsupervised Learning)
- **Principle:** Anomalies are "few and different". It isolates abnormal points by randomly picking a feature and split value. Anomalous observations reside at shallower tree depths than dense normal clusters.
- **Why used in Smart Grid:** Detects novel zero-day attacks without requiring labeled historical attack data.

### 2. Random Forest (Supervised Bagging)
- **Principle:** Ensemble of 100 de-correlated decision trees trained on bootstrap samples with random feature sub-selection. Outputs majority vote.
- **Why used in Smart Grid:** High noise tolerance, models non-linear power curves, and produces Gini feature importance rankings to identify attacked sensors.

### 3. XGBoost (Supervised Boosting)
- **Principle:** Sequential gradient boosted trees minimizing regularized loss functions using second-order Taylor approximations (gradients & Hessians).
- **Why used in Smart Grid:** Industry-standard tabular accuracy; highly effective at detecting low-magnitude slow-drift attacks.

### 4. Multi-Layer Perceptron (MLP Neural Network)
- **Principle:** Feedforward neural network with dense layers $(64, 32)$, ReLU activations, Adam optimizer, and early stopping on held-out validation loss.
- **Why used in Smart Grid:** Rapid sub-millisecond inference latency ($< 0.5\text{ ms}$) suitable for embedded substation controllers.

### 5. PyTorch LSTM (Temporal Recurrent Deep Learning)
- **Principle:** Long Short-Term Memory network featuring Forget, Input, and Output gates. Evaluates sliding time-window sequences ($k=5$ intervals).
- **Why used in Smart Grid:** Preserves sequential memory over time; essential for detecting Ramping FDI attacks where single-timestep readings appear normal.

---

## 5. Leak-Free Evaluation Methodology
1. **Chronological Splitting:**
   - Training partition: Earliest $75\%$ of telemetry records.
   - Testing partition: Strictly forward $25\%$ held-out records.
   - Random shuffling is prohibited to prevent lookahead leakage.
2. **Pipeline Boundary Isolation:**
   - Feature scaler ($\mu, \sigma$) and rolling window context are fit solely on the training partition.
   - Test partition rolling statistics use backward boundary history from the training tail.

---

## 6. Step-by-Step 7-Minute Live Demonstration Script

Follow this exact sequence when presenting to your lecturer:

### Minute 1: Introduction & Problem Statement
- Open **VS Code** showing the clean folder structure:
  - `data/` (raw and processed research datasets)
  - `ml_models/` (modular model implementations)
  - `notebooks/` (dedicated Jupyter Notebook)
  - `train.py` & `predict.py`
  - `app.py`

### Minute 2: Dataset Inspection in VS Code or Terminal
- Show the raw MSU research dataset:
  ```bash
  python -c "import pandas as pd; df = pd.read_csv('data/processed/msu_pmu_benchmark.csv'); print(df.head()); print(df['target'].value_counts())"
  ```
- Explain that the dataset contains genuine Phasor Measurement Unit (PMU) telemetry from Mississippi State University & Oak Ridge National Laboratory.

### Minute 3: Standalone Model Training Execution
- Run the CLI training command in front of the lecturer:
  ```bash
  python train.py --dataset msu_benchmark --epochs 5
  ```
- Point out the terminal output:
  - Chronological split (1,650 train, 550 test)
  - Model progress
  - The comprehensive test evaluation table showing Accuracy, Precision, Recall, F1-Score, ROC-AUC, FPR, and Latency.

### Minute 4: Launching the Streamlit Command Center
- Start the dashboard:
  ```bash
  python -m streamlit run app.py
  ```
- Walk through the top hero metrics: Total intervals monitored, active threat level, and system status.

### Minute 5: Interactive ML Training Studio (Tab 4)
- Navigate to **Tab 4: 🛠️ ML Training Studio**.
- Select the architecture(s), adjust hyperparameters, and click **🚀 Start Live Model Training**.
- Show the real-time progress bar, timestamped training logs, and updated model comparison table.

### Minute 6: Live ML Prediction Lab (Tab 5)
- Navigate to **Tab 5: 🔬 Live ML Prediction Lab**.
- Select the **"🟢 Nominal Baseline"** preset $\rightarrow$ Click **Run Live Model Prediction** $\rightarrow$ Show **🟢 NORMAL (0.00% anomaly score)**.
- Select the **"🚨 Voltage Spike Attack"** preset $\rightarrow$ Click **Run Live Model Prediction** $\rightarrow$ Show **🚨 ATTACK (98.00% anomaly score, $< 50\text{ ms}$ latency)**.
- Explain the physical Ohm Law check displayed below the verdict.

### Minute 7: Explainable AI & Forensic Report Export (Tabs 7 & 12)
- Navigate to **Tab 7: 💡 Explainable AI (XAI)** $\rightarrow$ Show Gini feature importance and single-observation attribution.
- Navigate to **Tab 12: 📑 Forensics & Export Reports** $\rightarrow$ Click **💾 Download Incident Audit Report (.HTML)** and open it in the browser.

---

## 7. Expected Lecturer Questions & Model Answers

### Q1: "Why did you use chronological splitting instead of k-fold cross validation?"
**Answer:**  
*"In electrical power systems, telemetry is inherently sequential time-series data. Standard k-fold cross-validation or random shuffling selects points from the future to predict the past, causing temporal lookahead leakage. Chronological splitting mimics real-world SCADA deployment where a model trained on past observations must predict unseen future events."*

### Q2: "Why is accuracy alone not sufficient for evaluating grid cyberattacks?"
**Answer:**  
*"Real-world power systems operate normally more than $90-95\%$ of the time. In an imbalanced dataset, a naive model that always predicts 'Normal' achieves $95\%$ accuracy but fails to detect $100\%$ of cyberattacks. Therefore, we prioritize **Recall** (catching every attack), **F1-Score** (harmonic mean of precision and recall), and **False Positive Rate** (preventing nuisance alarms that cause control center fatigue)."*

### Q3: "What is the difference between Isolation Forest and Random Forest here?"
**Answer:**  
*"Isolation Forest is **unsupervised**; it trains without attack labels and isolates anomalies by identifying data points that require fewer random feature partitions. This is vital for detecting zero-day attacks. Random Forest is **supervised**; it uses labeled historical telemetry to construct 100 de-correlated decision trees with bagging, maximizing classification accuracy and providing Gini feature importances."*

### Q4: "How does the PyTorch LSTM handle temporal sequences?"
**Answer:**  
*"The LSTM takes sliding time-window sequences ($k=5$ timesteps) rather than isolated single-point readings. This allows its internal cell state and gates (Forget, Input, Output) to capture rate-of-change and temporal acceleration, making it the only model capable of detecting progressive Ramping FDI attacks."*

### Q5: "How does your system verify domain physics alongside machine learning?"
**Answer:**  
*"GridShield AI employs a dual-defense architecture: machine learning provides statistical pattern recognition, while our cyber-physical engine computes active power conservation ($P = V \cdot I \cdot \cos\theta / 1000$), statutory voltage corridor limits ($[0.90, 1.10]\text{ p.u.}$), and instantaneous Rate of Change of Frequency ($|\text{RoCoF}| \le 0.5\text{ Hz/s}$). If an attacker injects mathematically plausible false telemetry that violates electrical physics, the physical engine flags it immediately."*

---

## 8. Limitations & Future Work
- **Hardware-in-the-Loop (HIL):** Future research can interface GridShield AI with physical RTDS (Real-Time Digital Simulators) or OPAL-RT testbeds.
- **Multiclass Attack Attribution:** Expanding the neural network to output granular attack classifications (e.g. differentiating FDIA from DoS) across all PMU bus lines.
- **Automated Mitigation Actuation:** Transitioning prescriptive recommendations into automated, human-supervised breaker interlocks.
