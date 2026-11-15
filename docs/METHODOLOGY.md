# GridShield AI — Experimental Methodology & Validation 🔬

**Document Version:** 2.0  
**Status:** IMPLEMENTED AND TESTED  

---

## 1. Problem Formulation

Smart grid cyber-physical infrastructure relies heavily on Supervisory Control and Data Acquisition (SCADA) systems and Phasor Measurement Units (PMUs). Attackers can execute False Data Injection Attacks (FDIA), parameter ramping, or denial-of-service packet loss to disguise grid instability or provoke unwarranted relay tripping.

Traditional threshold relays fail when attackers inject coordinated perturbations that stay near nominal margins. GridShield AI addresses this by coupling **data-driven machine learning models** with **physics-informed domain rules**.

---

## 2. Experimental Data Pipeline & Leak-Free Guarantee

To prevent temporal lookahead leakage, data preprocessing strictly enforces chronological progression:

1. **Chronological Splitting:**
   - Partitioning ratio: $75\%$ historical train, $25\%$ future test (`chronological_train_test_split`).
   - Observations are never shuffled randomly.
2. **Feature Engineering & Scaler Fitting:**
   - `StandardScaler` computes mean ($\mu_{train}$) and standard deviation ($\sigma_{train}$) exclusively from training records.
   - Forward rolling statistics ($5$-step moving average) and lag features are computed iteratively. During test set evaluation, only the historical tail boundary context is prepended.
3. **No Target Leakage:**
   - Ground truth labels (`target`) are strictly excluded from feature inputs.

---

## 3. Evaluated Machine Learning & Deep Learning Models

### 1. Isolation Forest (Unsupervised Anomaly Detection)
- **Objective:** Detect deviations without requiring labeled attack samples during training.
- **Methodology:** Calibrates tree depth path length; contamination hyperparameter is automatically bounded ($0.01 - 0.45$). Evaluated using decision function negation.

### 2. Random Forest Classifier (Supervised Ensemble)
- **Objective:** Robust non-linear decision boundaries between normal load fluctuations and cyber attacks.
- **Hyperparameters:** $100$ trees, balanced class weights, Gini impurity split criterion.

### 3. XGBoost Classifier (Gradient Boosted Decision Trees)
- **Objective:** High-precision gradient boosting focusing iteratively on hard-to-classify boundary samples.
- **Hyperparameters:** Depth 4, learning rate 0.08, log-loss objective.

### 4. Multi-Layer Perceptron (MLP Neural Network)
- **Objective:** Deep feedforward neural network baseline.
- **Architecture:** Two hidden layers $(64, 32)$, ReLU non-linearities, Adam optimizer with validation early stopping.

### 5. Temporal PyTorch LSTM (Deep Recurrent Sequence Detector)
- **Objective:** Sequential temporal anomaly detection capturing multi-timestep correlations.
- **Architecture:** 2 LSTM layers ($32$ hidden units), batch normalization, dropout $0.2$, linear classification head trained with binary cross-entropy with logits.

---

## 4. Evaluation Metrics & Class Imbalance Handling

Because cyberattack intervals may be sparse, accuracy alone is insufficient. GridShield AI prioritizes:
- **Precision:** $\frac{TP}{TP + FP}$ (minimizing false alarm operational disruptions).
- **Recall (Detection Rate):** $\frac{TP}{TP + FN}$ (ensuring attacks are not missed).
- **F1-Score:** Harmonic mean of Precision and Recall.
- **ROC-AUC & False Positive Rate (FPR):** Discriminative capacity across all threshold settings.
