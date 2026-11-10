"""
Machine Learning Training, Model Calibration, and Comprehensive Evaluation Engine
for GridShield AI Smart Grid Cyberattack Simulator.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier


def calibrate_contamination(y_train: Optional[Union[np.ndarray, list]], default: float = 0.05) -> float:
    """
    Calibrates contamination hyperparameter for Isolation Forest.
    Guarantees a valid float strictly within (0.001, 0.5) to satisfy scikit-learn requirements.
    """
    if y_train is None or len(y_train) == 0:
        return default

    y_arr = np.array(y_train)
    positive_count = np.sum(y_arr == 1)
    ratio = float(positive_count) / float(len(y_arr))

    # Bound contamination cleanly to prevent sklearn errors
    if ratio <= 0.001:
        return 0.01  # small expected baseline anomaly rate
    return min(max(ratio, 0.01), 0.45)


def train_isolation_forest(
    X_train: np.ndarray,
    contamination: Union[float, str] = 0.08,
    random_state: int = 42,
) -> IsolationForest:
    """
    Trains an Isolation Forest model for unsupervised anomaly detection.
    """
    if isinstance(contamination, (int, float)):
        bounded_contam = min(max(float(contamination), 0.005), 0.49)
    else:
        bounded_contam = "auto"

    model = IsolationForest(
        contamination=bounded_contam,
        random_state=random_state,
        n_estimators=100,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


def evaluate_isolation_forest(
    model: IsolationForest,
    X_test: np.ndarray,
    y_test: Union[np.ndarray, list],
) -> Dict[str, Any]:
    """
    Evaluates Isolation Forest on unseen test data.
    Computes Accuracy, Precision, Recall, F1, ROC-AUC (via decision scores),
    False Positive Rate, and Confusion Matrix.
    """
    y_true = np.array(y_test, dtype=int)
    # Predict raw labels: 1 is inlier (normal), -1 is outlier (attack)
    raw_preds = model.predict(X_test)
    y_pred = np.where(raw_preds == -1, 1, 0)

    # Anomaly scores: lower decision_function means more abnormal
    scores = -model.decision_function(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fpr = float(fp) / float(fp + tn) if (fp + tn) > 0 else 0.0

    # ROC-AUC (requires at least one positive and one negative sample)
    try:
        if len(np.unique(y_true)) > 1:
            auc = float(roc_auc_score(y_true, scores))
        else:
            auc = 0.5
    except Exception:
        auc = 0.5

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc),
        "false_positive_rate": float(fpr),
        "confusion_matrix": cm.tolist(),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "predictions": y_pred.tolist(),
        "anomaly_scores": scores.tolist(),
    }


def train_mlp_model(
    X_train: np.ndarray,
    y_train: Union[np.ndarray, list],
    random_state: int = 42,
    hidden_layer_sizes: Tuple[int, ...] = (64, 32),
) -> MLPClassifier:
    """
    Trains a Multi-layer Perceptron (MLP) Classifier as a robust supervised baseline.
    """
    y_arr = np.array(y_train, dtype=int)
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=300,
        random_state=random_state,
        early_stopping=True,
        n_iter_no_change=10,
    )
    model.fit(X_train, y_arr)
    return model


def evaluate_mlp_model(
    model: MLPClassifier,
    X_test: np.ndarray,
    y_test: Union[np.ndarray, list],
) -> Dict[str, Any]:
    """
    Evaluates MLP Classifier on unseen test data.
    """
    y_true = np.array(y_test, dtype=int)
    y_pred = model.predict(X_test)

    # Probabilities for class 1 (attack)
    if hasattr(model, "predict_proba"):
        prob_matrix = model.predict_proba(X_test)
        if prob_matrix.shape[1] > 1:
            scores = prob_matrix[:, 1]
        else:
            scores = prob_matrix[:, 0]
    else:
        scores = y_pred.astype(float)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fpr = float(fp) / float(fp + tn) if (fp + tn) > 0 else 0.0

    try:
        if len(np.unique(y_true)) > 1:
            auc = float(roc_auc_score(y_true, scores))
        else:
            auc = 0.5
    except Exception:
        auc = 0.5

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc),
        "false_positive_rate": float(fpr),
        "confusion_matrix": cm.tolist(),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "predictions": y_pred.tolist(),
        "anomaly_scores": scores.tolist(),
    }
