"""
Machine Learning Training, Model Calibration, and Comprehensive Evaluation Engine
for GridShield AI Smart Grid Cyberattack Simulator.
"""

import logging
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
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

logger = logging.getLogger(__name__)

# Optional XGBoost import with graceful fallback
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    xgb = None
    XGBClassifier = None
    HAS_XGBOOST = False
    logger.warning("XGBoost library is not installed or failed to import.")


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

    # Anomaly scores: lower decision_function means more abnormal (negated for higher = more anomalous)
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


def train_random_forest(
    X_train: np.ndarray,
    y_train: Union[np.ndarray, list],
    n_estimators: int = 100,
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestClassifier:
    """
    Trains a Random Forest Classifier for supervised grid anomaly detection.
    """
    y_arr = np.array(y_train, dtype=int)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_arr)
    return model


def evaluate_random_forest(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: Union[np.ndarray, list],
) -> Dict[str, Any]:
    """
    Evaluates Random Forest Classifier on unseen test data.
    Computes Accuracy, Precision, Recall, F1, ROC-AUC, FPR, and Confusion Matrix.
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


def train_xgboost(
    X_train: np.ndarray,
    y_train: Union[np.ndarray, list],
    n_estimators: int = 100,
    random_state: int = 42,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    n_jobs: int = -1,
) -> Optional[Any]:
    """
    Trains an XGBoost Classifier with graceful fallback if xgboost is unavailable.
    """
    if not HAS_XGBOOST or XGBClassifier is None:
        logger.warning("XGBoost is not installed or unavailable in this environment.")
        return None

    y_arr = np.array(y_train, dtype=int)
    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
        eval_metric="logloss",
    )
    model.fit(X_train, y_arr)
    return model


def evaluate_xgboost(
    model: Optional[Any],
    X_test: np.ndarray,
    y_test: Union[np.ndarray, list],
) -> Optional[Dict[str, Any]]:
    """
    Evaluates XGBoost Classifier on unseen test data.
    Gracefully returns None if model is None.
    """
    if model is None:
        logger.warning("No XGBoost model provided to evaluate.")
        return None

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


def compare_all_models(
    models_dict: Dict[str, Any],
    X_test: np.ndarray,
    y_test: Union[np.ndarray, list],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Evaluates a dictionary of trained models on unseen test data.
    Automatically dispatches to the corresponding evaluator based on model type or dictionary key.
    
    Parameters
    ----------
    models_dict : Dict[str, Any]
        Dictionary mapping model names (e.g., 'Isolation Forest', 'Random Forest', 'XGBoost', 'MLP Neural Network')
        to their trained model objects.
    X_test : np.ndarray
        Test feature matrix.
    y_test : Union[np.ndarray, list]
        Ground-truth labels (0 = normal, 1 = attack).

    Returns
    -------
    comparison_df : pd.DataFrame
        DataFrame summarizing metrics across all evaluated models:
        [Model, Accuracy, Precision, Recall, F1-score, ROC-AUC, False Positive Rate, TP, FP, TN, FN]
    detailed_results : Dict[str, Dict[str, Any]]
        Full evaluation dictionaries including raw predictions, anomaly scores, and confusion matrices.
    """
    y_true = np.array(y_test, dtype=int)
    records = []
    detailed_results: Dict[str, Dict[str, Any]] = {}

    for name, model in models_dict.items():
        if model is None:
            continue

        eval_res = None
        # Dispatch based on model instance or name
        if isinstance(model, IsolationForest) or "isolation" in name.lower():
            eval_res = evaluate_isolation_forest(model, X_test, y_true)
        elif isinstance(model, RandomForestClassifier) or ("random" in name.lower() and "forest" in name.lower()):
            eval_res = evaluate_random_forest(model, X_test, y_true)
        elif (HAS_XGBOOST and isinstance(model, XGBClassifier)) or "xgb" in name.lower():
            eval_res = evaluate_xgboost(model, X_test, y_true)
        elif isinstance(model, MLPClassifier) or "mlp" in name.lower() or "neural" in name.lower():
            eval_res = evaluate_mlp_model(model, X_test, y_true)
        else:
            # Generic fallback: try standard predict and predict_proba
            try:
                y_pred = model.predict(X_test)
                if hasattr(model, "predict_proba"):
                    prob_matrix = model.predict_proba(X_test)
                    scores = prob_matrix[:, 1] if prob_matrix.shape[1] > 1 else prob_matrix[:, 0]
                elif hasattr(model, "decision_function"):
                    scores = model.decision_function(X_test)
                else:
                    scores = y_pred.astype(float)

                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
                fpr = float(fp) / float(fp + tn) if (fp + tn) > 0 else 0.0
                try:
                    auc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else 0.5
                except Exception:
                    auc = 0.5

                eval_res = {
                    "accuracy": float(accuracy_score(y_true, y_pred)),
                    "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                    "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                    "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                    "roc_auc": float(auc),
                    "false_positive_rate": float(fpr),
                    "confusion_matrix": cm.tolist(),
                    "tp": int(tp),
                    "fp": int(fp),
                    "tn": int(tn),
                    "fn": int(fn),
                    "predictions": y_pred.tolist(),
                    "anomaly_scores": scores.tolist() if isinstance(scores, np.ndarray) else list(scores),
                }
            except Exception as e:
                logger.error(f"Failed generic evaluation for model '{name}': {e}")
                continue

        if eval_res is not None:
            detailed_results[name] = eval_res
            records.append({
                "Model": name,
                "Accuracy": eval_res["accuracy"],
                "Precision": eval_res["precision"],
                "Recall": eval_res["recall"],
                "F1-score": eval_res["f1"],
                "ROC-AUC": eval_res["roc_auc"],
                "False Positive Rate": eval_res["false_positive_rate"],
                "TP": eval_res["tp"],
                "FP": eval_res["fp"],
                "TN": eval_res["tn"],
                "FN": eval_res["fn"],
            })

    comparison_df = pd.DataFrame(records)
    return comparison_df, detailed_results


def train_and_evaluate_all_models(
    X_train: np.ndarray,
    y_train: Union[np.ndarray, list],
    X_test: np.ndarray,
    y_test: Union[np.ndarray, list],
    random_state: int = 42,
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Trains all four models (Isolation Forest, Random Forest, XGBoost, MLP)
    and evaluates them systematically on unseen test data without data leakage.

    Returns
    -------
    models : Dict[str, Any]
        Dictionary containing trained model instances.
    comparison_df : pd.DataFrame
        Consolidated metrics table across all four models.
    detailed_results : Dict[str, Dict[str, Any]]
        Full dictionary containing per-model metrics, predictions, and anomaly scores.
    """
    models: Dict[str, Any] = {}

    # 1. Isolation Forest (Unsupervised)
    contam = calibrate_contamination(y_train)
    models["Isolation Forest"] = train_isolation_forest(
        X_train, contamination=contam, random_state=random_state
    )

    # 2. Random Forest (Supervised Ensemble)
    models["Random Forest"] = train_random_forest(
        X_train, y_train, n_estimators=100, random_state=random_state
    )

    # 3. XGBoost (Gradient Boosted Trees)
    if HAS_XGBOOST:
        xgb_model = train_xgboost(
            X_train, y_train, n_estimators=100, random_state=random_state
        )
        if xgb_model is not None:
            models["XGBoost"] = xgb_model

    # 4. MLP Neural Network (Deep/Neural Baseline)
    models["MLP Neural Network"] = train_mlp_model(
        X_train, y_train, random_state=random_state
    )

    # Evaluate all models
    comparison_df, detailed_results = compare_all_models(models, X_test, y_test)
    return models, comparison_df, detailed_results
