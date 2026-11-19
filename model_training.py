"""
Machine Learning Training, Model Calibration, and Comprehensive Evaluation Engine
for GridShield AI Smart Grid Cyberattack Simulator.
Refactored to eliminate redundant code and delegate to dedicated ml_models modules.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
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

from ml_models.random_forest import GridRandomForestModel
from ml_models.xgboost_model import GridXGBoostModel, HAS_XGBOOST
if HAS_XGBOOST:
    from xgboost import XGBClassifier
else:
    XGBClassifier = None
from ml_models.isolation_forest import GridIsolationForestModel
from ml_models.mlp_model import GridMLPModel
from ml_models.lstm_model import GridLSTMModel, HAS_TORCH, PyTorchLSTMModule

logger = logging.getLogger(__name__)


def calibrate_contamination(y_train: Optional[Union[np.ndarray, list]], default: float = 0.05) -> float:
    """
    Calibrates contamination hyperparameter for Isolation Forest.
    Guarantees a valid float strictly within (0.001, 0.45) to satisfy scikit-learn requirements.
    """
    if y_train is None or len(y_train) == 0:
        return default

    y_arr = np.array(y_train)
    positive_count = np.sum(y_arr == 1)
    ratio = float(positive_count) / float(len(y_arr))

    if ratio <= 0.001:
        return 0.01
    return min(max(ratio, 0.01), 0.45)


def train_isolation_forest(
    X_train: np.ndarray,
    contamination: Union[float, str] = 0.08,
    random_state: int = 42,
) -> IsolationForest:
    """
    Trains an Isolation Forest model for unsupervised anomaly detection.
    """
    model_wrapper = GridIsolationForestModel(
        contamination=contamination,
        random_state=random_state,
    )
    model_wrapper.fit(X_train)
    return model_wrapper.model


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
    wrapper = GridIsolationForestModel()
    wrapper.model = model
    wrapper.is_fitted = True
    return wrapper.evaluate(X_test, y_test)


# Compatibility aliases for PyTorch LSTM
GridLSTMAnomalyDetector = GridLSTMModel
PyTorchLSTM = PyTorchLSTMModule


def train_lstm_model(
    X_train: np.ndarray,
    y_train: Union[np.ndarray, list],
    window_size: int = 5,
    epochs: int = 15,
    random_state: int = 42,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> Optional[GridLSTMModel]:
    """
    Trains a temporal PyTorch LSTM neural network for sequential smart grid anomaly detection.
    """
    if not HAS_TORCH:
        return None
    detector = GridLSTMModel(window_size=window_size, epochs=epochs, random_state=random_state)
    detector.fit(X_train, y_train, progress_callback=progress_callback)
    return detector


def evaluate_lstm_model(
    model: GridLSTMModel,
    X_test: np.ndarray,
    y_test: Union[np.ndarray, list],
) -> Dict[str, Any]:
    """
    Evaluates PyTorch LSTM on unseen test data.
    """
    return model.evaluate(X_test, y_test)


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
    wrapper = GridRandomForestModel(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    wrapper.fit(X_train, y_train)
    return wrapper.model


def evaluate_random_forest(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: Union[np.ndarray, list],
) -> Dict[str, Any]:
    """
    Evaluates Random Forest Classifier on unseen test data.
    """
    wrapper = GridRandomForestModel()
    wrapper.model = model
    wrapper.is_fitted = True
    return wrapper.evaluate(X_test, y_test)


def train_xgboost(
    X_train: np.ndarray,
    y_train: Union[np.ndarray, list],
    n_estimators: int = 100,
    max_depth: int = 4,
    learning_rate: float = 0.08,
    random_state: int = 42,
) -> Optional[Any]:
    """
    Trains an XGBoost Classifier for extreme gradient boosted attack classification.
    """
    if not HAS_XGBOOST:
        logger.warning("XGBoost is not installed in the environment.")
        return None
    wrapper = GridXGBoostModel(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    wrapper.fit(X_train, y_train)
    return wrapper.model


def evaluate_xgboost(
    model: Any,
    X_test: np.ndarray,
    y_test: Union[np.ndarray, list],
) -> Dict[str, Any]:
    """
    Evaluates XGBoost model on unseen test data.
    """
    if model is None:
        return {}
    wrapper = GridXGBoostModel()
    wrapper.model = model
    wrapper.is_fitted = True
    return wrapper.evaluate(X_test, y_test)


def train_mlp_model(
    X_train: np.ndarray,
    y_train: Union[np.ndarray, list],
    hidden_layer_sizes: Tuple[int, ...] = (64, 32),
    max_iter: int = 150,
    random_state: int = 42,
) -> MLPClassifier:
    """
    Trains a Multi-Layer Perceptron neural network for smart grid anomaly classification.
    """
    wrapper = GridMLPModel(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=random_state,
    )
    wrapper.fit(X_train, y_train)
    return wrapper.model


def evaluate_mlp_model(
    model: MLPClassifier,
    X_test: np.ndarray,
    y_test: Union[np.ndarray, list],
) -> Dict[str, Any]:
    """
    Evaluates Multi-Layer Perceptron model on unseen test data.
    """
    wrapper = GridMLPModel()
    wrapper.model = model
    wrapper.is_fitted = True
    return wrapper.evaluate(X_test, y_test)


def compare_all_models(
    models_or_evals: Union[Dict[str, Any], Dict[str, Dict[str, Any]]],
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[Union[np.ndarray, list]] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]]:
    """
    Constructs a comparative summary DataFrame across all models.
    Supports two calling patterns:
    1. compare_all_models(evaluations_dict) -> pd.DataFrame
    2. compare_all_models(models_dict, X_test, y_test) -> (pd.DataFrame, detailed_evals)
    """
    if X_test is not None and y_test is not None:
        # Pattern 2: evaluate models on the fly
        detailed: Dict[str, Dict[str, Any]] = {}
        for name, m in models_or_evals.items():
            if m is None:
                continue
            if isinstance(m, IsolationForest):
                detailed[name] = evaluate_isolation_forest(m, X_test, y_test)
            elif isinstance(m, RandomForestClassifier):
                detailed[name] = evaluate_random_forest(m, X_test, y_test)
            elif isinstance(m, MLPClassifier):
                detailed[name] = evaluate_mlp_model(m, X_test, y_test)
            elif HAS_XGBOOST and isinstance(m, XGBClassifier):
                detailed[name] = evaluate_xgboost(m, X_test, y_test)
            elif isinstance(m, GridLSTMModel):
                detailed[name] = evaluate_lstm_model(m, X_test, y_test)
            elif hasattr(m, "evaluate"):
                detailed[name] = m.evaluate(X_test, y_test)

        rows = []
        for model_name, metrics in detailed.items():
            rows.append({
                "Model": model_name,
                "Accuracy": metrics.get("accuracy", 0.0),
                "Precision": metrics.get("precision", 0.0),
                "Recall": metrics.get("recall", 0.0),
                "F1-score": metrics.get("f1", 0.0),
                "ROC-AUC": metrics.get("roc_auc", 0.5),
                "False Positive Rate": metrics.get("false_positive_rate", 0.0),
                "TP": metrics.get("tp", 0),
                "FP": metrics.get("fp", 0),
                "TN": metrics.get("tn", 0),
                "FN": metrics.get("fn", 0),
                "Inference Latency (ms)": metrics.get("inference_time_sec", 0.0) * 1000.0,
            })
        return pd.DataFrame(rows), detailed

    # Pattern 1: passed a precomputed evaluations dict
    rows = []
    for model_name, metrics in models_or_evals.items():
        if metrics:
            rows.append({
                "Model": model_name,
                "Accuracy": metrics.get("accuracy", 0.0),
                "Precision": metrics.get("precision", 0.0),
                "Recall": metrics.get("recall", 0.0),
                "F1-score": metrics.get("f1", 0.0),
                "F1-Score": metrics.get("f1", 0.0),
                "ROC-AUC": metrics.get("roc_auc", 0.5),
                "False Positive Rate": metrics.get("false_positive_rate", 0.0),
                "TP": metrics.get("tp", 0),
                "FP": metrics.get("fp", 0),
                "TN": metrics.get("tn", 0),
                "FN": metrics.get("fn", 0),
                "Inference Latency (ms)": metrics.get("inference_time_sec", 0.0) * 1000.0,
            })
    return pd.DataFrame(rows)


def train_and_evaluate_all_models(
    X_train: np.ndarray,
    y_train: Union[np.ndarray, list],
    X_test: np.ndarray,
    y_test: Union[np.ndarray, list],
    contamination: Optional[float] = None,
    random_state: int = 42,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], pd.DataFrame]:
    """
    Comprehensive pipeline training and evaluating all five architectures:
    1. Isolation Forest (Unsupervised)
    2. Random Forest (Supervised Bagging)
    3. XGBoost (Supervised Gradient Boosting)
    4. MLP Neural Network (Deep Feedforward)
    5. PyTorch LSTM (Deep Temporal RNN)
    """
    models: Dict[str, Any] = {}
    evaluations: Dict[str, Dict[str, Any]] = {}

    if contamination is None:
        contamination = calibrate_contamination(y_train)

    # 1. Isolation Forest
    if progress_callback:
        progress_callback("Isolation Forest", 0.1)
    if_model = train_isolation_forest(X_train, contamination=contamination, random_state=random_state)
    models["Isolation Forest"] = if_model
    evaluations["Isolation Forest"] = evaluate_isolation_forest(if_model, X_test, y_test)

    # 2. Random Forest
    if progress_callback:
        progress_callback("Random Forest", 0.3)
    rf_model = train_random_forest(X_train, y_train, random_state=random_state)
    models["Random Forest"] = rf_model
    evaluations["Random Forest"] = evaluate_random_forest(rf_model, X_test, y_test)

    # 3. XGBoost
    if HAS_XGBOOST:
        if progress_callback:
            progress_callback("XGBoost", 0.5)
        xgb_model = train_xgboost(X_train, y_train, random_state=random_state)
        if xgb_model is not None:
            models["XGBoost"] = xgb_model
            evaluations["XGBoost"] = evaluate_xgboost(xgb_model, X_test, y_test)

    # 4. MLP Neural Network
    if progress_callback:
        progress_callback("MLP Neural Network", 0.7)
    mlp_model = train_mlp_model(X_train, y_train, random_state=random_state)
    models["MLP Neural Network"] = mlp_model
    evaluations["MLP Neural Network"] = evaluate_mlp_model(mlp_model, X_test, y_test)

    # 5. Temporal PyTorch LSTM
    if HAS_TORCH:
        if progress_callback:
            progress_callback("PyTorch LSTM", 0.85)
        lstm_model = train_lstm_model(X_train, y_train, random_state=random_state)
        if lstm_model is not None:
            models["PyTorch LSTM"] = lstm_model
            evaluations["PyTorch LSTM"] = evaluate_lstm_model(lstm_model, X_test, y_test)

    if progress_callback:
        progress_callback("Complete", 1.0)

    comparison_df = compare_all_models(evaluations)
    return models, comparison_df, evaluations
