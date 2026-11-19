"""
Abstract Base Model Class and Evaluation Protocol for GridShield AI ML Models.
Defines unified interface for algorithm explanation, training, prediction,
evaluation, and serialization.
"""

from abc import ABC, abstractmethod
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class BaseGridModel(ABC):
    """
    Abstract Base Class for all GridShield AI machine learning and deep learning models.
    """

    def __init__(self, name: str, algorithm_type: str, description: str):
        self.name = name
        self.algorithm_type = algorithm_type  # 'Supervised Classical ML', 'Unsupervised ML', 'Deep Learning'
        self.description = description
        self.model: Any = None
        self.hyperparameters: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self.is_fitted: bool = False
        self.training_time_sec: float = 0.0

    @abstractmethod
    def get_algorithm_explanation(self) -> Dict[str, Any]:
        """Returns structured explanation of algorithm, mechanics, and grid relevance."""
        pass

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: Optional[Union[np.ndarray, list]] = None, **kwargs) -> "BaseGridModel":
        """Fits the model on training data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Generates binary predictions (0 = normal, 1 = attack/anomaly)."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns anomaly probabilities or scores (shape: [N, 2] or [N])."""
        pass

    def evaluate(self, X_test: np.ndarray, y_test: Union[np.ndarray, list]) -> Dict[str, Any]:
        """
        Evaluates model on unseen test set and computes standardized metrics.
        No hardcoded values — computed directly from inputs.
        """
        y_true = np.array(y_test, dtype=int)
        start_t = time.perf_counter()
        y_pred = self.predict(X_test)
        inference_t = time.perf_counter() - start_t

        prob_mat = self.predict_proba(X_test)
        if prob_mat.ndim == 2 and prob_mat.shape[1] > 1:
            scores = prob_mat[:, 1]
        else:
            scores = prob_mat.ravel()

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        fpr = float(fp) / float(fp + tn) if (fp + tn) > 0 else 0.0

        try:
            auc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else 0.5
        except Exception:
            auc = 0.5

        return {
            "model_name": self.name,
            "algorithm_type": self.algorithm_type,
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
            "training_time_sec": float(self.training_time_sec),
            "inference_time_sec": float(inference_t),
            "predictions": y_pred.tolist(),
            "anomaly_scores": scores.tolist(),
        }

    @abstractmethod
    def save(self, filepath: str) -> str:
        """Saves model to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, filepath: str) -> "BaseGridModel":
        """Loads model from disk."""
        pass
