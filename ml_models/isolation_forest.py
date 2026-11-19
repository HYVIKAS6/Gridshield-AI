"""
Isolation Forest Implementation for GridShield AI Smart Grid Intrusion Detection.
"""

import time
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from ml_models.base_model import BaseGridModel


class GridIsolationForestModel(BaseGridModel):
    """
    Unsupervised Tree Ensemble for Smart Grid Anomaly Detection.
    Isolates anomalies instead of profiling normal points, eliminating dependency
    on labeled attack instances during training.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: Union[float, str] = 0.08,
        max_samples: Union[int, float, str] = "auto",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        super().__init__(
            name="Isolation Forest",
            algorithm_type="Unsupervised Anomaly Detection",
            description="Isolates anomalies by randomly selecting a feature and split value in tree partitions.",
        )
        if isinstance(contamination, (int, float)):
            bounded_contam = min(max(float(contamination), 0.005), 0.49)
        else:
            bounded_contam = "auto"

        self.hyperparameters = {
            "n_estimators": n_estimators,
            "contamination": bounded_contam,
            "max_samples": max_samples,
            "random_state": random_state,
            "n_jobs": n_jobs,
        }
        self.model = IsolationForest(**self.hyperparameters)

    def get_algorithm_explanation(self) -> Dict[str, Any]:
        return {
            "algorithm": "Isolation Forest (iForest)",
            "family": "Unsupervised Machine Learning",
            "working_mechanism": (
                "Anomalies are few and different, meaning they require fewer random recursive partitions to isolate "
                "in isolation trees (iTrees). Shorter path lengths from the root indicate higher anomaly propensity. "
                "No prior ground-truth labels are required during model fitting."
            ),
            "smart_grid_suitability": (
                "Crucial for zero-day cyberattack detection where historical attack signatures do not exist. "
                "Detects out-of-corridor voltage spikes, severe frequency dips, and sudden load excursions without supervision."
            ),
            "input_features": ["voltage", "current", "load", "frequency", "roll_mean", "lags"],
            "hyperparameters": self.hyperparameters,
        }

    def fit(self, X_train: np.ndarray, y_train: Optional[Union[np.ndarray, list]] = None, **kwargs) -> "GridIsolationForestModel":
        start_t = time.perf_counter()
        # Isolation Forest is unsupervised; fits solely on X_train
        self.model.fit(X_train)
        self.training_time_sec = time.perf_counter() - start_t
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        # If threshold is default (0.5), use standard model decision or calibrated score
        scores = self.predict_proba(X)[:, 1]
        return (scores >= threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba.")
        # Decision function: lower values mean more anomalous
        raw_scores = self.model.decision_function(X)
        # Convert decision scores into normalized anomaly probability [0, 1]
        # Invert so higher score = higher anomaly probability
        normalized_scores = 1.0 / (1.0 + np.exp(raw_scores * 8.0))
        normalized_scores = np.clip(normalized_scores, 0.0, 1.0)
        return np.column_stack([1.0 - normalized_scores, normalized_scores])

    def save(self, filepath: str) -> str:
        payload = {
            "model": self.model,
            "hyperparameters": self.hyperparameters,
            "training_time_sec": self.training_time_sec,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(payload, filepath)
        return filepath

    @classmethod
    def load(cls, filepath: str) -> "GridIsolationForestModel":
        payload = joblib.load(filepath)
        instance = cls(**payload.get("hyperparameters", {}))
        instance.model = payload["model"]
        instance.training_time_sec = payload.get("training_time_sec", 0.0)
        instance.is_fitted = payload.get("is_fitted", True)
        return instance
