"""
Random Forest Classifier Implementation for GridShield AI Smart Grid Intrusion Detection.
"""

import time
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ml_models.base_model import BaseGridModel


class GridRandomForestModel(BaseGridModel):
    """
    Supervised Ensemble Bagging Classifier utilizing orthogonal decision trees
    to detect subtle cyberattack patterns and telemetry anomalies in smart grids.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 15,
        min_samples_split: int = 2,
        class_weight: str = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        super().__init__(
            name="Random Forest",
            algorithm_type="Supervised Ensemble Learning",
            description="Ensemble of de-correlated decision trees with feature sub-sampling and bagging.",
        )
        self.hyperparameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "class_weight": class_weight,
            "random_state": random_state,
            "n_jobs": n_jobs,
        }
        self.model = RandomForestClassifier(**self.hyperparameters)

    def get_algorithm_explanation(self) -> Dict[str, Any]:
        return {
            "algorithm": "Random Forest Classifier (Bagging)",
            "family": "Supervised Machine Learning",
            "working_mechanism": (
                "Constructs an ensemble of de-correlated decision trees during training. Each tree is "
                "trained on a bootstrap sample of smart grid telemetry. For each split, a random subset of "
                "features is considered. Final classification is achieved via majority voting or probability averaging."
            ),
            "smart_grid_suitability": (
                "Exceptional resilience against sensor noise, non-linear physical power correlations, and "
                "feature collinearity. Provides native Gini-based feature importances to highlight attacked sensors."
            ),
            "input_features": ["voltage", "current", "load", "frequency", "roll_mean", "lags"],
            "hyperparameters": self.hyperparameters,
        }

    def fit(self, X_train: np.ndarray, y_train: Optional[Union[np.ndarray, list]] = None, **kwargs) -> "GridRandomForestModel":
        if y_train is None:
            raise ValueError("Random Forest requires labeled target values (y_train).")
        y_arr = np.array(y_train, dtype=int)
        start_t = time.perf_counter()
        self.model.fit(X_train, y_arr)
        self.training_time_sec = time.perf_counter() - start_t
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba.")
        probs = self.model.predict_proba(X)
        if probs.shape[1] == 1:
            p = probs[:, 0]
            return np.column_stack([1.0 - p, p])
        return probs

    def get_feature_importances(self, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        if not self.is_fitted:
            return {}
        raw_imp = self.model.feature_importances_
        if feature_names and len(feature_names) == len(raw_imp):
            return {name: float(val) for name, val in zip(feature_names, raw_imp)}
        return {f"feature_{i}": float(val) for i, val in enumerate(raw_imp)}

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
    def load(cls, filepath: str) -> "GridRandomForestModel":
        payload = joblib.load(filepath)
        instance = cls(**payload.get("hyperparameters", {}))
        instance.model = payload["model"]
        instance.training_time_sec = payload.get("training_time_sec", 0.0)
        instance.is_fitted = payload.get("is_fitted", True)
        return instance
