"""
XGBoost Classifier Implementation for GridShield AI Smart Grid Intrusion Detection.
"""

import time
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np

try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    xgb = None
    XGBClassifier = None
    HAS_XGBOOST = False

from ml_models.base_model import BaseGridModel


class GridXGBoostModel(BaseGridModel):
    """
    Supervised Extreme Gradient Boosting Classifier utilizing regularized gradient
    boosted decision trees to detect cyber physical attacks in smart grids.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.08,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        super().__init__(
            name="XGBoost",
            algorithm_type="Supervised Gradient Boosting",
            description="Scalable, regularized gradient boosted decision tree ensemble with second-order gradients.",
        )
        self.hyperparameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "eval_metric": "logloss",
        }
        if HAS_XGBOOST:
            self.model = XGBClassifier(**self.hyperparameters)
        else:
            self.model = None

    def get_algorithm_explanation(self) -> Dict[str, Any]:
        return {
            "algorithm": "XGBoost (Extreme Gradient Boosting)",
            "family": "Supervised Machine Learning (Boosting)",
            "working_mechanism": (
                "Iteratively constructs decision trees by optimizing an objective function with second-order "
                "Taylor approximations (gradients and Hessians). Each subsequent tree is trained to correct "
                "the residual errors of the existing ensemble, regularized by L1/L2 penalties to prevent overfitting."
            ),
            "smart_grid_suitability": (
                "Industry standard for tabular telemetry. High detection accuracy for low-magnitude ramping "
                "and stealthy false data injection attacks (FDIA) that exploit subtle inter-sensor residual boundaries."
            ),
            "input_features": ["voltage", "current", "load", "frequency", "roll_mean", "lags"],
            "hyperparameters": self.hyperparameters,
        }

    def fit(self, X_train: np.ndarray, y_train: Optional[Union[np.ndarray, list]] = None, **kwargs) -> "GridXGBoostModel":
        if not HAS_XGBOOST:
            raise RuntimeError("XGBoost library is not installed in the environment.")
        if y_train is None:
            raise ValueError("XGBoost requires labeled target values (y_train).")
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
    def load(cls, filepath: str) -> "GridXGBoostModel":
        payload = joblib.load(filepath)
        instance = cls(**payload.get("hyperparameters", {}))
        instance.model = payload["model"]
        instance.training_time_sec = payload.get("training_time_sec", 0.0)
        instance.is_fitted = payload.get("is_fitted", True)
        return instance
