"""
Multi-Layer Perceptron (MLP) Neural Network for GridShield AI Smart Grid Intrusion Detection.
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier

from ml_models.base_model import BaseGridModel


class GridMLPModel(BaseGridModel):
    """
    Feedforward Multi-Layer Perceptron (Deep Neural Network) for Smart Grid Attack Detection.
    Maps non-linear sensor cross-correlations into cyberattack probability states.
    """

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (64, 32),
        activation: str = "relu",
        alpha: float = 0.0001,
        learning_rate_init: float = 0.001,
        max_iter: int = 150,
        random_state: int = 42,
        early_stopping: bool = True,
        validation_fraction: float = 0.15,
    ):
        super().__init__(
            name="MLP Neural Network",
            algorithm_type="Deep Learning / Feedforward Neural Network",
            description="Multi-layer artificial neural network with backpropagation and adaptive learning.",
        )
        self.hyperparameters = {
            "hidden_layer_sizes": hidden_layer_sizes,
            "activation": activation,
            "alpha": alpha,
            "learning_rate_init": learning_rate_init,
            "max_iter": max_iter,
            "random_state": random_state,
            "early_stopping": early_stopping,
            "validation_fraction": validation_fraction,
        }
        self.model = MLPClassifier(**self.hyperparameters)

    def get_algorithm_explanation(self) -> Dict[str, Any]:
        return {
            "algorithm": "Multi-Layer Perceptron (MLPClassifier)",
            "family": "Feedforward Neural Network",
            "working_mechanism": (
                "Passes telemetry input vectors through stacked dense layers with non-linear activation functions (ReLU). "
                "Calculates cross-entropy loss against ground truth and updates synaptic weights via backpropagation using the "
                "Adam optimizer. Incorporates early stopping on a held-out validation fraction to prevent overfitting."
            ),
            "smart_grid_suitability": (
                "Excels at capturing continuous non-linear couplings between grid voltage, phase currents, and frequency deviations. "
                "Fast inference latency suitable for real-time substation edge deployment."
            ),
            "input_features": ["voltage", "current", "load", "frequency", "roll_mean", "lags"],
            "hyperparameters": self.hyperparameters,
        }

    def fit(self, X_train: np.ndarray, y_train: Optional[Union[np.ndarray, list]] = None, **kwargs) -> "GridMLPModel":
        if y_train is None:
            raise ValueError("MLP Neural Network requires labeled target values (y_train).")
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

    def get_loss_curve(self) -> List[float]:
        """Returns epoch-wise loss trajectory for visualization."""
        if not self.is_fitted or not hasattr(self.model, "loss_curve_"):
            return []
        return [float(x) for x in self.model.loss_curve_]

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
    def load(cls, filepath: str) -> "GridMLPModel":
        payload = joblib.load(filepath)
        instance = cls(**payload.get("hyperparameters", {}))
        instance.model = payload["model"]
        instance.training_time_sec = payload.get("training_time_sec", 0.0)
        instance.is_fitted = payload.get("is_fitted", True)
        return instance
