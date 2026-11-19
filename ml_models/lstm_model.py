"""
PyTorch Temporal LSTM Recurrent Neural Network for GridShield AI Smart Grid Intrusion Detection.
"""

import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    HAS_TORCH = False

from ml_models.base_model import BaseGridModel


if HAS_TORCH:
    class PyTorchLSTMModule(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 2, dropout: float = 0.2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, 16),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(16, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            last_step = out[:, -1, :]
            logits = self.fc(last_step)
            return logits.squeeze(-1)
else:
    PyTorchLSTMModule = None


class GridLSTMModel(BaseGridModel):
    """
    Temporal Recurrent Neural Network (LSTM) for Smart Grid Telemetry Sequence Modeling.
    Captures temporal dependencies, rate of change, and multi-step progressive cyberattacks (e.g. Ramping FDI).
    """

    def __init__(
        self,
        window_size: int = 5,
        hidden_dim: int = 32,
        num_layers: int = 2,
        epochs: int = 15,
        batch_size: int = 32,
        lr: float = 0.005,
        dropout: float = 0.2,
        random_state: int = 42,
    ):
        super().__init__(
            name="PyTorch LSTM",
            algorithm_type="Temporal Deep Learning (RNN)",
            description="2-Layer Long Short-Term Memory network evaluating sliding temporal sequences.",
        )
        self.hyperparameters = {
            "window_size": window_size,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "dropout": dropout,
            "random_state": random_state,
        }
        self.epoch_losses: List[float] = []
        self.input_dim: Optional[int] = None
        self.torch_module: Optional[Any] = None

    def get_algorithm_explanation(self) -> Dict[str, Any]:
        return {
            "algorithm": "Long Short-Term Memory (LSTM)",
            "family": "Recurrent Deep Learning Architecture",
            "working_mechanism": (
                "Maintains a persistent cell state regulated by three non-linear gates: Forget Gate (discards "
                "irrelevant history), Input Gate (stores incoming telemetry trends), and Output Gate (projects the "
                "hidden state). Evaluates sliding time-window sequences $X_{t-k:t}$ to capture dynamic temporal correlations."
            ),
            "smart_grid_suitability": (
                "Essential for detecting time-correlated cyber threats such as Ramping FDI attacks, slow load-drift "
                "degradation, and progressive frequency destabilization that single-snapshot models miss."
            ),
            "input_features": ["voltage", "current", "load", "frequency", "roll_mean", "lags (windowed)"],
            "hyperparameters": self.hyperparameters,
        }

    def _create_sequences(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Creates sliding time-window sequences with backward padding to preserve exact row alignment."""
        window_size = self.hyperparameters["window_size"]
        n_samples, n_feats = X.shape
        padded_X = np.pad(X, ((window_size - 1, 0), (0, 0)), mode="edge")
        seqs = np.zeros((n_samples, window_size, n_feats), dtype=np.float32)
        for i in range(n_samples):
            seqs[i] = padded_X[i : i + window_size]
        return seqs, y

    def fit(
        self,
        X_train: np.ndarray,
        y_train: Optional[Union[np.ndarray, list]] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        **kwargs,
    ) -> "GridLSTMModel":
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is not installed in the environment.")
        if y_train is None:
            raise ValueError("LSTM requires labeled target values (y_train).")

        torch.manual_seed(self.hyperparameters["random_state"])
        np.random.seed(self.hyperparameters["random_state"])

        X_arr = np.asarray(X_train, dtype=np.float32)
        y_arr = np.asarray(y_train, dtype=np.float32)
        self.input_dim = X_arr.shape[1]

        seqs, labels = self._create_sequences(X_arr, y_arr)
        t_X = torch.tensor(seqs, dtype=torch.float32)
        t_y = torch.tensor(labels, dtype=torch.float32)

        dataset = TensorDataset(t_X, t_y)
        loader = DataLoader(dataset, batch_size=self.hyperparameters["batch_size"], shuffle=True)

        self.torch_module = PyTorchLSTMModule(
            input_dim=self.input_dim,
            hidden_dim=self.hyperparameters["hidden_dim"],
            num_layers=self.hyperparameters["num_layers"],
            dropout=self.hyperparameters["dropout"],
        )
        self.torch_module.train()

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.torch_module.parameters(), lr=self.hyperparameters["lr"])

        self.epoch_losses = []
        start_t = time.perf_counter()

        epochs = self.hyperparameters["epochs"]
        for ep in range(epochs):
            total_loss = 0.0
            count = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = self.torch_module(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                count += 1
            avg_loss = total_loss / max(count, 1)
            self.epoch_losses.append(float(avg_loss))

            if progress_callback is not None:
                progress_callback(ep + 1, epochs, float(avg_loss))

        self.training_time_sec = time.perf_counter() - start_t
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not HAS_TORCH or self.torch_module is None:
            p = np.full(len(X), 0.5)
            return np.column_stack([1.0 - p, p])

        self.torch_module.eval()
        X_arr = np.asarray(X, dtype=np.float32)
        seqs, _ = self._create_sequences(X_arr)
        t_X = torch.tensor(seqs, dtype=torch.float32)

        with torch.no_grad():
            logits = self.torch_module(t_X)
            probs = torch.sigmoid(logits).cpu().numpy()

        probs = np.clip(probs, 0.0, 1.0)
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def get_loss_curve(self) -> List[float]:
        return list(self.epoch_losses)

    def save(self, filepath: str) -> str:
        if not HAS_TORCH or self.torch_module is None:
            raise RuntimeError("Cannot save uninitialized PyTorch model.")
        payload = {
            "state_dict": self.torch_module.state_dict(),
            "hyperparameters": self.hyperparameters,
            "input_dim": self.input_dim,
            "training_time_sec": self.training_time_sec,
            "epoch_losses": self.epoch_losses,
            "is_fitted": self.is_fitted,
        }
        torch.save(payload, filepath)
        return filepath

    @classmethod
    def load(cls, filepath: str) -> "GridLSTMModel":
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required to load GridLSTMModel.")
        payload = torch.load(filepath, map_location="cpu", weights_only=False)
        instance = cls(**payload.get("hyperparameters", {}))
        instance.input_dim = payload.get("input_dim", 6)
        instance.torch_module = PyTorchLSTMModule(
            input_dim=instance.input_dim,
            hidden_dim=instance.hyperparameters["hidden_dim"],
            num_layers=instance.hyperparameters["num_layers"],
            dropout=instance.hyperparameters["dropout"],
        )
        instance.torch_module.load_state_dict(payload["state_dict"])
        instance.torch_module.eval()
        instance.training_time_sec = payload.get("training_time_sec", 0.0)
        instance.epoch_losses = payload.get("epoch_losses", [])
        instance.is_fitted = payload.get("is_fitted", True)
        return instance
