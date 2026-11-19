"""
Unified ML Models Package for GridShield AI.
Exposes standardized model classes, factories, and persistence helpers.
"""

from ml_models.base_model import BaseGridModel
from ml_models.random_forest import GridRandomForestModel
from ml_models.xgboost_model import GridXGBoostModel, HAS_XGBOOST
from ml_models.isolation_forest import GridIsolationForestModel
from ml_models.mlp_model import GridMLPModel
from ml_models.lstm_model import GridLSTMModel, HAS_TORCH

AVAILABLE_MODELS = {
    "random_forest": GridRandomForestModel,
    "isolation_forest": GridIsolationForestModel,
    "mlp": GridMLPModel,
    "lstm": GridLSTMModel,
}

if HAS_XGBOOST:
    AVAILABLE_MODELS["xgboost"] = GridXGBoostModel

__all__ = [
    "BaseGridModel",
    "GridRandomForestModel",
    "GridXGBoostModel",
    "GridIsolationForestModel",
    "GridMLPModel",
    "GridLSTMModel",
    "AVAILABLE_MODELS",
    "HAS_XGBOOST",
    "HAS_TORCH",
]
