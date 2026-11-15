"""
Explainable AI (XAI) Engine for GridShield AI Smart Grid Telemetry.
Provides global feature importance and instance-level SHAP / contribution explanations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Safe SHAP import
try:
    import shap
    HAS_SHAP = True
except ImportError:
    shap = None
    HAS_SHAP = False


def compute_feature_importances(
    models_dict: Dict[str, Any],
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Extracts normalized feature importances across supported tree-based models (Random Forest, XGBoost).
    """
    records = []

    for name, model in models_dict.items():
        if model is None:
            continue

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            # Normalize to sum = 1.0
            total = np.sum(importances)
            norm_imp = importances / total if total > 0 else importances

            for feat, val in zip(feature_names, norm_imp):
                records.append({
                    "Model": name,
                    "Feature": feat,
                    "Importance": round(float(val), 4),
                    "Percentage": round(float(val) * 100.0, 2),
                })

    if not records:
        # Default uniform baseline if no tree models present
        for feat in feature_names:
            records.append({
                "Model": "Baseline Heuristic",
                "Feature": feat,
                "Importance": round(1.0 / len(feature_names), 4),
                "Percentage": round(100.0 / len(feature_names), 2),
            })

    return pd.DataFrame(records)


def explain_single_prediction(
    model: Any,
    sample_vector: np.ndarray,
    baseline_background: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Explains why a specific telemetry observation was flagged as an anomaly or normal.
    Uses SHAP TreeExplainer when feasible, or standardized residual divergence.
    """
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(sample_vector))]

    x_2d = np.asarray(sample_vector).reshape(1, -1)
    explanation: Dict[str, Any] = {
        "method": "Feature Contribution Heuristic",
        "contributions": {},
        "top_contributor": None,
        "summary": "",
    }

    # Attempt Tree SHAP if applicable
    if HAS_SHAP and hasattr(model, "feature_importances_"):
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(x_2d)

            # Handle binary classifier output shape
            if isinstance(shap_values, list) and len(shap_values) > 1:
                vals = shap_values[1][0]
            elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                vals = shap_values[0, :, 1]
            elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 2:
                vals = shap_values[0]
            else:
                vals = np.asarray(shap_values).flatten()

            explanation["method"] = "SHAP TreeExplainer"
            contribs = {feat: round(float(v), 4) for feat, v in zip(feature_names, vals)}
            explanation["contributions"] = contribs

            top_feat = max(contribs.items(), key=lambda item: abs(item[1]))
            explanation["top_contributor"] = top_feat[0]
            direction = "increased" if top_feat[1] > 0 else "reduced"
            explanation["summary"] = (
                f"Feature '{top_feat[0]}' {direction} the attack probability by {abs(top_feat[1]):.4f} log-odds."
            )
            return explanation
        except Exception:
            # Fall back to divergence ranking
            pass

    # Heuristic divergence explanation against background reference
    if baseline_background is not None and len(baseline_background) > 0:
        bg_mean = np.mean(baseline_background, axis=0)
        bg_std = np.std(baseline_background, axis=0) + 1e-4
        z_scores = np.abs((sample_vector - bg_mean) / bg_std)

        contribs = {feat: round(float(z), 3) for feat, z in zip(feature_names, z_scores)}
        explanation["contributions"] = contribs
        top_feat = max(contribs.items(), key=lambda item: item[1])
        explanation["top_contributor"] = top_feat[0]
        explanation["summary"] = (
            f"Sensor '{top_feat[0]}' demonstrated the largest deviation ({top_feat[1]:.2f} standard deviations from baseline)."
        )
    else:
        # Simple magnitude ranking
        contribs = {feat: round(float(abs(v)), 3) for feat, v in zip(feature_names, sample_vector)}
        explanation["contributions"] = contribs
        top_feat = max(contribs.items(), key=lambda item: item[1])
        explanation["top_contributor"] = top_feat[0]
        explanation["summary"] = f"Feature '{top_feat[0]}' has the highest scaled variance."

    return explanation
