"""
Model Registry, Persistence, and Inference Engine for GridShield AI.
Saves, loads, and registers trained models and their evaluation metadata.
"""

import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np

# Safe PyTorch import
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models", "saved")
DEFAULT_META_DIR = os.path.join(os.path.dirname(__file__), "models", "metadata")


def save_model_registry(
    models_dict: Dict[str, Any],
    metadata: Dict[str, Any],
    models_dir: str = DEFAULT_MODELS_DIR,
    meta_dir: str = DEFAULT_META_DIR,
) -> Dict[str, str]:
    """
    Saves trained models and serialized metadata to disk.
    Returns dictionary mapping model names to saved file paths.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_paths: Dict[str, str] = {}

    for name, model in models_dict.items():
        if model is None:
            continue

        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        filename = f"{safe_name}_{timestamp}.joblib"
        target_path = os.path.join(models_dir, filename)

        try:
            joblib.dump(model, target_path)
            saved_paths[name] = target_path
        except Exception as e:
            # For non-picklable or torch structures
            if HAS_TORCH and hasattr(model, "model") and isinstance(model.model, torch.nn.Module):
                torch_target = os.path.join(models_dir, f"{safe_name}_{timestamp}.pt")
                torch.save(model.model.state_dict(), torch_target)
                saved_paths[name] = torch_target

    # Save summary metadata JSON
    meta_filename = f"registry_meta_{timestamp}.json"
    meta_path = os.path.join(meta_dir, meta_filename)
    serializable_meta = {
        "timestamp": timestamp,
        "models_saved": list(saved_paths.keys()),
        "saved_paths": saved_paths,
        "metrics": metadata.get("metrics", {}),
        "feature_names": metadata.get("feature_names", []),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(serializable_meta, f, indent=2)

    return saved_paths


def load_model(file_path: str) -> Any:
    """Loads a persisted model from disk."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")

    if file_path.endswith(".pt") or file_path.endswith(".pth"):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required to load .pt weights.")
        return torch.load(file_path)
    else:
        return joblib.load(file_path)


def list_saved_models(models_dir: str = DEFAULT_MODELS_DIR) -> List[Dict[str, str]]:
    """Returns a list of saved model files in the registry."""
    if not os.path.exists(models_dir):
        return []

    entries = []
    for f in os.listdir(models_dir):
        if f.endswith(".joblib") or f.endswith(".pt"):
            full_p = os.path.join(models_dir, f)
            entries.append({
                "filename": f,
                "path": full_p,
                "modified": datetime.fromtimestamp(os.path.getmtime(full_p)).strftime("%Y-%m-%d %H:%M:%S"),
            })
    return entries
