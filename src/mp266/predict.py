
"""Model prediction and inference module for MP266 Cold Storage Spoilage.

Model Lead: Aman Bhandari (S24CSEU1002)
Role: Model Lead (Model Serving & Inference Logic)
"""

from pathlib import Path
from typing import Any, Dict, Union
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path("models/model.joblib")
_MODEL = None


def load_model(model_path: Union[str, Path] = MODEL_PATH):
    """Load trained pipeline artifact from disk with caching."""
    global _MODEL
    path = Path(model_path)
    if _MODEL is not None:
        return _MODEL
    if path.exists():
        _MODEL = joblib.load(path)
        return _MODEL
    return None


def predict_spoilage(features: Union[Dict[str, Any], pd.DataFrame]) -> float:
    """Generate spoilage percentage prediction for a storage lot.

    Args:
        features: Dictionary or DataFrame with lot characteristics.

    Returns:
        Predicted spoilage_pct as a float (or 0.0 on error).
    """
    try:
        model = load_model()
        if model is None:
            return 0.0

        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = features.copy()

        prediction = model.predict(df)
        return float(np.round(prediction[0], 4))
    except Exception:
        return 0.0

