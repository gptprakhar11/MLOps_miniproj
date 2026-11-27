"""Model inference logic for MP266 Cold Storage Spoilage prediction.

Service Lead: Aayushman Mathpati (S24CSEU1033)
"""

from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path("models/model.joblib")
_model = None


def load_model():
    """Load and cache the trained champion model from disk."""
    global _model
    if _model is None and MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_spoilage(data: dict) -> float:
    """Generate spoilage percentage prediction for chamber input data."""
    model = load_model()
    if model is None:
        return 0.0

    input_df = pd.DataFrame([data])
    prediction = float(model.predict(input_df)[0])
    return prediction
