"""Tests for model serving and inference logic.

Model Lead: Aman Bhandari (S24CSEU1002)
"""

import pytest
import pandas as pd
from src.mp266.predict import load_model, predict_spoilage


def test_load_model_returns_pipeline():
    model = load_model()
    if model is not None:
        assert hasattr(model, "predict")


def test_predict_spoilage_with_dict():
    payload = {
        "storage_days": 60.0,
        "temp_set_c": 5.0,
        "humidity_pct": 85.0,
        "power_cuts_hours": 10.0,
        "bag_weight_kg": 50.0,
        "door_openings_per_day": 15.0,
        "crop": "potato",
        "chamber": "Chamber 1",
    }
    pred = predict_spoilage(payload)
    assert isinstance(pred, float)
    assert pred >= 0.0


def test_predict_spoilage_with_dataframe():
    df = pd.DataFrame([{
        "storage_days": 40.0,
        "temp_set_c": 6.0,
        "humidity_pct": 80.0,
        "power_cuts_hours": 5.0,
        "bag_weight_kg": 48.0,
        "door_openings_per_day": 12.0,
        "crop": "onion",
        "chamber": "Chamber 2",
    }])
    pred = predict_spoilage(df)
    assert isinstance(pred, float)
    assert pred >= 0.0


def test_predict_spoilage_fallback_on_missing_model(monkeypatch):
    import src.mp266.predict as p
    monkeypatch.setattr(p, "_MODEL", None)
    pred = p.predict_spoilage({"invalid": 123})
    assert pred == 0.0
