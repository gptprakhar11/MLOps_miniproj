"""FastAPI endpoint and client validation tests.

Data Lead: Aditya Raj (S24CSEU1001)
"""

import json
import logging
import pytest
from fastapi.testclient import TestClient
from app.main import app, JSONFormatter


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_payload(client):
    payload = {
        "storage_days": 45.0,
        "temp_set_c": 4.5,
        "humidity_pct": 82.0,
        "power_cuts_hours": 8.0,
        "bag_weight_kg": 50.0,
        "door_openings_per_day": 14.0,
        "crop": "potato",
        "chamber": "Chamber 1",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], float)


def test_predict_rule1_temp_above_20(client):
    """Client Rule 1: temp_set_c > 20 is impossible and must return 422."""
    payload = {
        "storage_days": 45.0,
        "temp_set_c": 22.5,
        "humidity_pct": 82.0,
        "power_cuts_hours": 8.0,
        "bag_weight_kg": 50.0,
        "door_openings_per_day": 14.0,
        "crop": "potato",
        "chamber": "Chamber 1",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rule2_negative_power_cuts(client):
    """Client Rule 2: power_cuts_hours < 0 is a typing mistake and must return 422."""
    payload = {
        "storage_days": 45.0,
        "temp_set_c": 5.0,
        "humidity_pct": 82.0,
        "power_cuts_hours": -2.0,
        "bag_weight_kg": 50.0,
        "door_openings_per_day": 14.0,
        "crop": "potato",
        "chamber": "Chamber 1",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_missing_required_field(client):
    payload = {"storage_days": 45.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_json_logging_format():
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="prediction_served",
        args=(),
        exc_info=None,
    )
    record.prediction_data = {"crop": "potato", "result": 5.4}
    formatted = formatter.format(record)
    parsed = json.loads(formatted)
    assert parsed["level"] == "INFO"
    assert parsed["prediction"]["crop"] == "potato"
    assert parsed["prediction"]["result"] == 5.4
