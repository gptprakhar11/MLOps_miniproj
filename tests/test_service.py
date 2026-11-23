"""Comprehensive unit tests for the MP266 FastAPI prediction service.

Lead Author: Aayushman Mathpati (Service Lead, S24CSEU1033)
Tests:
    - Health check endpoint (GET /health)
    - Prediction endpoint (POST /predict) with valid inputs
    - Client Rule 1: temp_set_c > 20 rejected with HTTP 422
    - Client Rule 1: temp_set_c == 20 boundary accepted
    - Client Rule 2: power_cuts_hours < 0 rejected with HTTP 422
    - Client Rule 2: power_cuts_hours == 0 boundary accepted
    - Missing required fields rejection with HTTP 422
    - Data type validation rejection with HTTP 422
    - All crop categories (potato, onion, apple) handled
    - All chambers (Chamber 1, Chamber 2, Chamber 3) handled
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check_returns_ok_and_status_200():
    """Verify GET /health responds with 200 OK and expected JSON payload."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"


def test_predict_with_valid_potato_payload():
    """Verify POST /predict returns prediction float for a standard potato request."""
    payload = {
        "storage_days": 60.0,
        "temp_set_c": 4.0,
        "humidity_pct": 90.0,
        "power_cuts_hours": 1.5,
        "bag_weight_kg": 50.0,
        "door_openings_per_day": 5.0,
        "crop": "potato",
        "chamber": "Chamber 1",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))


def test_predict_with_valid_onion_payload():
    """Verify POST /predict processes onion lots across Chamber 2."""
    payload = {
        "storage_days": 45.0,
        "temp_set_c": 2.5,
        "humidity_pct": 85.0,
        "power_cuts_hours": 3.0,
        "bag_weight_kg": 45.0,
        "door_openings_per_day": 4.0,
        "crop": "onion",
        "chamber": "Chamber 2",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))


def test_predict_with_valid_apple_payload():
    """Verify POST /predict processes apple lots across Chamber 3."""
    payload = {
        "storage_days": 90.0,
        "temp_set_c": 1.0,
        "humidity_pct": 92.0,
        "power_cuts_hours": 0.5,
        "bag_weight_kg": 20.0,
        "door_openings_per_day": 3.0,
        "crop": "apple",
        "chamber": "Chamber 3",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))


def test_rule1_temp_set_c_above_20_rejected_with_422():
    """Client Rule 1: temp_set_c above 20 is impossible and must return 422."""
    payload = {
        "storage_days": 30.0,
        "temp_set_c": 21.5,  # Exceeds maximum threshold of 20
        "humidity_pct": 88.0,
        "power_cuts_hours": 0.0,
        "bag_weight_kg": 50.0,
        "door_openings_per_day": 4.0,
        "crop": "potato",
        "chamber": "Chamber 1",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_rule1_temp_set_c_exact_boundary_20_accepted():
    """Client Rule 1 boundary: temp_set_c == 20.0 is valid and must be accepted."""
    payload = {
        "storage_days": 30.0,
        "temp_set_c": 20.0,  # Exact boundary
        "humidity_pct": 88.0,
        "power_cuts_hours": 0.0,
        "bag_weight_kg": 50.0,
        "door_openings_per_day": 4.0,
        "crop": "potato",
        "chamber": "Chamber 1",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_rule2_power_cuts_hours_below_zero_rejected_with_422():
    """Client Rule 2: power_cuts_hours below 0 is typing mistake and must return 422."""
    payload = {
        "storage_days": 30.0,
        "temp_set_c": 5.0,
        "humidity_pct": 88.0,
        "power_cuts_hours": -1.5,  # Negative power cuts is invalid
        "bag_weight_kg": 50.0,
        "door_openings_per_day": 4.0,
        "crop": "onion",
        "chamber": "Chamber 2",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_rule2_power_cuts_hours_exact_boundary_zero_accepted():
    """Client Rule 2 boundary: power_cuts_hours == 0.0 is valid and must be accepted."""
    payload = {
        "storage_days": 30.0,
        "temp_set_c": 5.0,
        "humidity_pct": 88.0,
        "power_cuts_hours": 0.0,  # Exact boundary
        "bag_weight_kg": 50.0,
        "door_openings_per_day": 4.0,
        "crop": "onion",
        "chamber": "Chamber 2",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_predict_rejects_missing_required_fields():
    """Verify omitting any required field triggers HTTP 422 validation error."""
    incomplete_payload = {
        "storage_days": 30.0,
        "temp_set_c": 5.0,
        # missing humidity_pct, crop, chamber, etc.
    }
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422


def test_predict_rejects_invalid_data_types():
    """Verify passing string values for numeric fields triggers HTTP 422."""
    invalid_payload = {
        "storage_days": "non_numeric_days",
        "temp_set_c": 5.0,
        "humidity_pct": 85.0,
        "power_cuts_hours": 0.0,
        "bag_weight_kg": 50.0,
        "door_openings_per_day": 4.0,
        "crop": "potato",
        "chamber": "Chamber 1",
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
