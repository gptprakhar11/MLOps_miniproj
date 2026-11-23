"""Pytest tests for MP266 Cold Storage Spoilage project.

Contains 3+ tests including at least one using FastAPI TestClient (handbook section 5).
Tests must NOT depend on the real CSV files (they are not in git).
"""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.mp266.clean import clean_dataframe, normalize_crop_name


# ---------------------------------------------------------------------------
# Helper: create small synthetic test data (no real CSV dependency)
# ---------------------------------------------------------------------------
def make_sample_df():
    """Create a small synthetic DataFrame that mimics train.csv structure."""
    return pd.DataFrame({
        "record_id": ["R001", "R002", "R003", "R004", "R005"],
        "date": ["2026-01-15", "2026-02-10", "2026-02-20", "2026-03-05", "2026-04-10"],
        "storage_days": [45, -1, 90, 120, 60],
        "temp_set_c": [5.0, 7.0, 3.0, 8.0, 6.0],
        "humidity_pct": [85, 90, 78, 92, 88],
        "power_cuts_hours": [10.0, 5.0, 20.0, 2.0, 15.0],
        "bag_weight_kg": [50.0, 42.7, 42.7, 55.0, 48.0],
        "door_openings_per_day": [15, 20, 10, 25, 18],
        "crop": ["potato", "ONION", "pot.", " apple", "oni."],
        "chamber": ["Chamber 1", "Chamber 2", "Chamber 1", "Chamber 3", "Chamber 2"],
        "spoilage_found_at_unloading": [5.2, 8.1, 6.0, 12.3, 7.5],
        "spoilage_pct": [5.0, 8.0, 6.2, 12.0, 7.8],
    })


# ---------------------------------------------------------------------------
# Test 1: Crop spelling normalization
# ---------------------------------------------------------------------------
class TestCropNormalization:
    def test_standard_names(self):
        assert normalize_crop_name("potato") == "potato"
        assert normalize_crop_name("onion") == "onion"
        assert normalize_crop_name("apple") == "apple"

    def test_case_insensitive(self):
        assert normalize_crop_name("POTATO") == "potato"
        assert normalize_crop_name("Onion") == "onion"
        assert normalize_crop_name("APPLE") == "apple"

    def test_whitespace_handling(self):
        assert normalize_crop_name(" potato") == "potato"
        assert normalize_crop_name("onion ") == "onion"
        assert normalize_crop_name(" apple ") == "apple"

    def test_abbreviations(self):
        assert normalize_crop_name("pot.") == "potato"
        assert normalize_crop_name("oni.") == "onion"
        assert normalize_crop_name("app.") == "apple"


# ---------------------------------------------------------------------------
# Test 2: Data cleaning pipeline
# ---------------------------------------------------------------------------
class TestCleanDataframe:
    def test_drops_leaking_column(self):
        df = make_sample_df()
        cleaned, _ = clean_dataframe(df, is_train=True)
        assert "spoilage_found_at_unloading" not in cleaned.columns

    def test_sentinel_value_replaced(self):
        df = make_sample_df()
        cleaned, metadata = clean_dataframe(df, is_train=True)
        # No negative storage_days should remain
        assert (cleaned["storage_days"] < 0).sum() == 0
        # The sentinel row (R002) should have been replaced with the median
        assert metadata["median_storage_days"] is not None

    def test_stuck_sensor_corrected(self):
        df = make_sample_df()
        cleaned, metadata = clean_dataframe(df, is_train=True)
        # Rows in the stuck date range should not all be 42.7
        stuck_rows = cleaned[
            (pd.to_datetime(cleaned["date"]) >= "2026-02-07")
            & (pd.to_datetime(cleaned["date"]) <= "2026-02-28")
        ]
        if len(stuck_rows) > 0:
            assert not all(stuck_rows["bag_weight_kg"] == 42.7)

    def test_crop_normalized_to_three_levels(self):
        df = make_sample_df()
        cleaned, _ = clean_dataframe(df, is_train=True)
        unique_crops = set(cleaned["crop"].unique())
        assert unique_crops <= {"potato", "onion", "apple"}

    def test_feature_engineering_columns_added(self):
        df = make_sample_df()
        cleaned, _ = clean_dataframe(df, is_train=True)
        assert "month" in cleaned.columns
        assert "day_of_week" in cleaned.columns
        assert "storage_power_risk" in cleaned.columns
        assert "temp_humidity_index" in cleaned.columns


# ---------------------------------------------------------------------------
# Test 3: FastAPI service (TestClient) — handbook requires at least one
# ---------------------------------------------------------------------------
class TestAPIService:
    @pytest.fixture(autouse=True)
    def setup_client(self):
        """Import the app here so the test does not fail if model file is missing."""
        from app.main import app
        self.client = TestClient(app)

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"

    def test_predict_valid_request(self):
        payload = {
            "storage_days": 60,
            "temp_set_c": 5.0,
            "humidity_pct": 85,
            "power_cuts_hours": 10.0,
            "bag_weight_kg": 50.0,
            "door_openings_per_day": 15,
            "crop": "potato",
            "chamber": "Chamber 1",
        }
        resp = self.client.post("/predict", json=payload)
        assert resp.status_code == 200
        assert "prediction" in resp.json()

    def test_predict_rejects_temp_above_20(self):
        """Client rule 1: temp_set_c above 20 is impossible -> 422."""
        payload = {
            "storage_days": 60,
            "temp_set_c": 25.0,  # above 20
            "humidity_pct": 85,
            "power_cuts_hours": 10.0,
            "bag_weight_kg": 50.0,
            "door_openings_per_day": 15,
            "crop": "potato",
            "chamber": "Chamber 1",
        }
        resp = self.client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_predict_rejects_negative_power_cuts(self):
        """Client rule 2: power_cuts_hours below 0 is a typing mistake -> 422."""
        payload = {
            "storage_days": 60,
            "temp_set_c": 5.0,
            "humidity_pct": 85,
            "power_cuts_hours": -5.0,  # negative
            "bag_weight_kg": 50.0,
            "door_openings_per_day": 15,
            "crop": "potato",
            "chamber": "Chamber 1",
        }
        resp = self.client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_predict_rejects_missing_field(self):
        """Missing required fields should return 422."""
        payload = {
            "storage_days": 60,
            "temp_set_c": 5.0,
            # missing humidity_pct and other required fields
        }
        resp = self.client.post("/predict", json=payload)
        assert resp.status_code == 422
