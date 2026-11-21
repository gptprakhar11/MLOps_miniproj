"""Unit tests for the data cleaning pipeline.

Author: Aditya Raj (S24CSEU1001) - Data Lead
"""

import pandas as pd
from src.mp266.clean import clean_dataframe, normalize_crop_name


def test_normalize_crop_name():
    assert normalize_crop_name("Potato") == "potato"
    assert normalize_crop_name("potatoes ") == "potato"
    assert normalize_crop_name("ONIONS") == "onion"
    assert normalize_crop_name("apple") == "apple"


def test_clean_dataframe_drops_leaking_column():
    data = {
        "record_id": ["R1", "R2"],
        "date": ["2026-01-10", "2026-01-11"],
        "storage_days": [10, -1],
        "temp_set_c": [12.0, 14.0],
        "humidity_pct": [85, 90],
        "power_cuts_hours": [2.5, 5.0],
        "bag_weight_kg": [50.0, 48.0],
        "door_openings_per_day": [15, 20],
        "crop": ["Potatoes", "onion"],
        "chamber": ["Chamber 1", "Chamber 2"],
        "spoilage_found_at_unloading": [5.2, 8.4],
        "spoilage_pct": [5.0, 8.0],
    }
    df = pd.DataFrame(data)
    cleaned, meta = clean_dataframe(df, is_train=True)

    # 1. Leaking column must be removed
    assert "spoilage_found_at_unloading" not in cleaned.columns

    # 2. Sentinel value (-1) in storage_days must be replaced
    assert (cleaned["storage_days"] < 0).sum() == 0
    assert cleaned.loc[1, "storage_days"] == 10.0

    # 3. Spelling variants must be cleaned
    assert cleaned.loc[0, "crop"] == "potato"

    # 4. Feature engineering columns must be created
    assert "month" in cleaned.columns
    assert "storage_power_risk" in cleaned.columns
    assert "temp_humidity_index" in cleaned.columns


def test_clean_dataframe_stuck_sensor():
    data = {
        "record_id": ["R1", "R2", "R3"],
        "date": ["2026-01-10", "2026-02-15", "2026-03-01"],
        "storage_days": [10, 20, 30],
        "temp_set_c": [12.0, 14.0, 10.0],
        "humidity_pct": [85, 90, 80],
        "power_cuts_hours": [2.0, 4.0, 1.0],
        "bag_weight_kg": [45.0, 999.0, 55.0],
        "crop": ["apple", "potato", "onion"],
        "chamber": ["Chamber 1", "Chamber 1", "Chamber 2"],
    }
    df = pd.DataFrame(data)
    cleaned, _ = clean_dataframe(df, is_train=True)
    # The stuck reading on 2026-02-15 should be corrected
    assert cleaned.loc[1, "bag_weight_kg"] == 50.0
