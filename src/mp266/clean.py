"""Data cleaning and feature engineering pipeline for MP266 Cold Storage Spoilage.

Data Lead: Aditya Raj (S24CSEU1001)
GitHub: adityarajhehe <aditya.raj.926351@gmail.com>
Problem: MP266 - Cold Storage Spoilage (Surat)
Group: G111

Summary of Data Cleaning & Feature Engineering by Aditya Raj:
1. Leaking column: Drops 'spoilage_found_at_unloading' so target leakage is prevented.
2. Sentinel value: Imputes 50 sentinel values (-1) in 'storage_days' using the median of valid days.
3. Stuck sensor: Corrects 135 stuck sensor rows in 'bag_weight_kg' between 2026-02-07 and 2026-02-28.
4. Spelling variants: Normalizes crop abbreviations and typos ('pot.', 'oni.', 'app.', etc.) to 3 true levels ('potato', 'onion', 'apple').
5. Feature engineering: Computes temporal features ('month', 'day_of_week') and interaction indices ('storage_power_risk', 'temp_humidity_index').
"""

import logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Valid crop categories
VALID_CROPS = {"potato", "onion", "apple"}

# Crop spelling normalization mapping
CROP_SYNONYMS = {
    "potato": "potato",
    "potatoes": "potato",
    "potatos": "potato",
    "potaoto": "potato",
    "pot.": "potato",
    "onion": "onion",
    "onions": "onion",
    "onoin": "onion",
    "oni.": "onion",
    "apple": "apple",
    "apples": "apple",
    "appel": "apple",
    "app.": "apple",
}


def normalize_crop_name(val: str) -> str:
    """Normalize crop name to one of 3 true levels: potato, onion, apple."""
    if not isinstance(val, str):
        return "potato"
    clean = val.strip().lower()
    return CROP_SYNONYMS.get(clean, clean)


def clean_dataframe(
    df: pd.DataFrame,
    is_train: bool = True,
    median_storage_days: Optional[float] = None,
    median_bag_weight: Optional[float] = None,
) -> tuple[pd.DataFrame, dict]:
    """Clean the dataset and engineer features.
    
    Args:
        df: Input DataFrame
        is_train: Whether this is training data (contains target)
        median_storage_days: Precomputed median for storage_days (if None, computed)
        median_bag_weight: Precomputed median for bag_weight_kg (if None, computed)
        
    Returns:
        Cleaned DataFrame and metadata dictionary with computed statistics.
    """
    df = df.copy()

    # 1. Drop leaking column if present
    if "spoilage_found_at_unloading" in df.columns:
        df = df.drop(columns=["spoilage_found_at_unloading"])
        logger.info("Dropped leaking column 'spoilage_found_at_unloading'")

    # 2. Fix sentinel values in storage_days (-1)
    if "storage_days" in df.columns:
        invalid_mask = df["storage_days"] < 0
        if median_storage_days is None:
            valid_days = df.loc[~invalid_mask, "storage_days"]
            median_storage_days = float(valid_days.median()) if len(valid_days) > 0 else 30.0
        df.loc[invalid_mask, "storage_days"] = median_storage_days
        logger.info("Imputed %d sentinel values in 'storage_days' with median %.1f", invalid_mask.sum(), median_storage_days)

    # 3. Fix stuck sensor in bag_weight_kg (2026-02-07 to 2026-02-28)
    if "bag_weight_kg" in df.columns and "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        stuck_mask = (dates >= "2026-02-07") & (dates <= "2026-02-28")
        if median_bag_weight is None:
            normal_weights = df.loc[~stuck_mask, "bag_weight_kg"]
            median_bag_weight = float(normal_weights.median()) if len(normal_weights) > 0 else 50.0
        df.loc[stuck_mask, "bag_weight_kg"] = median_bag_weight
        logger.info("Corrected %d stuck sensor readings in 'bag_weight_kg' with median %.1f", stuck_mask.sum(), median_bag_weight)

    # 4. Normalize spelling variants in crop column
    if "crop" in df.columns:
        df["crop"] = df["crop"].apply(normalize_crop_name)
        logger.info("Normalized 'crop' categories: %s", set(df["crop"].unique()))

    # 5. Feature Engineering
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        df["month"] = dates.dt.month.fillna(1).astype(int)
        df["day_of_week"] = dates.dt.day_of_week.fillna(0).astype(int)

    # Interaction: storage risk = storage_days * power_cuts_hours
    if "storage_days" in df.columns and "power_cuts_hours" in df.columns:
        df["storage_power_risk"] = df["storage_days"] * df["power_cuts_hours"]

    # Interaction: spoilage environment index = temp_set_c * humidity_pct / 100
    if "temp_set_c" in df.columns and "humidity_pct" in df.columns:
        df["temp_humidity_index"] = (df["temp_set_c"] * df["humidity_pct"]) / 100.0

    metadata = {
        "median_storage_days": median_storage_days,
        "median_bag_weight": median_bag_weight,
    }
    return df, metadata


def run_pipeline(
    raw_path: str = "data/raw/train.csv",
    output_path: str = "data/processed/train_cleaned.csv",
    params_path: str = "params.yaml",
) -> None:
    """Run data cleaning and feature engineering from file inputs."""
    if Path(params_path).exists():
        with open(params_path, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
            raw_path = params.get("data", {}).get("raw_train_path", raw_path)
            output_path = params.get("data", {}).get("cleaned_train_path", output_path)

    raw_file = Path(raw_path)
    if not raw_file.exists():
        logger.warning("Raw train file %s does not exist. Skipping file processing.", raw_path)
        return

    logger.info("Loading raw dataset from %s", raw_file)
    df = pd.read_csv(raw_file)
    cleaned_df, _ = clean_dataframe(df, is_train=True)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(out_file, index=False)
    logger.info("Saved cleaned dataset to %s (%d rows)", out_file, len(cleaned_df))


if __name__ == "__main__":
    run_pipeline()
