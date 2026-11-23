"""Model training pipeline with MLflow tracking for MP266 Cold Storage Spoilage.

Trains regression models on cleaned train.csv, logs parameters and metrics
to MLflow (SQLite backend), and registers the best model with a champion alias.

Usage:
    python -m src.mp266.train
"""

import hashlib
import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from src.mp266.clean import clean_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Configuration -----------------------------------------------------------

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "mp266_spoilage"
REGISTERED_MODEL_NAME = "mp266_spoilage_model"
RAW_TRAIN_PATH = "data/raw/train.csv"

# Features the prediction service accepts (from PROBLEM.md section 5)
NUMERIC_FEATURES = [
    "storage_days",
    "temp_set_c",
    "humidity_pct",
    "power_cuts_hours",
    "bag_weight_kg",
    "door_openings_per_day",
]
CATEGORICAL_FEATURES = ["crop", "chamber"]
TARGET = "spoilage_pct"


def compute_data_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of raw data file."""
    return hashlib.sha256(Path(filepath).read_bytes()).hexdigest()


def load_and_clean_train(raw_path: str = RAW_TRAIN_PATH) -> pd.DataFrame:
    """Load raw train.csv, clean it, and return a DataFrame ready for training."""
    df = pd.read_csv(raw_path)
    cleaned_df, metadata = clean_dataframe(df, is_train=True)
    logger.info("Cleaned training data: %d rows, metadata=%s", len(cleaned_df), metadata)
    return cleaned_df


def build_preprocessor():
    """Build a sklearn ColumnTransformer for the feature set."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def get_model_configs() -> list[dict]:
    """Return 3+ model configurations to try (handbook requirement)."""
    return [
        {
            "name": "ridge_alpha1",
            "model": Ridge(alpha=1.0),
            "params": {"model_type": "Ridge", "alpha": 1.0},
        },
        {
            "name": "rf_100trees",
            "model": RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42
            ),
            "params": {"model_type": "RandomForest", "n_estimators": 100, "max_depth": 10},
        },
        {
            "name": "rf_200trees",
            "model": RandomForestRegressor(
                n_estimators=200, max_depth=15, random_state=42
            ),
            "params": {"model_type": "RandomForest", "n_estimators": 200, "max_depth": 15},
        },
        {
            "name": "gbr_100trees",
            "model": GradientBoostingRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
            ),
            "params": {
                "model_type": "GradientBoosting",
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
            },
        },
    ]


def train_all_models(
    raw_path: str = RAW_TRAIN_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Train multiple models with MLflow tracking and register the best one.

    Returns:
        dict with best_run_id, best_mae, champion_version, and registered_model_name.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Compute data hash of the original raw file
    data_sha256 = compute_data_sha256(raw_path)
    logger.info("data_sha256: %s", data_sha256)

    # Load and clean
    df = load_and_clean_train(raw_path)

    # Only use the service-accepted features + target for training
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    best_mae = float("inf")
    best_run_id = None
    best_pipeline = None

    configs = get_model_configs()
    for config in configs:
        with mlflow.start_run(run_name=config["name"]) as run:
            # Log the data hash in every run (handbook section 8)
            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("random_state", random_state)
            for k, v in config["params"].items():
                mlflow.log_param(k, v)

            # Build pipeline: preprocessor + regressor
            preprocessor = build_preprocessor()
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("regressor", config["model"]),
            ])
            pipeline.fit(X_train, y_train)

            # Evaluate
            y_pred_train = pipeline.predict(X_train)
            y_pred_val = pipeline.predict(X_val)
            mae_train = mean_absolute_error(y_train, y_pred_train)
            mae_val = mean_absolute_error(y_val, y_pred_val)

            mlflow.log_metric("mae_train", mae_train)
            mlflow.log_metric("mae_val", mae_val)
            mlflow.log_metric("mae", mae_val)

            logger.info(
                "Run %s (%s): MAE train=%.4f, val=%.4f",
                config["name"], run.info.run_id, mae_train, mae_val,
            )

            if mae_val < best_mae:
                best_mae = mae_val
                best_run_id = run.info.run_id
                best_pipeline = pipeline

    # --- Register the best model ---
    logger.info("Best run: %s with MAE=%.4f", best_run_id, best_mae)

    # Save model artifact with joblib for the FastAPI service
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "model.joblib"
    joblib.dump(best_pipeline, model_path)
    logger.info("Saved champion model to %s", model_path)

    # Log model to MLflow and register it
    with mlflow.start_run(run_id=best_run_id):
        mlflow.sklearn.log_model(
            best_pipeline,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            serialization_format="cloudpickle",
        )

    # Set champion alias on version 1
    client = MlflowClient(tracking_uri=TRACKING_URI)
    # Get the latest version number
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    latest_version = max(int(v.version) for v in versions)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", str(latest_version))
    logger.info(
        "Registered model '%s' version %s with alias 'champion'",
        REGISTERED_MODEL_NAME, latest_version,
    )

    result = {
        "experiment_name": EXPERIMENT_NAME,
        "best_run_id": best_run_id,
        "best_mae": best_mae,
        "registered_model_name": REGISTERED_MODEL_NAME,
        "champion_version": latest_version,
    }
    return result


def update_findings_yaml(result: dict, findings_path: str = "findings.yaml") -> None:
    """Update the tracking section of findings.yaml with MLflow results."""
    path = Path(findings_path)
    if not path.exists():
        logger.warning("findings.yaml not found at %s", findings_path)
        return

    with open(path, "r", encoding="utf-8") as f:
        findings = yaml.safe_load(f)

    findings["tracking"] = {
        "experiment_name": result["experiment_name"],
        "best_run_id": result["best_run_id"],
        "registered_model_name": result["registered_model_name"],
        "champion_version": result["champion_version"],
    }

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(findings, f, default_flow_style=False, sort_keys=False)
    logger.info("Updated tracking section in %s", findings_path)


if __name__ == "__main__":
    # Load params.yaml if present
    params_path = Path("params.yaml")
    raw_path = RAW_TRAIN_PATH
    test_size = 0.2
    random_state = 42

    if params_path.exists():
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        raw_path = params.get("data", {}).get("raw_train_path", raw_path)
        test_size = params.get("model", {}).get("test_size", test_size)
        random_state = params.get("model", {}).get("random_state", random_state)

    result = train_all_models(
        raw_path=raw_path,
        test_size=test_size,
        random_state=random_state,
    )

    # Update both copies of findings.yaml
    update_findings_yaml(result, "findings.yaml")
    update_findings_yaml(result, "reports/findings.yaml")

    print("\n=== Training Complete ===")
    print(f"  Experiment:       {result['experiment_name']}")
    print(f"  Best run ID:      {result['best_run_id']}")
    print(f"  Best MAE (val):   {result['best_mae']:.4f}")
    print(f"  Registered model: {result['registered_model_name']}")
    print(f"  Champion version: {result['champion_version']}")
