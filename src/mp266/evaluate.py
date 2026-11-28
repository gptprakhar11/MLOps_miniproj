"""Stage 4 Pipeline Orchestrator: Drift, Evaluation, Retraining, and Findings.

Ops Lead: Prakhar Gupta (S24CSEU1006)
Role: Ops Lead (Pipeline Orchestration, Continuous Integration & Monitoring)
Problem: MP266 - Cold Storage Spoilage (Surat)
Group: G111
"""

import hashlib
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

from src.mp266.clean import clean_dataframe
from src.mp266.train import (
    CATEGORICAL_FEATURES,
    EXPERIMENT_NAME,
    NUMERIC_FEATURES,
    REGISTERED_MODEL_NAME,
    TARGET,
    TRACKING_URI,
    build_preprocessor,
)

try:
    from src.mp266.chamber_eval import evaluate_chamber_mae
    from src.mp266.drift import compute_feature_drift
    from src.mp266.retrain import (
        generate_holdout_predictions,
        log_and_register_v2,
        retrain_v2_model,
    )
except ImportError:
    # Graceful standalone fallback if modular packages are pending merge
    def compute_psi(expected: np.ndarray, actual: np.ndarray, num_bins: int = 10) -> float:
        quantiles = np.linspace(0, 100, num_bins + 1)
        bins = np.unique(np.percentile(expected, quantiles))
        bins[0] = -np.inf
        bins[-1] = np.inf
        exp_counts, _ = np.histogram(expected, bins=bins)
        act_counts, _ = np.histogram(actual, bins=bins)
        exp_pct = np.where(exp_counts == 0, 1e-4, exp_counts / len(expected))
        act_pct = np.where(act_counts == 0, 1e-4, act_counts / len(actual))
        return float(np.round(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)), 4))

    def compute_feature_drift(train_clean: pd.DataFrame, later_df: pd.DataFrame):
        psi_dict = {col: compute_psi(train_clean[col].values, later_df[col].values) for col in NUMERIC_FEATURES}
        return psi_dict, max(psi_dict, key=psi_dict.get)

    def evaluate_chamber_mae(model, later_df: pd.DataFrame):
        mae_by_group = {ch: float(np.round(mean_absolute_error(sub[TARGET], model.predict(sub)), 4))
                        for ch in sorted(later_df["chamber"].unique())
                        for sub in [later_df[later_df["chamber"] == ch]]}
        return mae_by_group, max(mae_by_group, key=mae_by_group.get)

    def retrain_v2_model(train_clean: pd.DataFrame, later_df: pd.DataFrame, v1_model):
        later_sorted = later_df.sort_values("date").reset_index(drop=True)
        split = int(len(later_sorted) * 0.7)
        tr, chk = later_sorted.iloc[:split], later_sorted.iloc[split:]
        features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        X_tr = pd.concat([train_clean[features], tr[features]], ignore_index=True)
        y_tr = pd.concat([train_clean[TARGET], tr[TARGET]], ignore_index=True)
        v1_mae = float(np.round(mean_absolute_error(chk[TARGET], v1_model.predict(chk[features])), 4))
        v2 = Pipeline([("preprocessor", build_preprocessor()), ("regressor", Ridge(alpha=1.0))]).fit(X_tr, y_tr)
        v2_mae = float(np.round(mean_absolute_error(chk[TARGET], v2.predict(chk[features])), 4))
        decision = "promote" if (v1_mae - v2_mae) / v1_mae >= 0.15 else "keep"
        return v2, v1_mae, v2_mae, decision

    def log_and_register_v2(v2_pipeline, mae_v1, mae_v2, decision):
        import mlflow
        import mlflow.sklearn
        from mlflow import MlflowClient
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name="ridge_v2_retrained") as run:
            mlflow.log_metric("mae_v1_on_check", mae_v1)
            mlflow.log_metric("mae_v2_on_check", mae_v2)
            mlflow.sklearn.log_model(v2_pipeline, "model", registered_model_name=REGISTERED_MODEL_NAME, serialization_format="cloudpickle")
        client = MlflowClient(tracking_uri=TRACKING_URI)
        latest = max(int(v.version) for v in client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'"))
        if decision == "promote":
            client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", str(latest))
            joblib.dump(v2_pipeline, "models/model.joblib")
        return run.info.run_id, latest

    def generate_holdout_predictions(champion_model, holdout_path="data/raw/holdout_features.csv"):
        df = pd.read_csv(holdout_path)
        out = Path("predictions/holdout_predictions.csv")
        out.parent.mkdir(exist_ok=True)
        pd.DataFrame({"record_id": df["record_id"], "prediction": np.round(champion_model.predict(df), 4)}).to_csv(out, index=False)
        return out

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def update_findings(results: dict, findings_paths=("findings.yaml", "reports/findings.yaml")):
    """Update monitoring section of findings.yaml files."""
    for path_str in findings_paths:
        path = Path(path_str)
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            findings = yaml.safe_load(f)

        findings["monitoring"] = {
            "psi": results["psi"],
            "most_drifted_feature": results["most_drifted_feature"],
            "mae_by_group": results["mae_by_group"],
            "worst_served_group": results["worst_served_group"],
            "mae_v1_on_check": results["mae_v1_on_check"],
            "mae_v2_on_check": results["mae_v2_on_check"],
            "decision": results["decision"],
        }

        if results["decision"] == "promote":
            findings["tracking"]["champion_version"] = results["champion_version"]

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(findings, f, default_flow_style=False, sort_keys=False)
        logger.info("Updated %s with Stage 4 monitoring findings", path_str)


def run_pipeline():
    """Execute end-to-end Stage 4 monitoring, retraining, and holdout pipeline."""
    # 1. Load datasets
    train_raw = pd.read_csv("data/raw/train.csv")
    train_clean, _ = clean_dataframe(train_raw)
    later_df = pd.read_csv("data/raw/later.csv")
    v1_model = joblib.load("models/model.joblib")

    # 2. Drift monitoring
    logger.info("Running feature drift analysis...")
    psi_dict, most_drifted = compute_feature_drift(train_clean, later_df)
    logger.info("Most drifted feature: %s (PSI=%.4f)", most_drifted, psi_dict[most_drifted])

    # 3. Chamber-wise slice performance
    logger.info("Evaluating chamber-wise slice performance...")
    mae_by_group, worst_served = evaluate_chamber_mae(v1_model, later_df)
    logger.info("Worst-served chamber: %s (MAE=%.4f)", worst_served, mae_by_group[worst_served])

    # 4. Model retraining & comparison
    logger.info("Retraining model v2 on train + 70%% later.csv...")
    v2_pipeline, mae_v1_check, mae_v2_check, decision = retrain_v2_model(
        train_clean, later_df, v1_model
    )

    # 5. MLflow logging & champion registration
    logger.info("Logging v2 to MLflow and checking promotion rule...")
    run_id, champion_version = log_and_register_v2(v2_pipeline, mae_v1_check, mae_v2_check, decision)

    # 6. Generate holdout predictions with champion model
    champion_model = v2_pipeline if decision == "promote" else v1_model
    pred_path = generate_holdout_predictions(champion_model)

    # 7. Update findings files
    results = {
        "psi": psi_dict,
        "most_drifted_feature": most_drifted,
        "mae_by_group": mae_by_group,
        "worst_served_group": worst_served,
        "mae_v1_on_check": mae_v1_check,
        "mae_v2_on_check": mae_v2_check,
        "decision": decision,
        "champion_version": champion_version,
    }
    update_findings(results)

    print("\n" + "=" * 50)
    print("STAGE 4 PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 50)
    print(f"  Most drifted feature: {most_drifted} (PSI={psi_dict[most_drifted]:.4f})")
    print(f"  Worst served chamber: {worst_served} (MAE={mae_by_group[worst_served]:.4f})")
    print(f"  MAE v1 on check:      {mae_v1_check:.4f}")
    print(f"  MAE v2 on check:      {mae_v2_check:.4f}")
    print(f"  Promotion decision:   {decision}")
    print(f"  Champion version:     {champion_version}")
    print(f"  Holdout predictions:  {pred_path}")
    print("=" * 50)


if __name__ == "__main__":
    run_pipeline()
