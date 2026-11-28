"""Chamber-wise performance evaluation module for cold storage lots.

Service Lead: Aayushman Mathpati (S24CSEU1033)
Role: Service Lead (Service Performance & Group Slice Evaluation)
Problem: MP266 - Cold Storage Spoilage (Surat)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def evaluate_chamber_mae(model, later_df: pd.DataFrame, target_col: str = "spoilage_pct") -> tuple[dict, str]:
    """Compute model MAE on later.csv separately for each chamber."""
    mae_by_group = {}
    for chamber in sorted(later_df["chamber"].unique()):
        subset = later_df[later_df["chamber"] == chamber]
        y_true = subset[target_col]
        y_pred = model.predict(subset)
        mae = float(np.round(mean_absolute_error(y_true, y_pred), 4))
        mae_by_group[chamber] = mae

    worst_group = max(mae_by_group, key=mae_by_group.get)
    return mae_by_group, worst_group
