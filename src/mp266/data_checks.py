"""Data quality checks for MP266 Cold Storage Spoilage dataset.

Detects the 7 kinds of data issues described in the Student Handbook:
1. Leaking column
2. Unit mixup
3. Sentinel value
4. Duplicate rows
5. Stuck sensor
6. Target typos
7. Spelling variants
"""

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_dataset(filepath: str):
    with open(filepath, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return rows, fieldnames


def check_leaking_column(train_rows, train_cols, holdout_cols, target_col="spoilage_pct"):
    """Check for columns present in training but absent in holdout that correlate closely with target."""
    extra_cols = [c for c in train_cols if c not in holdout_cols and c != target_col]
    leaking = []
    for col in extra_cols:
        diffs = []
        for r in train_rows:
            try:
                v = float(r[col])
                t = float(r[target_col])
                diffs.append(abs(v - t))
            except (ValueError, TypeError):
                continue
        if diffs:
            mean_diff = sum(diffs) / len(diffs)
            # If the column agrees almost exactly with the target
            if mean_diff < 1.0:
                leaking.append((col, mean_diff))
    if leaking:
        col, mean_diff = leaking[0]
        return {
            "present": True,
            "column": col,
            "mean_abs_diff": mean_diff,
        }
    return {"present": False, "column": None}


def check_unit_mixup(train_rows, group_col="chamber"):
    """Check if any group has values that are 10x, 100x, 1000x different due to unit confusion."""
    numeric_cols = [
        "storage_days",
        "temp_set_c",
        "humidity_pct",
        "power_cuts_hours",
        "bag_weight_kg",
        "door_openings_per_day",
    ]
    groups = sorted(list(set(r[group_col] for r in train_rows)))

    for col in numeric_cols:
        group_means = {}
        for g in groups:
            vals = [
                float(r[col])
                for r in train_rows
                if r[group_col] == g and r[col] != "" and float(r[col]) >= 0
            ]
            if vals:
                group_means[g] = sum(vals) / len(vals)

        # Check ratio between any pair of groups
        for g1 in groups:
            for g2 in groups:
                if g1 != g2 and g2 in group_means and group_means[g2] > 0:
                    ratio = group_means[g1] / group_means[g2]
                    for factor in [10, 12, 60, 100, 1000]:
                        if 0.8 * factor <= ratio <= 1.2 * factor:
                            return {
                                "present": True,
                                "column": col,
                                "group_value": g1,
                                "factor": factor,
                            }
    return {"present": False, "column": None, "group_value": None, "factor": None}


def check_sentinel_value(train_rows):
    """Check for impossible special numbers typed instead of missing values."""
    numeric_cols = [
        "storage_days",
        "temp_set_c",
        "humidity_pct",
        "power_cuts_hours",
        "bag_weight_kg",
        "door_openings_per_day",
    ]
    # Standard sentinels like -1, -999, 999, 9999
    sentinel_candidates = {-1, -999, -99, 999, 9999}
    for col in numeric_cols:
        counts = Counter()
        for r in train_rows:
            try:
                counts[float(r[col])] += 1
            except (ValueError, TypeError):
                continue
        for val, cnt in counts.items():
            if val in sentinel_candidates and cnt > 5:
                # Format value as int if whole number
                int_val = int(val) if val.is_integer() else val
                return {
                    "present": True,
                    "column": col,
                    "value": int_val,
                    "count": cnt,
                }
    return {"present": False, "column": None, "value": None, "count": None}


def check_duplicate_rows(train_rows, train_cols):
    """Check for exact duplicate rows across all columns."""
    seen = set()
    extra_copies = 0
    for r in train_rows:
        t = tuple(r[c] for c in train_cols)
        if t in seen:
            extra_copies += 1
        else:
            seen.add(t)
    return {
        "present": extra_copies > 0,
        "count": extra_copies if extra_copies > 0 else None,
    }


def check_stuck_sensor(train_rows):
    """Check if a sensor repeated the exact same reading across a long run of dates."""
    sorted_rows = sorted(train_rows, key=lambda x: x["date"])
    cols = [
        "bag_weight_kg",
        "temp_set_c",
        "humidity_pct",
        "power_cuts_hours",
        "door_openings_per_day",
        "storage_days",
    ]

    for col in cols:
        current_val = None
        current_dates = []
        longest_dates = []
        longest_val = None

        for r in sorted_rows:
            v = r[col]
            d = r["date"]
            if v == current_val:
                current_dates.append(d)
            else:
                if len(current_dates) > len(longest_dates):
                    longest_dates = current_dates
                    longest_val = current_val
                current_val = v
                current_dates = [d]
        if len(current_dates) > len(longest_dates):
            longest_dates = current_dates
            longest_val = current_val

        # A stuck sensor typically lasts weeks (multiple dates and many rows)
        unique_dates = sorted(list(set(longest_dates)))
        if len(unique_dates) >= 7 and len(longest_dates) > 20:
            return {
                "present": True,
                "column": col,
                "value": longest_val,
                "start_date": unique_dates[0],
                "end_date": unique_dates[-1],
                "rows_affected": len(longest_dates),
            }
    return {"present": False, "column": None, "start_date": None, "end_date": None}


def check_target_typos(train_rows, target_col="spoilage_pct"):
    """Check if any targets were typed with an extra zero (10x too big) leaving a large gap."""
    targets = sorted([float(r[target_col]) for r in train_rows if r[target_col] != ""])
    if not targets:
        return {"present": False, "count": None}

    # Check for large gap in top percentiles
    max_val = targets[-1]
    second_max = targets[-2]
    # Check if max values are ~10x normal max or > 100 for percentages
    typo_count = sum(1 for v in targets if v > 100)
    if typo_count > 0:
        return {"present": True, "count": typo_count}

    # Also check sudden gap ratio > 3x between consecutive sorted targets
    gaps = [
        (targets[i + 1] / targets[i], targets[i], targets[i + 1], len(targets) - i - 1)
        for i in range(len(targets) - 1)
        if targets[i] > 0
    ]
    large_gaps = [g for g in gaps if g[0] >= 5.0 and g[3] <= 20]
    if large_gaps:
        return {"present": True, "count": large_gaps[0][3]}

    return {"present": False, "count": None}


def check_spelling_variants(train_rows, later_rows, text_cols=("crop", "chamber")):
    """Check for inconsistent category spellings in categorical columns."""
    for col in text_cols:
        train_vals = set(r[col] for r in train_rows)
        later_vals = set(r[col] for r in later_rows)
        # Check if train has more distinct variations than later
        if len(train_vals) > len(later_vals):
            return {
                "present": True,
                "column": col,
                "true_levels": len(later_vals),
                "variants_found": sorted(list(train_vals)),
            }
    return {"present": False, "column": None, "true_levels": None}


def run_all_checks(train_path: str, later_path: str, holdout_path: str):
    train_rows, train_cols = load_dataset(train_path)
    later_rows, later_cols = load_dataset(later_path)
    holdout_rows, holdout_cols = load_dataset(holdout_path)

    findings = {
        "leaking_column": check_leaking_column(train_rows, train_cols, holdout_cols),
        "unit_mixup": check_unit_mixup(train_rows),
        "sentinel_value": check_sentinel_value(train_rows),
        "duplicate_rows": check_duplicate_rows(train_rows, train_cols),
        "stuck_sensor": check_stuck_sensor(train_rows),
        "target_typos": check_target_typos(train_rows),
        "spelling_variants": check_spelling_variants(train_rows, later_rows),
    }
    return findings


if __name__ == "__main__":
    train_csv = sys.argv[1] if len(sys.argv) > 1 else "data/raw/train.csv"
    later_csv = sys.argv[2] if len(sys.argv) > 2 else "data/raw/later.csv"
    holdout_csv = (
        sys.argv[3] if len(sys.argv) > 3 else "data/raw/holdout_features.csv"
    )

    if not Path(train_csv).exists():
        # Fallback to data/
        train_csv = "data/train.csv"
        later_csv = "data/later.csv"
        holdout_csv = "data/holdout_features.csv"

    results = run_all_checks(train_csv, later_csv, holdout_csv)

    print("=== DATA QUALITY FINDINGS (7 Checks) ===")
    for issue, res in results.items():
        print(f"\n[{issue}]")
        for k, v in res.items():
            print(f"  {k}: {v}")
