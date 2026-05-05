from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class PredictionCohortStats:
    n_patients: int
    n_positive: int
    n_negative: int


def _infer_pred_cols(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    if "pred" in df.columns:
        cols.append("pred")
    cols.extend([c for c in df.columns if str(c).startswith("pred_")])
    return sorted(set(cols))


def _validation_cfg(project_config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    raw = (project_config or {}).get("validation", None)
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("Project config validation must be a mapping (YAML dict).")
    return raw


def _cohort_stats_from_df(df: pd.DataFrame, *, patient_col: str, label_col: str) -> PredictionCohortStats:
    patients = df[patient_col].astype("string")
    if patients.isna().any():
        n_missing = int(patients.isna().sum())
        raise ValueError(f"Found {n_missing} missing values in patient column {patient_col!r}.")

    dup_mask = patients.duplicated(keep=False)
    if bool(dup_mask.any()):
        dups = sorted(patients[dup_mask].dropna().unique().tolist())
        preview = ", ".join(map(str, dups[:10]))
        suffix = "" if len(dups) <= 10 else f", ... (+{len(dups) - 10} more)"
        raise ValueError(f"Duplicate patients found in predictions: {preview}{suffix}")

    labels = pd.to_numeric(df[label_col], errors="coerce")
    if labels.isna().any():
        n_missing = int(labels.isna().sum())
        raise ValueError(f"Found {n_missing} missing/non-numeric values in label column {label_col!r}.")
    bad_vals = sorted(set(labels.unique().tolist()) - {0, 1})
    if bad_vals:
        raise ValueError(f"Label column {label_col!r} must contain only 0/1; found values: {bad_vals}")

    n_patients = int(patients.nunique())
    n_positive = int((labels == 1).sum())
    n_negative = int((labels == 0).sum())
    return PredictionCohortStats(n_patients=n_patients, n_positive=n_positive, n_negative=n_negative)


def validate_predictions_complete(
    predictions_csv: str | Path,
    project_config: Mapping[str, Any],
) -> PredictionCohortStats:
    """
    Validate that `all_predictions_<model>.csv` covers the expected cohort when
    project config contains a `validation:` block.

    If the `validation:` block is missing, this function returns cohort stats
    without enforcing expected counts (backwards compatible).
    """
    path = Path(str(predictions_csv))
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions CSV: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Failed to read predictions CSV: {path}") from exc

    validation = _validation_cfg(project_config)
    columns_cfg = (project_config or {}).get("columns", {}) or {}

    # Backwards compatible: if no validation block is present, do not enforce
    # strict checks; return best-effort cohort stats for logging/outputs.
    if validation is None:
        patient_col = str(columns_cfg.get("pred_id") or "patient")
        label_col = str(columns_cfg.get("label") or "recur")

        if patient_col in df.columns:
            n_patients = int(df[patient_col].astype("string").nunique(dropna=True))
        else:
            n_patients = int(len(df))

        if label_col in df.columns:
            labels = pd.to_numeric(df[label_col], errors="coerce")
            n_positive = int((labels == 1).sum())
            n_negative = int((labels == 0).sum())
        else:
            n_positive, n_negative = 0, 0

        return PredictionCohortStats(n_patients=n_patients, n_positive=n_positive, n_negative=n_negative)

    patient_col = str(validation.get("patient_col") or columns_cfg.get("pred_id") or "patient")
    label_col = str(validation.get("label_col") or columns_cfg.get("label") or "recur")

    if patient_col not in df.columns:
        raise ValueError(
            f"Prediction completeness check failed for {path}: missing patient column {patient_col!r}."
        )
    if label_col not in df.columns:
        raise ValueError(
            f"Prediction completeness check failed for {path}: missing label column {label_col!r}."
        )

    pred_cols = _infer_pred_cols(df)
    if not pred_cols:
        raise ValueError(
            f"Prediction completeness check failed for {path}: missing prediction column "
            "(expected 'pred' or at least one 'pred_*' column)."
        )

    stats = _cohort_stats_from_df(df, patient_col=patient_col, label_col=label_col)

    expected_n_patients = validation.get("expected_n_patients", None)
    expected_n_positive = validation.get("expected_n_positive", None)
    expected_n_negative = validation.get("expected_n_negative", None)

    missing_expected = [
        k
        for k, v in [
            ("validation.expected_n_patients", expected_n_patients),
            ("validation.expected_n_positive", expected_n_positive),
            ("validation.expected_n_negative", expected_n_negative),
        ]
        if v is None
    ]
    if missing_expected:
        raise ValueError(
            "Project config validation block is present but missing required fields: "
            + ", ".join(missing_expected)
        )

    exp_patients = int(expected_n_patients)
    exp_pos = int(expected_n_positive)
    exp_neg = int(expected_n_negative)

    if stats.n_patients != exp_patients:
        raise ValueError(
            f"Prediction completeness check failed for {path}: expected {exp_patients} unique patients, "
            f"found {stats.n_patients}."
        )
    if stats.n_positive != exp_pos:
        raise ValueError(
            f"Prediction completeness check failed for {path}: expected {exp_pos} positives, found {stats.n_positive}."
        )
    if stats.n_negative != exp_neg:
        raise ValueError(
            f"Prediction completeness check failed for {path}: expected {exp_neg} negatives, found {stats.n_negative}."
        )
    if (stats.n_positive + stats.n_negative) != stats.n_patients:
        raise ValueError(
            f"Prediction completeness check failed for {path}: label counts do not sum to patients "
            f"({stats.n_positive}+{stats.n_negative}!={stats.n_patients})."
        )

    return stats
