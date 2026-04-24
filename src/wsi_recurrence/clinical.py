from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import yaml


def load_project_config(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at {path}, got {type(data).__name__}")
    return data


def clinical_features_table_path(project_cfg: Dict[str, Any]) -> Path:
    paths = project_cfg.get("paths", {}) or {}
    path = paths.get("clinical_features_table")
    if path:
        return Path(str(path))
    legacy = paths.get("clinical_table")
    if legacy:
        return Path(str(legacy))
    raise ValueError("Missing paths.clinical_features_table (or legacy paths.clinical_table) in project config.")


def clinical_defaults(project_cfg: Dict[str, Any]) -> Dict[str, Any]:
    clinical = project_cfg.get("clinical", {}) or {}
    if not isinstance(clinical, dict):
        return {}
    return clinical


def load_clinical_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def infer_merge_columns(
    clinical_df: pd.DataFrame,
    *,
    clinical_id_col: str | None = None,
    clinical_stage_col: str | None = None,
) -> Tuple[str, str]:
    if clinical_id_col is None:
        for cand in ("slide_id", "patient", "final scan name"):
            if cand in clinical_df.columns:
                clinical_id_col = cand
                break
    if clinical_stage_col is None:
        for cand in ("stage_cont", "stage"):
            if cand in clinical_df.columns:
                clinical_stage_col = cand
                break
    if clinical_id_col is None:
        raise ValueError("Could not infer clinical id column; pass via config/CLI.")
    if clinical_stage_col is None:
        raise ValueError("Could not infer clinical stage column; pass via config/CLI.")
    return clinical_id_col, clinical_stage_col


def merge_predictions_with_clinical(
    pred_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    *,
    pred_id_col: str,
    pred_col: str,
    label_col: str,
    clinical_id_col: str,
    clinical_stage_col: str,
    extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    missing = [c for c in (pred_id_col, pred_col, label_col) if c not in pred_df.columns]
    if missing:
        raise ValueError(f"Predictions CSV missing columns: {missing}")

    clin = clinical_df.copy()
    pred = pred_df.copy()

    # Keep core fusion columns plus optional extra columns if present (e.g. dates/event for KM).
    if extra_cols is None:
        extra_cols = [
            "date of surgery",
            "date of recurrence or most recent followup",
            "recurrence (1=yes)",
        ]
    keep_cols = [clinical_id_col, clinical_stage_col] + [c for c in extra_cols if c in clin.columns]
    clin = clin[keep_cols].copy()

    clin = clin.rename(columns={clinical_id_col: pred_id_col, clinical_stage_col: "stage_cont"})
    pred = pred.rename(columns={pred_col: "pred"})

    merged = pred.merge(clin, on=pred_id_col, how="left")

    merged["stage_cont"] = pd.to_numeric(merged["stage_cont"], errors="coerce")
    merged["pred"] = pd.to_numeric(merged["pred"], errors="coerce")
    merged[label_col] = pd.to_numeric(merged[label_col], errors="coerce")

    return merged


def add_time_to_event_event_columns(
    df_in: pd.DataFrame,
    *,
    surgery_col: str = "",
    followup_col: str = "",
    event_col: str = "",
    event_positive_value: object = 1,
    label_fallback_col: str = "recur",
    out_time_col: str = "time_to_event",
    out_event_col: str = "event",
) -> pd.DataFrame:
    """
    Add `time_to_event` (days) and `event` columns for KM plotting.

    - time_to_event = followup_date - surgery_date (days)
    - event = recurrence (1=yes) (numeric), falling back to label column if missing.
    - Leaves missing values as NaN; callers can drop missing rows as needed.
    """
    df = df_in.copy()

    required = [surgery_col, followup_col, event_col]
    required = [c for c in required if str(c).strip()]
    if len(required) < 3:
        print("WARNING: date/time/event columns are unset; skipping time_to_event/event creation.")
        return df

    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"WARNING: missing clinical columns {missing}; skipping time_to_event/event creation.")
        return df

    surg = pd.to_datetime(df[surgery_col], errors="coerce")
    foll = pd.to_datetime(df[followup_col], errors="coerce")
    df[out_time_col] = (foll - surg).dt.days

    raw_event = df[event_col]
    df[out_event_col] = (raw_event == event_positive_value).astype(int)

    if label_fallback_col in df.columns:
        # Only fill where event could not be computed due to missing raw values.
        df[out_event_col] = df[out_event_col].where(raw_event.notna(), pd.to_numeric(df[label_fallback_col], errors="coerce"))

    df[out_time_col] = pd.to_numeric(df[out_time_col], errors="coerce")
    df[out_event_col] = pd.to_numeric(df[out_event_col], errors="coerce")
    return df
