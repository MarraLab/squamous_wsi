#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wsi_recurrence.clinical import (
    add_time_to_event_event_columns,
    clinical_features_table_path,
    fusion_enabled,
    infer_merge_columns,
    load_clinical_table,
    load_project_config,
    merge_predictions_with_clinical,
    validate_fusion_config,
)
from wsi_recurrence.fusion import evaluate_fusion_groupkfold


def _infer_pred_col(df: pd.DataFrame) -> str:
    if "pred" in df.columns:
        return "pred"
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    if len(pred_cols) == 1:
        return pred_cols[0]
    raise ValueError("Could not infer prediction column (expected 'pred' or a single 'pred_*').")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", "--predictions", dest="pred_csv", type=Path, required=True, help="OOF predictions CSV from analyze_stamp_cv.py")
    ap.add_argument("--project", type=Path, default=Path("configs/project_lusc.yaml"))
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--pred_id_col", type=str, default="patient")
    ap.add_argument("--label_col", type=str, default="recur")
    ap.add_argument("--pred_col", type=str, default="")
    ap.add_argument("--clinical_id_col", type=str, default="")
    ap.add_argument("--clinical_stage_col", type=str, default="")
    ap.add_argument("--date_surgery_col", type=str, default="")
    ap.add_argument("--date_followup_col", type=str, default="")
    ap.add_argument("--event_col", type=str, default="")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument(
        "--class_weight",
        type=str,
        default="balanced",
        help='LogisticRegression class_weight ("balanced" or "none").',
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cw_raw = str(args.class_weight).strip().lower()
    if cw_raw in ("", "none", "null"):
        class_weight = None
    elif cw_raw == "balanced":
        class_weight = "balanced"
    else:
        raise ValueError('Invalid --class_weight (expected "balanced" or "none").')

    print(f"Using LogisticRegression(C={args.C}, class_weight={class_weight})")

    project_cfg = load_project_config(args.project)
    try:
        run_fusion = fusion_enabled(project_cfg)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not run_fusion:
        raise SystemExit(
            f"Fusion disabled in project config ({args.project}). "
            "Set analysis.run_fusion=true (and configure clinical_path / paths.clinical_features_table) to run fusion."
        )
    validate_fusion_config(project_cfg, project_path=args.project)
    clin_path = clinical_features_table_path(project_cfg)
    defaults = project_cfg.get("clinical", {}) or {}

    pred_df = pd.read_csv(args.pred_csv)
    pred_col = args.pred_col.strip() or _infer_pred_col(pred_df)

    clinical_df = load_clinical_table(clin_path)
    clinical_id_col, clinical_stage_col = infer_merge_columns(
        clinical_df,
        clinical_id_col=args.clinical_id_col.strip() or defaults.get("id_col") or None,
        clinical_stage_col=args.clinical_stage_col.strip() or defaults.get("stage_col") or None,
    )

    date_surgery_col = args.date_surgery_col.strip() or defaults.get("date_surgery_col") or ""
    date_followup_col = args.date_followup_col.strip() or defaults.get("date_followup_col") or ""
    event_col = args.event_col.strip() or defaults.get("event_col") or ""
    event_positive_value = defaults.get("event_positive_value", 1)
    drop_missing_time = bool(defaults.get("drop_missing_time", True))

    merged = merge_predictions_with_clinical(
        pred_df,
        clinical_df,
        pred_id_col=args.pred_id_col,
        pred_col=pred_col,
        label_col=args.label_col,
        clinical_id_col=clinical_id_col,
        clinical_stage_col=clinical_stage_col,
        extra_cols=[c for c in [date_surgery_col, date_followup_col, event_col] if str(c).strip()],
    )
    merged = add_time_to_event_event_columns(
        merged,
        surgery_col=str(date_surgery_col),
        followup_col=str(date_followup_col),
        event_col=str(event_col),
        event_positive_value=event_positive_value,
        label_fallback_col=args.label_col,
    )

    res = evaluate_fusion_groupkfold(
        merged,
        id_col=args.pred_id_col,
        label_col=args.label_col,
        pred_col="pred",
        stage_col="stage_cont",
        n_splits=args.n_splits,
        C=args.C,
        class_weight=class_weight,
        solver="lbfgs",
        max_iter=5000,
    )

    # Optionally drop rows with missing time/event for KM use, without affecting model fit.
    pred_df_out = res.predictions.copy()
    if "time_to_event" in pred_df_out.columns and "event" in pred_df_out.columns:
        if drop_missing_time:
            before = len(pred_df_out)
            pred_df_out = pred_df_out.dropna(subset=["time_to_event", "event"]).copy()
            dropped = before - len(pred_df_out)
            if dropped:
                print(f"WARNING: dropped {dropped}/{before} rows due to missing time_to_event/event")

    pred_out = args.out_dir / "fusion_predictions.csv"
    metrics_out = args.out_dir / "fusion_metrics.csv"
    pred_df_out.to_csv(pred_out, index=False)
    res.metrics.to_csv(metrics_out, index=False)

    print(f"Saved: {pred_out}")
    print(f"Saved: {metrics_out}")


if __name__ == "__main__":
    main()
