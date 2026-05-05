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
from wsi_recurrence.fusion_config import resolve_fusion_model_params
from wsi_recurrence.validation import validate_predictions_complete


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
    ap.add_argument("--pred_id_col", type=str, default="")
    ap.add_argument("--label_col", type=str, default="")
    ap.add_argument("--pred_col", type=str, default="")
    ap.add_argument("--clinical_id_col", type=str, default="")
    ap.add_argument("--clinical_stage_col", type=str, default="")  # legacy; prefer analysis.clinical_features
    ap.add_argument("--date_surgery_col", type=str, default="")
    ap.add_argument("--date_followup_col", type=str, default="")
    ap.add_argument("--event_col", type=str, default="")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--C", type=float, default=None, help="LogisticRegression C (overrides config analysis.fusion_model.C).")
    ap.add_argument(
        "--class_weight",
        type=str,
        default="",
        help='LogisticRegression class_weight ("balanced" or "none"). Overrides config analysis.fusion_model.class_weight.',
    )
    ap.add_argument("--solver", type=str, default="", help="LogisticRegression solver (overrides config analysis.fusion_model.solver).")
    ap.add_argument("--max_iter", type=int, default=None, help="LogisticRegression max_iter (overrides config analysis.fusion_model.max_iter).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    project_cfg = load_project_config(args.project)

    # -------------------------
    # Config-driven defaults (CLI may override)
    # -------------------------
    label_col = args.label_col.strip() or str((project_cfg.get("columns", {}) or {}).get("label") or "").strip() or str(
        (project_cfg.get("crossval", {}) or {}).get("ground_truth_label") or ""
    ).strip()
    pred_id_col = args.pred_id_col.strip() or str((project_cfg.get("columns", {}) or {}).get("pred_id") or "").strip() or str(
        (project_cfg.get("crossval", {}) or {}).get("patient_label") or ""
    ).strip()
    if not label_col:
        raise SystemExit("Could not infer label column from config; pass --label_col or set columns.label / crossval.ground_truth_label.")
    if not pred_id_col:
        raise SystemExit("Could not infer prediction id column from config; pass --pred_id_col or set columns.pred_id / crossval.patient_label.")

    analysis_cfg = project_cfg.get("analysis", {}) or {}
    clinical_features = analysis_cfg.get("clinical_features", None)
    if clinical_features is None:
        clinical_features = []
    if not isinstance(clinical_features, list):
        raise SystemExit("analysis.clinical_features must be a list (e.g. ['stage_cont']).")
    clinical_features = [str(c).strip() for c in clinical_features if str(c).strip()]
    if not clinical_features:
        raise SystemExit(
            "Fusion is enabled but analysis.clinical_features is empty. "
            "Set analysis.clinical_features (e.g. ['stage_cont']) or disable fusion (analysis.run_fusion=false)."
        )

    try:
        model_params = resolve_fusion_model_params(
            project_cfg=project_cfg,
            cli_C=args.C,
            cli_class_weight=(args.class_weight if str(args.class_weight).strip() else None),
            cli_solver=(args.solver if str(args.solver).strip() else None),
            cli_max_iter=args.max_iter,
        )
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    print(
        "Using LogisticRegression("
        f"C={model_params.C}, class_weight={model_params.class_weight}, solver={model_params.solver}, max_iter={model_params.max_iter}"
        ")"
    )

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

    try:
        _ = validate_predictions_complete(args.pred_csv, project_cfg)
    except Exception as exc:
        raise SystemExit(f"Prediction completeness check failed for {args.pred_csv}:\n{exc}") from exc

    pred_df = pd.read_csv(args.pred_csv)
    pred_col = args.pred_col.strip() or _infer_pred_col(pred_df)

    clinical_df = load_clinical_table(clin_path)
    clinical_id_col, clinical_stage_col = infer_merge_columns(
        clinical_df,
        project_cfg=project_cfg,
        pred_id_col=pred_id_col,
        clinical_id_col=args.clinical_id_col.strip() or defaults.get("id_col") or None,
        clinical_stage_col=args.clinical_stage_col.strip() or defaults.get("stage_col") or None,
    )

    # Survival/time-to-event handling:
    # - Prefer analysis.outcome_time_col / analysis.outcome_event_col when set.
    # - Otherwise fall back to legacy clinical date/event derivation for backward compatibility (LUSC).
    outcome_time_col = str(analysis_cfg.get("outcome_time_col") or "").strip()
    outcome_event_col = str(analysis_cfg.get("outcome_event_col") or "").strip()

    date_surgery_col = args.date_surgery_col.strip() or defaults.get("date_surgery_col") or ""
    date_followup_col = args.date_followup_col.strip() or defaults.get("date_followup_col") or ""
    event_col = args.event_col.strip() or defaults.get("event_col") or ""
    event_positive_value = defaults.get("event_positive_value", 1)
    drop_missing_time = bool(defaults.get("drop_missing_time", True))

    merged = merge_predictions_with_clinical(
        pred_df,
        clinical_df,
        pred_id_col=pred_id_col,
        pred_col=pred_col,
        label_col=label_col,
        clinical_id_col=clinical_id_col,
        clinical_features=clinical_features,
        clinical_stage_col=clinical_stage_col,
        extra_cols=[c for c in [date_surgery_col, date_followup_col, event_col] if str(c).strip()],
    )

    if not outcome_time_col and not outcome_event_col:
        merged = add_time_to_event_event_columns(
            merged,
            surgery_col=str(date_surgery_col),
            followup_col=str(date_followup_col),
            event_col=str(event_col),
            event_positive_value=event_positive_value,
            label_fallback_col=label_col,
        )
    else:
        for c in (outcome_time_col, outcome_event_col):
            if c and c not in merged.columns:
                print(f"WARNING: configured outcome column {c!r} missing from merged table; KM plotting may be skipped.")

    res = evaluate_fusion_groupkfold(
        merged,
        id_col=pred_id_col,
        label_col=label_col,
        pred_col="pred",
        clinical_features=clinical_features,
        n_splits=args.n_splits,
        C=model_params.C,
        class_weight=model_params.class_weight,
        solver=model_params.solver,
        max_iter=model_params.max_iter,
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
