#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wsi_recurrence.clinical import (
    clinical_features_table_path,
    infer_merge_columns,
    load_clinical_table,
    load_project_config,
    merge_predictions_with_clinical,
)


def _infer_pred_col(df: pd.DataFrame) -> str:
    if "pred" in df.columns:
        return "pred"
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    if len(pred_cols) == 1:
        return pred_cols[0]
    raise ValueError("Could not infer prediction column (expected 'pred' or a single 'pred_*').")


def _evaluate_fusion_groupkfold(
    df_in: pd.DataFrame,
    *,
    id_col: str,
    label_col: str,
    pred_col: str,
    stage_col: str,
    n_splits: int,
    C: float,
    class_weight: str | None,
) -> tuple[float, float]:
    df = df_in.copy()
    needed = [id_col, label_col, pred_col, stage_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[stage_col].isna().any():
        df[stage_col] = df[stage_col].fillna(df[stage_col].median())

    df = df.dropna(subset=[pred_col, label_col, id_col]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows after dropping NA predictions/labels/ids.")

    X = df[[pred_col, stage_col]].copy()
    y = df[label_col].astype(int).values
    groups = df[id_col].values

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=float(C),
                    solver="lbfgs",
                    max_iter=5000,
                    class_weight=class_weight,
                ),
            ),
        ]
    )

    gkf = GroupKFold(n_splits=int(n_splits))
    oof = np.zeros(len(df), dtype=float)
    for tr, te in gkf.split(X, y, groups=groups):
        pipe.fit(X.iloc[tr], y[tr])
        oof[te] = pipe.predict_proba(X.iloc[te])[:, 1]

    roc_auc = float(roc_auc_score(y, oof))
    pr_auc = float(average_precision_score(y, oof))
    return roc_auc, pr_auc


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions_csv", type=Path, required=True, help="OOF predictions CSV (e.g. all_predictions_ctranspath.csv)")
    ap.add_argument("--project", type=Path, default=Path("configs/project_lusc.yaml"))
    ap.add_argument("--out_dir", type=Path, required=True)

    ap.add_argument("--pred_id_col", type=str, default="patient")
    ap.add_argument("--label_col", type=str, default="recur")
    ap.add_argument("--pred_col", type=str, default="")
    ap.add_argument("--clinical_id_col", type=str, default="")
    ap.add_argument("--clinical_stage_col", type=str, default="")
    ap.add_argument("--n_splits", type=int, default=5)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    project_cfg = load_project_config(args.project)
    defaults = project_cfg.get("clinical", {}) or {}
    clin_path = clinical_features_table_path(project_cfg)
    clinical_df = load_clinical_table(clin_path)

    pred_df = pd.read_csv(args.predictions_csv)
    pred_col = args.pred_col.strip() or _infer_pred_col(pred_df)

    clinical_id_col, clinical_stage_col = infer_merge_columns(
        clinical_df,
        clinical_id_col=args.clinical_id_col.strip() or defaults.get("id_col") or None,
        clinical_stage_col=args.clinical_stage_col.strip() or defaults.get("stage_col") or None,
    )

    merged = merge_predictions_with_clinical(
        pred_df,
        clinical_df,
        pred_id_col=args.pred_id_col,
        pred_col=pred_col,
        label_col=args.label_col,
        clinical_id_col=clinical_id_col,
        clinical_stage_col=clinical_stage_col,
        extra_cols=[],
    )

    # Define sweep space
    C_values = [0.01, 0.1, 1, 10, 100]
    class_weights: list[str | None] = [None, "balanced"]

    base_df = merged.rename(columns={"pred": "pred", "stage_cont": "stage_cont"}).copy()
    # Basic sanity subset: keep only required columns (plus id for GroupKFold)
    base_df = base_df[[args.pred_id_col, args.label_col, "pred", "stage_cont"]].copy()

    base = {
        "n": int(len(base_df)),
        "n_pos": int(pd.to_numeric(base_df[args.label_col], errors="coerce").fillna(0).astype(int).sum()),
        "n_groups": int(pd.Series(base_df[args.pred_id_col]).nunique()),
        "n_splits": int(args.n_splits),
    }

    rows: list[dict] = []
    for C in C_values:
        for cw in class_weights:
            roc_auc, pr_auc = _evaluate_fusion_groupkfold(
                base_df,
                id_col=args.pred_id_col,
                label_col=args.label_col,
                pred_col="pred",
                stage_col="stage_cont",
                n_splits=args.n_splits,
                C=C,
                class_weight=cw,
            )
            rows.append(
                {
                    **base,
                    "C": float(C),
                    "class_weight": "" if cw is None else str(cw),
                    "roc_auc": float(roc_auc),
                    "pr_auc": float(pr_auc),
                }
            )

    results = pd.DataFrame(rows).sort_values(["roc_auc", "pr_auc"], ascending=False).reset_index(drop=True)
    out_csv = args.out_dir / "fusion_sweep_results.csv"
    results.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    best = results.iloc[0].to_dict()
    print("Best (by roc_auc):")
    print({k: best[k] for k in ["C", "class_weight", "roc_auc", "pr_auc"] if k in best})


if __name__ == "__main__":
    main()

