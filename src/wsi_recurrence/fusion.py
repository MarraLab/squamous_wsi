from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wsi_recurrence.metrics import compute_auc, compute_pr_auc


@dataclass(frozen=True)
class FusionResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame


def evaluate_fusion_groupkfold(
    df_in: pd.DataFrame,
    *,
    id_col: str = "patient",
    label_col: str = "recur",
    pred_col: str = "pred",
    stage_col: str = "stage_cont",
    n_splits: int = 5,
    C: float = 0.01,
    class_weight: str | None = "balanced",
    solver: str = "lbfgs",
    max_iter: int = 5000,
) -> FusionResult:
    """
    Grouped 5-fold fusion model: LogisticRegression on [pred, stage_cont] with StandardScaler.
    Mirrors the fusion core in CLAM/run_stamp_pipeline.py (no plotting).
    """
    df = df_in.copy()

    needed = [id_col, label_col, pred_col, stage_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Match run_stamp_pipeline.py behavior (it imputed earlier); keep scope minimal here:
    # fill stage_cont median so StandardScaler/LogReg can run without changing feature set.
    if df[stage_col].isna().any():
        df[stage_col] = df[stage_col].fillna(df[stage_col].median())

    df = df.dropna(subset=[pred_col, label_col, id_col]).reset_index(drop=True)

    X = df[[pred_col, stage_col]].copy()
    y = df[label_col].astype(int).values
    groups = df[id_col].values

    # Fusion model: pred + stage_cont (unchanged)
    preprocess_fusion = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), [pred_col, stage_col]),
        ]
    )
    pipe_fusion = Pipeline(
        [
            ("prep", preprocess_fusion),
            (
                "clf",
                LogisticRegression(
                    C=float(C),
                    class_weight=class_weight,
                    solver=str(solver),
                    max_iter=int(max_iter),
                ),
            ),
        ]
    )

    # Clinical-only baseline: stage_cont only
    preprocess_clin = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), [stage_col]),
        ]
    )
    pipe_clin = Pipeline(
        [
            ("prep", preprocess_clin),
            (
                "clf",
                LogisticRegression(
                    C=float(C),
                    class_weight=class_weight,
                    solver=str(solver),
                    max_iter=int(max_iter),
                ),
            ),
        ]
    )

    gkf = GroupKFold(n_splits=n_splits)
    oof_fusion = np.zeros(len(df), dtype=float)
    oof_clin = np.zeros(len(df), dtype=float)
    fold_idx = np.full(len(df), -1, dtype=int)
    fold_aucs_fusion: List[float] = []
    fold_aucs_clin: List[float] = []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups)):
        pipe_fusion.fit(X.iloc[tr], y[tr])
        p_fusion = pipe_fusion.predict_proba(X.iloc[te])[:, 1]
        oof_fusion[te] = p_fusion

        pipe_clin.fit(df[[stage_col]].iloc[tr], y[tr])
        p_clin = pipe_clin.predict_proba(df[[stage_col]].iloc[te])[:, 1]
        oof_clin[te] = p_clin

        fold_idx[te] = fold
        fold_aucs_fusion.append(float(compute_auc(y[te], p_fusion)))
        fold_aucs_clin.append(float(compute_auc(y[te], p_clin)))

    roc_auc_wsi = float(compute_auc(y, df[pred_col].values))
    pr_auc_wsi = float(compute_pr_auc(y, df[pred_col].values))
    roc_auc_clin = float(compute_auc(y, oof_clin))
    pr_auc_clin = float(compute_pr_auc(y, oof_clin))
    roc_auc_fusion = float(compute_auc(y, oof_fusion))
    pr_auc_fusion = float(compute_pr_auc(y, oof_fusion))

    base_cols = [id_col, label_col, pred_col, stage_col]
    for extra in ("time_to_event", "event"):
        if extra in df.columns:
            base_cols.append(extra)
    pred_out = df[base_cols].copy()
    pred_out["clinical_pred"] = oof_clin
    pred_out["fusion_pred"] = oof_fusion
    pred_out["fold"] = fold_idx

    base = {
        "n": int(len(df)),
        "n_pos": int(df[label_col].sum()),
        "n_groups": int(pd.Series(groups).nunique()),
        "n_splits": int(n_splits),
    }
    metrics = pd.DataFrame(
        [
            {
                **base,
                "method": "WSI",
                "roc_auc": roc_auc_wsi,
                "pr_auc": pr_auc_wsi,
            },
            {
                **base,
                "method": "Clinical",
                "roc_auc": roc_auc_clin,
                "pr_auc": pr_auc_clin,
                "fold_auc_mean": float(np.mean(fold_aucs_clin)) if fold_aucs_clin else np.nan,
                "fold_auc_std": float(np.std(fold_aucs_clin)) if fold_aucs_clin else np.nan,
                "fold_aucs": ",".join([f"{x:.6f}" for x in fold_aucs_clin]),
            },
            {
                **base,
                "method": "Fusion",
                "roc_auc": roc_auc_fusion,
                "pr_auc": pr_auc_fusion,
                "fold_auc_mean": float(np.mean(fold_aucs_fusion)) if fold_aucs_fusion else np.nan,
                "fold_auc_std": float(np.std(fold_aucs_fusion)) if fold_aucs_fusion else np.nan,
                "fold_aucs": ",".join([f"{x:.6f}" for x in fold_aucs_fusion]),
            },
        ]
    )

    return FusionResult(predictions=pred_out, metrics=metrics)
