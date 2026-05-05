from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
    clinical_features: list[str] | None = None,
    n_splits: int = 5,
    C: float = 0.01,
    class_weight: str | None = "balanced",
    solver: str = "lbfgs",
    max_iter: int = 5000,
) -> FusionResult:
    """
    Grouped K-fold fusion evaluation:

    - WSI-only: uses `pred_col` directly (no refit)
    - Clinical-only: LogisticRegression on `clinical_features`
    - Fusion: LogisticRegression on [`pred_col`] + `clinical_features`

    Categorical clinical features are one-hot encoded (handle_unknown="ignore").
    Numeric features are scaled with StandardScaler.

    Mirrors the fusion core in CLAM/run_stamp_pipeline.py (no plotting).
    """
    df = df_in.copy()

    if clinical_features is None:
        clinical_features = []
    clinical_features = [str(c) for c in clinical_features if str(c).strip()]

    needed = [id_col, label_col, pred_col, *clinical_features]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=[pred_col, label_col, id_col]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows after dropping NA predictions/labels/ids.")

    X_pred = df[[pred_col]].copy()
    X_clin = df[clinical_features].copy() if clinical_features else pd.DataFrame(index=df.index)
    y = df[label_col].astype(int).values
    groups = df[id_col].values

    def _split_num_cat(frame: pd.DataFrame, cols: list[str]) -> tuple[list[str], list[str]]:
        num_cols: list[str] = []
        cat_cols: list[str] = []
        for c in cols:
            s = frame[c]
            if pd.api.types.is_numeric_dtype(s):
                num_cols.append(c)
            else:
                cat_cols.append(c)
        return num_cols, cat_cols

    clin_num, clin_cat = _split_num_cat(df, clinical_features)
    fusion_num, fusion_cat = _split_num_cat(df, [pred_col, *clinical_features])

    def _make_pipe(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
        transformers = []
        if num_cols:
            transformers.append(("num", StandardScaler(), num_cols))
        if cat_cols:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
        preprocess = ColumnTransformer(transformers=transformers, remainder="drop")
        return Pipeline(
            [
                ("prep", preprocess),
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

    pipe_fusion = _make_pipe(fusion_num, fusion_cat)
    pipe_clin = _make_pipe(clin_num, clin_cat) if clinical_features else None

    gkf = GroupKFold(n_splits=n_splits)
    oof_fusion = np.zeros(len(df), dtype=float)
    oof_clin = np.full(len(df), np.nan, dtype=float)
    fold_idx = np.full(len(df), -1, dtype=int)
    fold_aucs_fusion: List[float] = []
    fold_aucs_clin: List[float] = []

    X_fusion = df[[pred_col, *clinical_features]].copy()

    for fold, (tr, te) in enumerate(gkf.split(X_fusion, y, groups=groups)):
        pipe_fusion.fit(X_fusion.iloc[tr], y[tr])
        p_fusion = pipe_fusion.predict_proba(X_fusion.iloc[te])[:, 1]
        oof_fusion[te] = p_fusion

        if pipe_clin is not None:
            pipe_clin.fit(X_clin.iloc[tr], y[tr])
            p_clin = pipe_clin.predict_proba(X_clin.iloc[te])[:, 1]
            oof_clin[te] = p_clin

        fold_idx[te] = fold
        fold_aucs_fusion.append(float(compute_auc(y[te], p_fusion)))
        if pipe_clin is not None:
            fold_aucs_clin.append(float(compute_auc(y[te], oof_clin[te])))

    roc_auc_wsi = float(compute_auc(y, df[pred_col].values))
    pr_auc_wsi = float(compute_pr_auc(y, df[pred_col].values))
    roc_auc_clin = float(compute_auc(y, oof_clin)) if pipe_clin is not None else float("nan")
    pr_auc_clin = float(compute_pr_auc(y, oof_clin)) if pipe_clin is not None else float("nan")
    roc_auc_fusion = float(compute_auc(y, oof_fusion))
    pr_auc_fusion = float(compute_pr_auc(y, oof_fusion))

    base_cols = [id_col, label_col, pred_col, *clinical_features]
    for extra in ("time_to_event", "event"):
        if extra in df.columns:
            base_cols.append(extra)
    pred_out = df[base_cols].copy()
    if pipe_clin is not None:
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
