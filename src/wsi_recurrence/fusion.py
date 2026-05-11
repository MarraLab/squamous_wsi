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


def _fixed_fold_splits(df: pd.DataFrame, *, fold_col: str, id_col: str) -> tuple[list[tuple[int, np.ndarray, np.ndarray]], int]:
    fold_values = pd.to_numeric(df[fold_col], errors="coerce")
    if fold_values.isna().any():
        n_bad = int(fold_values.isna().sum())
        raise ValueError(f"Fold column {fold_col!r} contains {n_bad} missing/non-numeric value(s).")

    patient_folds = pd.DataFrame({id_col: df[id_col].astype(str), fold_col: fold_values.astype(int)})
    n_fold_per_patient = patient_folds.groupby(id_col)[fold_col].nunique()
    bad_patients = n_fold_per_patient[n_fold_per_patient > 1]
    if not bad_patients.empty:
        preview = ", ".join(bad_patients.index.astype(str).tolist()[:10])
        raise ValueError(f"Patients appear in multiple folds in {fold_col!r}: {preview}")

    fold_ids = sorted(fold_values.astype(int).unique().tolist())
    if len(fold_ids) < 2:
        raise ValueError(f"Fold column {fold_col!r} must contain at least two unique folds.")

    splits: list[tuple[int, np.ndarray, np.ndarray]] = []
    fold_array = fold_values.astype(int).to_numpy()
    row_idx = np.arange(len(df))
    for fold_id in fold_ids:
        te = row_idx[fold_array == fold_id]
        tr = row_idx[fold_array != fold_id]
        if len(te) == 0 or len(tr) == 0:
            raise ValueError(f"Fold {fold_id} from {fold_col!r} produced an empty train/test split.")
        splits.append((int(fold_id), tr, te))
    return splits, len(fold_ids)


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
    fold_col: str | None = None,
) -> FusionResult:
    """
    Grouped K-fold fusion evaluation:

    - WSI-only: uses `pred_col` directly (no refit)
    - Clinical-only: LogisticRegression on `clinical_features`
    - Fusion: LogisticRegression on [`pred_col`] + `clinical_features`

    Categorical clinical features are one-hot encoded (handle_unknown="ignore").
    Numeric features are scaled with StandardScaler.

    Mirrors the fusion core in CLAM/run_stamp_pipeline.py (no plotting).
    If `fold_col` is provided, those fold assignments are reused instead of
    creating new GroupKFold splits.
    """
    df = df_in.copy()

    if clinical_features is None:
        clinical_features = []
    clinical_features = [str(c) for c in clinical_features if str(c).strip()]

    needed = [id_col, label_col, pred_col, *clinical_features]
    if fold_col:
        needed.append(fold_col)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    drop_subset = [pred_col, label_col, id_col]
    if fold_col:
        drop_subset.append(fold_col)
    df = df.dropna(subset=drop_subset).reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows after dropping NA predictions/labels/ids.")

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

    if fold_col:
        splits, effective_n_splits = _fixed_fold_splits(df, fold_col=fold_col, id_col=id_col)
        split_source = f"fixed:{fold_col}"
    else:
        gkf = GroupKFold(n_splits=n_splits)
        splits = [
            (fold, tr, te)
            for fold, (tr, te) in enumerate(gkf.split(df[[pred_col, *clinical_features]].copy(), y, groups=groups))
        ]
        effective_n_splits = int(n_splits)
        split_source = "GroupKFold"
    oof_fusion = np.zeros(len(df), dtype=float)
    oof_clin = np.full(len(df), np.nan, dtype=float)
    fold_idx = np.full(len(df), -1, dtype=int)
    fold_aucs_fusion: List[float] = []
    fold_aucs_clin: List[float] = []

    X_fusion = df[[pred_col, *clinical_features]].copy()

    for fold, tr, te in splits:
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
        "n_splits": int(effective_n_splits),
        "split_source": split_source,
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
