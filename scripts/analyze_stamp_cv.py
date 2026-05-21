#!/usr/bin/env python
"""
Analyze existing STAMP crossval outputs (no training).

Examples:
  python analyze_stamp_cv.py --cv_root data/lusc/stamp_crossval/virchow-full/cv_low_mid_high --model_name virchow-full
  python analyze_stamp_cv.py --cv_root data/lusc/stamp_crossval/virchow-full --cv_tag cv_low_mid_high --model_name virchow-full
  python analyze_stamp_cv.py --cv_map data/lusc/stamp_analysis/sweep_auc_summary.csv

Config-driven label/patient/probability columns:
  python analyze_stamp_cv.py --project configs/project_vulvar.yaml --cv_root data/vulvar/stamp_crossval/ctranspath --model_name ctranspath
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

import sys

from wsi_recurrence.clinical import load_project_config
from wsi_recurrence.stamp_io import resolve_cv_dir
from wsi_recurrence.metrics import plot_roc

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _infer_out_dir(cv_root: Path) -> Path:
    parts = cv_root.resolve().parts
    if "stamp_crossval" in parts:
        idx = parts.index("stamp_crossval")
        project_dir = Path(*parts[:idx])
        return project_dir / "stamp_analysis"
    return Path.cwd() / "stamp_analysis"


def _resolve_cv_dir(cv_root: Path, cv_tag: Optional[str]) -> Path:
    return resolve_cv_dir(cv_root, cv_tag)


def _is_prob_series(series: pd.Series) -> bool:
    vals = pd.to_numeric(series, errors="coerce")
    vals = vals.dropna()
    if vals.empty:
        return False
    if not np.isfinite(vals.to_numpy()).all():
        return False
    return bool(((0.0 <= vals) & (vals <= 1.0)).all())


def _infer_prob_col(df: pd.DataFrame, *, label_col: str, explicit_prob_col: str | None) -> str:
    if explicit_prob_col:
        if explicit_prob_col not in df.columns:
            raise ValueError(
                f"Requested prob_col {explicit_prob_col!r} not found in fold CSV; available columns: {sorted(df.columns)}"
            )
        return explicit_prob_col

    default_prob = f"{label_col}_1"
    if default_prob in df.columns:
        return default_prob

    if "pred" in df.columns and _is_prob_series(df["pred"]):
        return "pred"

    raise ValueError(
        "Could not infer probability column for ROC/PR. "
        f"Tried explicit --prob_col, then {default_prob!r}, then numeric 'pred' in [0,1]. "
        f"Available columns: {sorted(df.columns)}"
    )


def _load_fold_predictions(
    csv_path: Path,
    *,
    patient_col: str,
    label_col: str,
    prob_col: str | None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in (patient_col, label_col) if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing required column(s): {missing}. Available: {sorted(df.columns)}")

    chosen_prob_col = _infer_prob_col(df, label_col=label_col, explicit_prob_col=prob_col)

    out = df[[patient_col, label_col, chosen_prob_col]].copy()
    out = out.rename(columns={patient_col: "patient", chosen_prob_col: "pred"})
    out[label_col] = pd.to_numeric(out[label_col], errors="coerce")
    out["pred"] = pd.to_numeric(out["pred"], errors="coerce")
    return out


def load_model_predictions(
    cv_dir: Path,
    *,
    patient_col: str,
    label_col: str,
    prob_col: str | None,
) -> pd.DataFrame:
    all_preds: list[pd.DataFrame] = []
    split_dirs = sorted(cv_dir.glob("split-*"))
    for split_dir in split_dirs:
        pred_file = split_dir / "patient-preds.csv"
        if not pred_file.exists():
            continue
        fold = _load_fold_predictions(
            pred_file,
            patient_col=patient_col,
            label_col=label_col,
            prob_col=prob_col,
        )
        fold["split"] = int(split_dir.name.split("-")[-1])
        all_preds.append(fold)

    if not all_preds:
        return pd.DataFrame()
    return pd.concat(all_preds, ignore_index=True)


def _infer_model_name(row: pd.Series) -> str:
    if "model" in row and isinstance(row["model"], str) and row["model"].strip():
        return row["model"].strip()
    for key in ("config", "cv_root", "cv_dir"):
        if key in row and isinstance(row[key], str) and row[key].strip():
            path = Path(row[key])
            if key == "config":
                return path.parent.name
            return path.parent.name
    raise ValueError("Could not infer model name from row.")


def _load_cv_map(path: Path) -> List[Tuple[str, Path]]:
    df = pd.read_csv(path)
    if "cv_root" in df.columns:
        cv_key = "cv_root"
    elif "cv_dir" in df.columns:
        cv_key = "cv_dir"
    else:
        raise ValueError("cv_map must include a cv_root or cv_dir column.")

    entries = []
    for _, row in df.iterrows():
        cv_root = Path(str(row[cv_key]))
        model_name = _infer_model_name(row)
        entries.append((model_name, cv_root))
    return entries


def _bootstrap_roc(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 200, seed: int = 42):
    rng = np.random.default_rng(seed)
    tprs = []
    base_fpr = np.linspace(0, 1, 101)
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        fpr, tpr, _ = roc_curve(y_true[idx], y_pred[idx])
        tprs.append(np.interp(base_fpr, fpr, tpr))
    return base_fpr, np.percentile(tprs, [2.5, 97.5], axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv_root", type=Path, default=None)
    ap.add_argument("--cv_tag", type=str, default=None)
    ap.add_argument("--model_name", type=str, default=None)
    ap.add_argument("--cv_map", type=Path, default=None)
    ap.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Optional project YAML (e.g. configs/project_vulvar.yaml) to infer label/patient columns.",
    )
    ap.add_argument("--label_col", type=str, default="", help="Ground-truth label column (overrides config inference).")
    ap.add_argument("--patient_col", type=str, default="", help="Patient/id column in fold CSVs (overrides config inference).")
    ap.add_argument(
        "--prob_col",
        type=str,
        default="",
        help="Positive-class probability column in fold CSVs (overrides inference).",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="If provided, write all outputs here and do not infer stamp_analysis.",
    )
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    if args.cv_map:
        entries = _load_cv_map(args.cv_map)
    elif args.cv_root and args.model_name:
        entries = [(args.model_name, args.cv_root)]
    else:
        raise ValueError("Provide --cv_map or both --cv_root and --model_name.")

    run_name = args.model_name or "multi_model"

    base_out_dir = args.out_dir if args.out_dir is not None else _infer_out_dir(entries[0][1])
    base_out_dir.mkdir(parents=True, exist_ok=True)

    project_cfg: Mapping[str, Any] | None = None
    if args.project is not None:
        project_cfg = load_project_config(args.project)

    # -------------------------
    # Config-driven defaults (CLI may override)
    # -------------------------
    if project_cfg is not None:
        columns_cfg = (project_cfg.get("columns", {}) or {}) if isinstance(project_cfg, Mapping) else {}
        crossval_cfg = (project_cfg.get("crossval", {}) or {}) if isinstance(project_cfg, Mapping) else {}
        default_label_col = str(columns_cfg.get("label") or "").strip() or str(crossval_cfg.get("ground_truth_label") or "").strip()
        default_patient_col = str(columns_cfg.get("pred_id") or "").strip() or str(crossval_cfg.get("patient_label") or "").strip()
        if not default_label_col:
            raise SystemExit(
                "Could not infer label_col from project config; pass --label_col or set columns.label / crossval.ground_truth_label."
            )
        if not default_patient_col:
            raise SystemExit(
                "Could not infer patient_col from project config; pass --patient_col or set columns.pred_id / crossval.patient_label."
            )
    else:
        default_label_col = "recur"
        default_patient_col = "patient"

    label_col = args.label_col.strip() or default_label_col
    patient_col = args.patient_col.strip() or default_patient_col
    prob_col = args.prob_col.strip() or None

    all_results: Dict[str, pd.DataFrame] = {}
    for model_name, cv_root in entries:
        try:
            cv_dir = _resolve_cv_dir(cv_root, args.cv_tag)
        except Exception as exc:
            logger.error("%s: %s", model_name, exc)
            continue
        try:
            results_df = load_model_predictions(
                cv_dir,
                patient_col=patient_col,
                label_col=label_col,
                prob_col=prob_col,
            )
        except Exception as exc:
            logger.error(
                "%s: failed to load predictions under %s (patient_col=%s label_col=%s prob_col=%s): %s",
                model_name,
                cv_dir,
                patient_col,
                label_col,
                prob_col or f"{label_col}_1",
                exc,
            )
            continue
        if results_df.empty:
            logger.warning("%s: no predictions found under %s", model_name, cv_dir)
            continue
        all_results[model_name] = results_df
        logger.info("%s: %d predictions loaded from %s", model_name, len(results_df), cv_dir)

    if not all_results:
        example_model = entries[0][0] if entries else "<model>"
        example_root = str(entries[0][1]) if entries else "<cv_root>"
        raise RuntimeError(
            "No valid prediction files found. "
            f"Example: No valid prediction files found for model {example_model} under {example_root}. "
            f"Checked for patient_col={patient_col!r}, label_col={label_col!r}, "
            f"and probability column={prob_col or f'{label_col}_1'!r}."
        )

    # AUC summary
    auc_results = []
    for model_name, predictions_df in all_results.items():
        if ("patient" not in predictions_df.columns) or (label_col not in predictions_df.columns) or ("pred" not in predictions_df.columns):
            logger.warning("%s missing required columns for metrics", model_name)
            continue

        y_true = pd.to_numeric(predictions_df[label_col], errors="coerce")
        y_pred = pd.to_numeric(predictions_df["pred"], errors="coerce")
        keep = y_true.notna() & y_pred.notna()
        y_true = y_true[keep].astype(int)
        y_pred = y_pred[keep].astype(float)

        n_patients = int(predictions_df["patient"].astype("string").nunique(dropna=True))
        n_positive = int((y_true == 1).sum())
        n_negative = int((y_true == 0).sum())

        roc_auc = float("nan")
        pr_auc = float("nan")
        if n_positive > 0 and n_negative > 0:
            roc_auc = float(roc_auc_score(y_true, y_pred))
            pr_auc = float(average_precision_score(y_true, y_pred))
            logger.info("%s ROC AUC = %.4f | PR AUC = %.4f", model_name, roc_auc, pr_auc)
        else:
            logger.warning("%s: cannot compute AUC (need both classes). n_positive=%d n_negative=%d", model_name, n_positive, n_negative)

        auc_results.append(
            {
                "model": model_name,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "n_patients": n_patients,
                "n_positive": n_positive,
                "n_negative": n_negative,
            }
        )

    if not auc_results:
        raise RuntimeError(
            "No valid model results found for AUC summary. "
            f"Checked for patient_col={patient_col!r}, label_col={label_col!r}, "
            f"and probability column={prob_col or f'{label_col}_1'!r}."
        )

    auc_table = pd.DataFrame(auc_results).sort_values("roc_auc", ascending=False, na_position="last")
    auc_path = base_out_dir / f"auc_summary_{run_name}.csv"
    auc_table.to_csv(auc_path, index=False)
    logger.info("Saved AUC summary to %s", auc_path)

    # Combined predictions
    first_model = list(all_results.keys())[0]
    combined_df = all_results[first_model][["patient", label_col, "split", "pred"]].copy()
    combined_df = combined_df.rename(columns={"split": "fold"})
    combined_df = combined_df.rename(columns={"pred": f"pred_{first_model}"})

    for model_name in list(all_results.keys())[1:]:
        model_preds = all_results[model_name][["patient", "split", "pred"]].copy()
        model_preds = model_preds.rename(columns={"split": f"fold_{model_name}"})
        model_preds = model_preds.rename(columns={"pred": f"pred_{model_name}"})
        combined_df = combined_df.merge(model_preds, on="patient", how="left")
        both = combined_df["fold"].notna() & combined_df[f"fold_{model_name}"].notna()
        mismatch = both & (combined_df["fold"] != combined_df[f"fold_{model_name}"])
        if bool(mismatch.any()):
            preview = combined_df.loc[mismatch, ["patient", "fold", f"fold_{model_name}"]].head(10)
            raise RuntimeError(
                f"Fold mismatch while combining model {model_name!r}; all models in a run must use the same splits.\n"
                + preview.to_string(index=False)
            )
        combined_df = combined_df.drop(columns=[f"fold_{model_name}"])

    combined_path = base_out_dir / f"all_predictions_{run_name}.csv"
    combined_df.to_csv(combined_path, index=False)
    logger.info("Saved combined predictions to %s", combined_path)

    if args.no_plots:
        return

    # ROC curves
    n_models = len(all_results)
    n_cols = 4
    n_rows = int(np.ceil(n_models / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for idx, (model_name, predictions_df) in enumerate(all_results.items()):
        ax = axes[idx]
        y_true = predictions_df[label_col].values
        y_pred = predictions_df["pred"].values

        plot_roc(ax, y_true, y_pred, label=model_name)
        ax.set_title(f"{model_name}")
        ax.legend(loc="lower right")

    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    date = pd.Timestamp.now().strftime("%Y-%m-%d")
    roc_path = base_out_dir / f"roc_curves_{run_name}_{date}.png"
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    logger.info("Saved ROC curves to %s", roc_path)


if __name__ == "__main__":
    main()
