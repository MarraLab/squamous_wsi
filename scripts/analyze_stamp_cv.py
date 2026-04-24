#!/usr/bin/env python
"""
Analyze existing STAMP crossval outputs (no training).

Examples:
  python analyze_stamp_cv.py --cv_root /projects/.../stamp_crossval/virchow-full/cv_low_mid_high --model_name virchow-full
  python analyze_stamp_cv.py --cv_root /projects/.../stamp_crossval/virchow-full --cv_tag cv_low_mid_high --model_name virchow-full
  python analyze_stamp_cv.py --cv_map /projects/.../stamp_analysis/sweep_auc_summary.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

import sys

from wsi_recurrence.stamp_io import load_patient_predictions, resolve_cv_dir
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


def load_model_predictions(cv_dir: Path) -> pd.DataFrame:
    return load_patient_predictions(cv_dir)


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

    all_results: Dict[str, pd.DataFrame] = {}
    for model_name, cv_root in entries:
        try:
            cv_dir = _resolve_cv_dir(cv_root, args.cv_tag)
        except Exception as exc:
            logger.error("%s: %s", model_name, exc)
            continue
        results_df = load_model_predictions(cv_dir)
        if results_df.empty:
            logger.warning("%s: no predictions found under %s", model_name, cv_dir)
            continue
        all_results[model_name] = results_df
        logger.info("%s: %d predictions loaded from %s", model_name, len(results_df), cv_dir)

    if not all_results:
        raise RuntimeError("No predictions loaded. Check cv paths and files.")

    # AUC summary
    auc_results = []
    for model_name, predictions_df in all_results.items():
        if "recur" in predictions_df.columns and "pred" in predictions_df.columns:
            auc_score = roc_auc_score(predictions_df["recur"], predictions_df["pred"])
            auc_results.append({
                "Model": model_name,
                "AUC": f"{auc_score:.4f}",
                "N Samples": len(predictions_df),
                "N Positive": predictions_df["recur"].sum(),
            })
            logger.info("%s AUC = %.4f", model_name, auc_score)
        else:
            logger.warning("%s missing recur/pred columns", model_name)

    auc_table = pd.DataFrame(auc_results).sort_values("AUC", ascending=False)
    auc_path = base_out_dir / f"auc_summary_{run_name}.csv"
    auc_table.to_csv(auc_path, index=False)
    logger.info("Saved AUC summary to %s", auc_path)

    # Combined predictions
    first_model = list(all_results.keys())[0]
    combined_df = all_results[first_model][["patient", "recur", "pred"]].copy()
    combined_df = combined_df.rename(columns={"pred": f"pred_{first_model}"})

    for model_name in list(all_results.keys())[1:]:
        model_preds = all_results[model_name][["patient", "pred"]].copy()
        model_preds = model_preds.rename(columns={"pred": f"pred_{model_name}"})
        combined_df = combined_df.merge(model_preds, on="patient", how="left")

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
        y_true = predictions_df["recur"].values
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
