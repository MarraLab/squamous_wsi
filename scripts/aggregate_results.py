#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METHOD_ORDER = ["WSI", "Clinical", "Fusion"]


def _find_model_dirs(analysis_dir: Path) -> list[Path]:
    if not analysis_dir.exists():
        raise FileNotFoundError(f"Missing analysis dir: {analysis_dir}")
    model_dirs = [p for p in analysis_dir.iterdir() if p.is_dir()]
    return sorted([p for p in model_dirs if p.name != "model_summary"])


def _load_summary_metrics(model_dir: Path) -> pd.DataFrame | None:
    path = model_dir / "figures" / "summary_metrics.csv"
    if not path.exists():
        print(f"WARNING: missing {path}")
        return None
    df = pd.read_csv(path)
    needed = {"method", "roc_auc", "pr_auc"}
    missing = sorted(needed - set(df.columns))
    if missing:
        print(f"WARNING: {path} missing columns {missing}; skipping")
        return None

    out = df[["method", "roc_auc", "pr_auc"]].copy()
    out["roc_auc"] = pd.to_numeric(out["roc_auc"], errors="coerce")
    out["pr_auc"] = pd.to_numeric(out["pr_auc"], errors="coerce")
    out.insert(0, "model", model_dir.name)
    return out


def _save_combined(df: pd.DataFrame, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

def _plot_fusion_auc_barplot(df: pd.DataFrame, out_path: Path) -> None:
    fusion = df[df["method"] == "Fusion"][["model", "roc_auc"]].dropna()
    fusion = fusion.sort_values("roc_auc", ascending=False)
    if fusion.empty:
        print("WARNING: no Fusion rows found; skipping fusion_auc_barplot.png")
        return

    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(fusion)), 4.5))
    ax.bar(fusion["model"], fusion["roc_auc"])
    ax.set_title("Fusion ROC AUC by model")
    ax.set_xlabel("Model")
    ax.set_ylabel("ROC AUC")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _plot_grouped_auc(
    df: pd.DataFrame,
    *,
    value_col: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    pivot = (
        df.pivot_table(index="model", columns="method", values=value_col, aggfunc="first")
        .reindex(columns=[m for m in METHOD_ORDER if m in df["method"].unique()])
    )
    pivot = pivot.sort_index()
    if pivot.empty:
        print(f"WARNING: no data for {out_path.name}; skipping")
        return

    models = list(pivot.index)
    methods = list(pivot.columns)
    x = np.arange(len(models))
    width = 0.8 / max(1, len(methods))

    fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(models)), 4.8))
    for i, method in enumerate(methods):
        vals = pivot[method].values.astype(float)
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, vals, width, label=method)

    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Run directory, e.g. outputs/runs/<run_id> (under wsi_recurrence/).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    analysis_dir = run_dir / "analysis"

    model_dirs = _find_model_dirs(analysis_dir)
    if not model_dirs:
        raise RuntimeError(f"No model dirs found under: {analysis_dir}")

    rows: list[pd.DataFrame] = []
    for model_dir in model_dirs:
        df = _load_summary_metrics(model_dir)
        if df is not None and not df.empty:
            rows.append(df)

    if not rows:
        raise RuntimeError(f"No summary_metrics.csv files found under: {analysis_dir}")

    combined = pd.concat(rows, ignore_index=True)
    combined = combined[["model", "method", "roc_auc", "pr_auc"]].copy()

    out_dir = analysis_dir / "model_summary"
    _save_combined(combined, out_dir / "combined_metrics.csv")

    _plot_fusion_auc_barplot(combined, out_dir / "fusion_auc_barplot.png")
    _plot_grouped_auc(
        combined,
        value_col="roc_auc",
        title="ROC AUC by model and method",
        ylabel="ROC AUC",
        out_path=out_dir / "method_auc_by_model.png",
    )
    _plot_grouped_auc(
        combined,
        value_col="pr_auc",
        title="PR AUC by model and method",
        ylabel="PR AUC",
        out_path=out_dir / "method_pr_auc_by_model.png",
    )


if __name__ == "__main__":
    main()
