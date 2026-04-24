#!/usr/bin/env python

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from wsi_recurrence.metrics import compute_auc, compute_pr_auc, plot_pr, plot_roc


def _try_import_lifelines():
    try:
        from lifelines import KaplanMeierFitter
    except Exception:
        return None
    return KaplanMeierFitter


def _plot_km(out_path: Path, time_s: pd.Series, event_s: pd.Series, risk_s: pd.Series, title: str) -> bool:
    KaplanMeierFitter = _try_import_lifelines()
    if KaplanMeierFitter is None:
        print("WARNING: lifelines not installed; skipping KM plotting.")
        return False

    df = pd.DataFrame(
        {
            "time": pd.to_numeric(time_s, errors="coerce"),
            "event": pd.to_numeric(event_s, errors="coerce"),
            "risk": pd.to_numeric(risk_s, errors="coerce"),
        }
    ).dropna()

    if df.empty:
        print(f"WARNING: no valid rows for KM plot {out_path.name}; skipping.")
        return False

    if df["event"].nunique() < 2:
        print(f"WARNING: event column has <2 unique values for {out_path.name}; skipping.")
        return False

    thr = float(df["risk"].median())
    low = df[df["risk"] < thr]
    high = df[df["risk"] >= thr]
    if low.empty or high.empty:
        print(f"WARNING: median split produced an empty group for {out_path.name}; skipping.")
        return False

    km_low = KaplanMeierFitter()
    km_high = KaplanMeierFitter()

    fig, ax = plt.subplots()
    km_low.fit(low["time"], event_observed=low["event"], label="Low risk").plot_survival_function(ci_show=True, ax=ax)
    km_high.fit(high["time"], event_observed=high["event"], label="High risk").plot_survival_function(ci_show=True, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion_predictions", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fusion_df = pd.read_csv(args.fusion_predictions)

    y_true = fusion_df["recur"].values
    y_wsi = fusion_df["pred"].values
    y_fusion = fusion_df["fusion_pred"].values
    has_clinical = "clinical_pred" in fusion_df.columns
    y_clin = fusion_df["clinical_pred"].values if has_clinical else None

    # -------------------------
    # ROC comparison
    # -------------------------
    fig, ax = plt.subplots()
    plot_roc(
        ax,
        y_true,
        y_wsi,
        label="WSI"
    )

    plot_roc(
        ax,
        y_true,
        y_fusion,
        label="Fusion"
    )
    if has_clinical:
        plot_roc(ax, y_true, y_clin, label="Clinical")

    ax.set_title("ROC: WSI vs Fusion")
    ax.legend()

    out_path = out_dir / "roc_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")

    # -------------------------
    # PR comparison
    # -------------------------
    fig, ax = plt.subplots()
    plot_pr(ax, y_true, y_wsi, label="WSI")
    plot_pr(ax, y_true, y_fusion, label="Fusion")
    if has_clinical:
        plot_pr(ax, y_true, y_clin, label="Clinical")
    ax.set_title("PR: WSI vs Fusion")
    ax.legend()
    pr_path = out_dir / "pr_comparison.png"
    plt.savefig(pr_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pr_path}")

    # -------------------------
    # Summary metrics
    # -------------------------
    rows = [
        {"method": "WSI", "roc_auc": float(compute_auc(y_true, y_wsi)), "pr_auc": float(compute_pr_auc(y_true, y_wsi))},
        {"method": "Fusion", "roc_auc": float(compute_auc(y_true, y_fusion)), "pr_auc": float(compute_pr_auc(y_true, y_fusion))},
    ]
    if has_clinical:
        rows.insert(
            1,
            {
                "method": "Clinical",
                "roc_auc": float(compute_auc(y_true, y_clin)),
                "pr_auc": float(compute_pr_auc(y_true, y_clin)),
            },
        )
    summary = pd.DataFrame(rows)
    summary_path = out_dir / "summary_metrics.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    # -------------------------
    # Optional Kaplan-Meier plots
    # -------------------------
    if "time_to_event" not in fusion_df.columns or "event" not in fusion_df.columns:
        print("WARNING: time_to_event/event not found; skipping KM plotting.")
        return

    time_s = fusion_df["time_to_event"]
    event_s = fusion_df["event"]

    _plot_km(out_dir / "km_wsi.png", time_s, event_s, fusion_df["pred"], title="KM: WSI (median split)")
    if has_clinical:
        _plot_km(out_dir / "km_clinical.png", time_s, event_s, fusion_df["clinical_pred"], title="KM: Clinical (median split)")
    _plot_km(out_dir / "km_fusion.png", time_s, event_s, fusion_df["fusion_pred"], title="KM: Fusion (median split)")


if __name__ == "__main__":
    main()
