#!/usr/bin/env python

import argparse
from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt

from wsi_recurrence.metrics import compute_auc, compute_pr_auc, plot_pr, plot_roc
from wsi_recurrence.clinical import analysis_defaults, load_project_config


def _safe_label(value: str) -> str:
    label = re.sub(r"[\s/:\\]+", "_", str(value).strip())
    label = re.sub(r"_+", "_", label).strip("_")
    return label


def _infer_model_label(out_dir: Path) -> str:
    # Expected pipeline layout: .../analysis/<model>/figures
    if out_dir.name == "figures" and out_dir.parent.name:
        return out_dir.parent.name
    return ""


def _infer_experiment_label(out_dir: Path) -> str:
    parts = out_dir.resolve().parts
    for i in range(len(parts) - 2):
        if parts[i] == "outputs" and parts[i + 1] == "runs":
            return parts[i + 2]
    return ""


def _title(base: str, model_label: str, experiment_label: str) -> str:
    if model_label and experiment_label:
        return f"{base} - {model_label}\n{experiment_label}"
    if model_label:
        return f"{base} - {model_label}"
    return base


def _plot_path(out_dir: Path, stem: str, model_label: str, *, legacy_stem: str | None = None) -> Path:
    safe = _safe_label(model_label)
    if safe:
        return out_dir / f"{stem}__{safe}.png"
    return out_dir / f"{legacy_stem or stem}.png"


def _savefig(fig: plt.Figure, out_path: Path, *, metadata: dict[str, str], legacy_path: Path | None = None) -> None:
    fig.savefig(out_path, dpi=300, bbox_inches="tight", metadata=metadata)
    print(f"Saved: {out_path}")
    if legacy_path is not None and legacy_path != out_path:
        fig.savefig(legacy_path, dpi=300, bbox_inches="tight", metadata=metadata)
        print(f"Saved legacy alias: {legacy_path}")


def _try_import_lifelines():
    try:
        from lifelines import KaplanMeierFitter
    except Exception:
        return None
    return KaplanMeierFitter


def _plot_km(
    out_path: Path,
    time_s: pd.Series,
    event_s: pd.Series,
    risk_s: pd.Series,
    title: str,
    *,
    metadata: dict[str, str],
    legacy_path: Path | None = None,
) -> bool:
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

    _savefig(fig, out_path, metadata=metadata, legacy_path=legacy_path)
    plt.close(fig)
    return True


def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _infer_wsi_pred_col(df: pd.DataFrame, *, explicit: str | None) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Missing prediction column {explicit!r}; available columns: {sorted(df.columns)}")
        return explicit

    direct = _first_present(df, ["pred", "pred_wsi"])
    if direct is not None:
        return direct

    pred_cols = [str(c) for c in df.columns if str(c).startswith("pred_")]
    if len(pred_cols) == 1:
        return pred_cols[0]
    raise ValueError(
        "Could not infer WSI prediction column. "
        "Expected 'pred'/'pred_wsi' or a single 'pred_*' column; "
        f"available columns: {sorted(df.columns)}. "
        "Pass --pred_col to select explicitly."
    )


def _infer_label_col(project_path: str, *, cli_label_col: str) -> str:
    if str(cli_label_col).strip():
        return str(cli_label_col).strip()
    if str(project_path).strip():
        cfg = load_project_config(Path(str(project_path)))
        columns_cfg = (cfg.get("columns", {}) or {}) if isinstance(cfg, dict) else {}
        crossval_cfg = (cfg.get("crossval", {}) or {}) if isinstance(cfg, dict) else {}
        inferred = str(columns_cfg.get("label") or "").strip() or str(crossval_cfg.get("ground_truth_label") or "").strip()
        if inferred:
            return inferred
    return "recur"


def _infer_aux_pred_cols(df: pd.DataFrame) -> tuple[str | None, str | None]:
    clinical = _first_present(df, ["clinical_pred", "pred_clinical"])
    fusion = _first_present(df, ["fusion_pred", "pred_fusion"])
    return clinical, fusion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--predictions",
        default="",
        help="Predictions CSV. Supports WSI-only outputs (e.g. all_predictions_<model>.csv) or fusion_predictions.csv.",
    )
    ap.add_argument(
        "--fusion_predictions",
        default="",
        help="Deprecated alias for --predictions (kept for backward compatibility).",
    )
    ap.add_argument("--project", default="", help="Optional project YAML for label/outcome column metadata.")
    ap.add_argument("--label_col", default="", help="Ground-truth label column (overrides project config).")
    ap.add_argument("--pred_col", default="", help="WSI prediction column (defaults to inferred).")
    ap.add_argument("--time_col", default="", help="Optional time-to-event column for KM plotting.")
    ap.add_argument("--event_col", default="", help="Optional event indicator column for KM plotting.")
    ap.add_argument("--model_label", default="", help="Model/tool label for plot titles and filenames.")
    ap.add_argument("--experiment_label", default="", help="Experiment/run label for plot titles and PNG metadata.")
    ap.add_argument("--legacy_plot_names", action="store_true", help="Also write old plot filenames as aliases.")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_label = str(args.model_label).strip() or _infer_model_label(out_dir)
    experiment_label = str(args.experiment_label).strip() or _infer_experiment_label(out_dir)

    pred_path = (args.predictions or "").strip() or (args.fusion_predictions or "").strip()
    if not pred_path:
        raise ValueError("Provide --predictions (or legacy --fusion_predictions).")
    pred_path = str(pred_path)
    metadata = {
        "model_label": model_label,
        "experiment_label": experiment_label,
        "source_predictions": pred_path,
    }

    df = pd.read_csv(pred_path)

    project_path = str(args.project).strip()
    label_col = _infer_label_col(project_path, cli_label_col=args.label_col)
    if label_col not in df.columns:
        raise ValueError(
            "Missing required label column "
            f"{label_col!r} in {pred_path}. "
            f"Available columns: {sorted(df.columns)}. "
            "Pass --project to infer from config or override with --label_col."
        )

    pred_col = _infer_wsi_pred_col(df, explicit=(str(args.pred_col).strip() or None))

    y_true = pd.to_numeric(df[label_col], errors="coerce").values
    y_wsi = pd.to_numeric(df[pred_col], errors="coerce").values

    clinical_col, fusion_col = _infer_aux_pred_cols(df)
    has_clinical = clinical_col is not None
    has_fusion = fusion_col is not None
    y_clin = pd.to_numeric(df[clinical_col], errors="coerce").values if clinical_col else None
    y_fusion = pd.to_numeric(df[fusion_col], errors="coerce").values if fusion_col else None

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

    if has_fusion:
        plot_roc(ax, y_true, y_fusion, label="Fusion")
    if has_clinical:
        plot_roc(ax, y_true, y_clin, label="Clinical")

    ax.set_title(_title("ROC curve", model_label, experiment_label))
    ax.legend()

    out_path = _plot_path(out_dir, "roc_curve", model_label, legacy_stem="roc_comparison")
    legacy_path = out_dir / "roc_comparison.png" if args.legacy_plot_names else None
    _savefig(fig, out_path, metadata=metadata, legacy_path=legacy_path)
    plt.close(fig)

    # -------------------------
    # PR comparison
    # -------------------------
    fig, ax = plt.subplots()
    plot_pr(ax, y_true, y_wsi, label="WSI")
    if has_fusion:
        plot_pr(ax, y_true, y_fusion, label="Fusion")
    if has_clinical:
        plot_pr(ax, y_true, y_clin, label="Clinical")
    ax.set_title(_title("Precision-Recall curve", model_label, experiment_label))
    ax.legend()
    pr_path = _plot_path(out_dir, "pr_curve", model_label, legacy_stem="pr_comparison")
    legacy_pr_path = out_dir / "pr_comparison.png" if args.legacy_plot_names else None
    _savefig(fig, pr_path, metadata=metadata, legacy_path=legacy_pr_path)
    plt.close(fig)

    # -------------------------
    # Summary metrics
    # -------------------------
    rows = [
        {"method": "WSI", "roc_auc": float(compute_auc(y_true, y_wsi)), "pr_auc": float(compute_pr_auc(y_true, y_wsi))},
    ]
    if has_fusion:
        rows.append(
            {
                "method": "Fusion",
                "roc_auc": float(compute_auc(y_true, y_fusion)),
                "pr_auc": float(compute_pr_auc(y_true, y_fusion)),
            }
        )
    if has_clinical:
        rows.append(
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
    time_col = str(args.time_col).strip()
    event_col = str(args.event_col).strip()
    if (not time_col) or (not event_col):
        if project_path:
            project_cfg = load_project_config(Path(project_path))
            analysis_cfg = analysis_defaults(project_cfg)
            time_col = time_col or str(analysis_cfg.get("outcome_time_col") or "").strip()
            event_col = event_col or str(analysis_cfg.get("outcome_event_col") or "").strip()
    time_col = time_col or "time_to_event"
    event_col = event_col or "event"

    if time_col not in df.columns or event_col not in df.columns:
        print(f"WARNING: {time_col!r}/{event_col!r} not found; skipping KM plotting.")
        return

    time_s = df[time_col]
    event_s = df[event_col]

    _plot_km(
        _plot_path(out_dir, "km_curve_wsi", model_label, legacy_stem="km_wsi"),
        time_s,
        event_s,
        df[pred_col],
        title=_title("Kaplan-Meier: WSI (median split)", model_label, experiment_label),
        metadata=metadata,
        legacy_path=(out_dir / "km_wsi.png") if args.legacy_plot_names else None,
    )
    if has_clinical:
        _plot_km(
            _plot_path(out_dir, "km_curve_clinical", model_label, legacy_stem="km_clinical"),
            time_s,
            event_s,
            df[clinical_col],
            title=_title("Kaplan-Meier: Clinical (median split)", model_label, experiment_label),
            metadata=metadata,
            legacy_path=(out_dir / "km_clinical.png") if args.legacy_plot_names else None,
        )
    if has_fusion:
        _plot_km(
            _plot_path(out_dir, "km_curve_fusion", model_label, legacy_stem="km_fusion"),
            time_s,
            event_s,
            df[fusion_col],
            title=_title("Kaplan-Meier: Fusion (median split)", model_label, experiment_label),
            metadata=metadata,
            legacy_path=(out_dir / "km_fusion.png") if args.legacy_plot_names else None,
        )


if __name__ == "__main__":
    main()
