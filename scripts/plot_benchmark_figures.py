#!/usr/bin/env python
"""Create manuscript draft figures from benchmark aggregate CSVs.

Expected inputs are one or more benchmark aggregate CSVs. Wide aggregate CSVs
are preferred and are used automatically when a sibling ``*_wide.csv`` exists.
Two layouts are supported:

1. Wide rows with columns such as:
   cohort, mode, tile_aggregator, slide_encoder, experiment_base, run_name,
   model, run_dir, clinical_pr_auc, fusion_pr_auc, wsi_pr_auc,
   clinical_roc_auc, fusion_roc_auc, wsi_roc_auc

2. Long rows with columns such as:
   cohort, mode, tile_aggregator, slide_encoder, experiment_base, run_name,
   model, run_dir, method, roc_auc, pr_auc

The script writes publication-style PNG/PDF/SVG figures plus the exact plotted
long-form CSV data under the selected output directory.
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = {
    "roc": {
        "label": "ROC AUC",
        "wsi": "wsi_roc_auc",
        "fusion": "fusion_roc_auc",
        "clinical": "clinical_roc_auc",
    },
    "pr": {
        "label": "PR AUC",
        "wsi": "wsi_pr_auc",
        "fusion": "fusion_pr_auc",
        "clinical": "clinical_pr_auc",
    },
}

FAMILY_ORDER = ["Linear", "MLP", "ViT", "TransMIL", "Slide encoder"]
TILE_FAMILY_ORDER = ["Linear", "MLP", "ViT", "TransMIL"]
SLIDE_ENCODER_ORDER = ["Chief", "Eagle", "Madeleine", "Prism", "Titan"]
METHOD_ORDER = ["wsi", "fusion", "clinical"]
METHOD_LABELS = {"wsi": "WSI-only", "fusion": "Fusion", "clinical": "Clinical-only"}
METHOD_COLORS = {"wsi": "#3B6EA8", "fusion": "#C44E52", "clinical": "#4A4A4A"}
FAMILY_COLORS = {
    "Linear": "#4C78A8",
    "MLP": "#F58518",
    "ViT": "#54A24B",
    "TransMIL": "#B279A2",
    "Slide encoder": "#E45756",
}


def _warn(messages: list[str], message: str) -> None:
    messages.append(message)
    print(f"WARNING: {message}")


def _clean_string(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _title(prefix: str, text: str) -> str:
    return f"{prefix} {text}".strip() if prefix else text


def _display_cohort(value: object) -> str:
    text = _clean_string(value)
    if text.lower() == "lusc":
        return "LUSC"
    if text.lower() == "vulvar":
        return "Vulvar"
    return text


def _metric_keys(metric_set: str) -> list[str]:
    if metric_set == "both":
        return ["roc", "pr"]
    return [metric_set]


def _method_from_value(value: object) -> str:
    text = _clean_string(value).lower().replace("_", " ").replace("-", " ")
    if text in {"wsi", "wsi only"}:
        return "wsi"
    if text == "fusion":
        return "fusion"
    if text in {"clinical", "clinical only"}:
        return "clinical"
    return text


def _normalize_aggregation_family(mode: object, tile_aggregator: object) -> str:
    if _clean_string(mode).lower() == "slide":
        return "Slide encoder"
    agg = _clean_string(tile_aggregator).lower()
    if agg == "linear":
        return "Linear"
    if agg == "mlp":
        return "MLP"
    if agg == "vit":
        return "ViT"
    if agg == "transmil":
        return "TransMIL"
    return _clean_string(tile_aggregator) or "Unknown"


def _display_model(model: object) -> str:
    text = _clean_string(model)
    return text if text else "Unknown"


def _display_encoder(value: object) -> str:
    text = _clean_string(value)
    if not text:
        return "Unknown"
    return text.replace("_", " ").replace("-", " ").title()


def _read_inputs(paths: list[str], warnings_out: list[str]) -> pd.DataFrame:
    frames = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.exists() and not path.stem.endswith("_wide"):
            wide_path = path.with_name(f"{path.stem}_wide{path.suffix}")
            if wide_path.exists():
                print(f"Using wide aggregate for {path}: {wide_path}")
                path = wide_path
        if not path.exists():
            _warn(warnings_out, f"input CSV not found and will be skipped: {path}")
            continue
        df = pd.read_csv(path)
        df["source_csv"] = str(path)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No readable input CSVs were provided.")
    return pd.concat(frames, ignore_index=True, sort=False)


def _id_columns(df: pd.DataFrame) -> list[str]:
    cols = [
        "cohort",
        "mode",
        "tile_aggregator",
        "slide_encoder",
        "experiment_base",
        "run_name",
        "model",
        "run_dir",
        "source_csv",
    ]
    if "n_patients" in df.columns:
        cols.append("n_patients")
    return cols


def _ensure_required_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "cohort" not in df.columns:
        df["cohort"] = "cohort"
    for col in ["mode", "tile_aggregator", "slide_encoder", "experiment_base", "run_name", "model", "run_dir"]:
        if col not in df.columns:
            df[col] = np.nan
    return df


def _has_wide_metrics(df: pd.DataFrame) -> bool:
    return any(spec[key] in df.columns for spec in METRICS.values() for key in METHOD_ORDER)


def _wide_from_long(df: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    if "method" not in df.columns:
        raise ValueError("Input must contain wide metric columns or a long-format 'method' column.")
    metric_cols = [c for c in ["roc_auc", "pr_auc"] if c in df.columns]
    if not metric_cols:
        raise ValueError("Long-format inputs must contain roc_auc and/or pr_auc columns.")

    work = df.copy()
    work["method_key"] = work["method"].map(_method_from_value)

    # pandas pivot_table drops NaN keys; use a sentinel so slide/tile rows are preserved.
    missing_marker = "__MISSING__"
    for col in id_cols:
        work[col] = work[col].astype("object").where(work[col].notna(), missing_marker)

    pieces = []
    for metric_col in metric_cols:
        metric_key = metric_col.replace("_auc", "")
        pivot = (
            work.pivot_table(
                index=id_cols,
                columns="method_key",
                values=metric_col,
                aggfunc="mean",
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )
        for method in METHOD_ORDER:
            pivot[f"{method}_{metric_key}_auc"] = pivot[method] if method in pivot.columns else np.nan
        pieces.append(pivot[id_cols + [f"{m}_{metric_key}_auc" for m in METHOD_ORDER]])

    result = pieces[0]
    for piece in pieces[1:]:
        result = result.merge(piece, on=id_cols, how="outer")
    for col in id_cols:
        result[col] = result[col].replace(missing_marker, np.nan)
    return result


def _normalize_results(raw: pd.DataFrame) -> pd.DataFrame:
    raw = _ensure_required_id_columns(raw)
    id_cols = _id_columns(raw)

    if _has_wide_metrics(raw):
        result = raw.copy()
    else:
        result = _wide_from_long(raw, id_cols)

    for spec in METRICS.values():
        for method in METHOD_ORDER:
            if spec[method] not in result.columns:
                result[spec[method]] = np.nan
    keep_cols = id_cols + [spec[m] for spec in METRICS.values() for m in METHOD_ORDER]
    result = result[keep_cols].drop_duplicates()

    result["cohort"] = result["cohort"].map(lambda x: _clean_string(x) or "cohort")
    result["mode"] = result["mode"].map(_clean_string)
    result["aggregation_family"] = [
        _normalize_aggregation_family(mode, agg)
        for mode, agg in zip(result["mode"], result["tile_aggregator"])
    ]
    result["model_label"] = result["model"].map(_display_model)
    result["slide_encoder_label"] = result["slide_encoder"].map(_display_encoder)
    result["method_id"] = [
        "|".join(
            [
                _clean_string(row.get("cohort")),
                _clean_string(row.get("mode")),
                _clean_string(row.get("aggregation_family")),
                _clean_string(row.get("slide_encoder_label")),
                _clean_string(row.get("model_label")),
                _clean_string(row.get("run_name")),
            ]
        )
        for _, row in result.iterrows()
    ]
    return result


def _run_checks(df: pd.DataFrame, warnings_out: list[str]) -> dict[str, object]:
    cohorts = sorted(df["cohort"].dropna().astype(str).unique())
    families = [f for f in FAMILY_ORDER if f in set(df["aggregation_family"])]
    extra_families = sorted(set(df["aggregation_family"]) - set(FAMILY_ORDER))
    families = families + extra_families

    clinical_consistency: dict[str, dict[str, object]] = {}
    for cohort, cdf in df.groupby("cohort", dropna=False):
        clinical_consistency[str(cohort)] = {}
        for metric_key, spec in METRICS.items():
            vals = pd.to_numeric(cdf[spec["clinical"]], errors="coerce").dropna().round(12).unique()
            clinical_consistency[str(cohort)][metric_key] = {
                "n_unique": int(len(vals)),
                "values": [float(v) for v in sorted(vals)],
                "consistent": len(vals) <= 1,
            }
            if len(vals) > 1:
                _warn(
                    warnings_out,
                    f"clinical-only {metric_key.upper()} has {len(vals)} unique values for cohort {cohort}",
                )
        if "n_patients" in cdf.columns:
            vals = pd.to_numeric(cdf["n_patients"], errors="coerce").dropna().unique()
            if len(vals) > 1:
                _warn(warnings_out, f"n_patients has {len(vals)} unique values for cohort {cohort}: {sorted(vals)}")

    print(f"Cohorts found: {', '.join(cohorts) if cohorts else '(none)'}")
    print(f"Aggregation families found: {', '.join(families) if families else '(none)'}")
    return {"cohorts": cohorts, "families": families, "clinical_consistency": clinical_consistency}


def _setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_plot_data(df: pd.DataFrame, out_dir: Path, stem: str, saved: list[Path]) -> None:
    path = out_dir / f"{stem}_data.csv"
    df.to_csv(path, index=False)
    saved.append(path)
    print(f"Saved: {path}")


def _save_figure(fig: plt.Figure, out_dir: Path, stem: str, formats: list[str], saved: list[Path]) -> None:
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        kwargs = {"bbox_inches": "tight"}
        if fmt.lower() == "png":
            kwargs["dpi"] = 300
        fig.savefig(path, **kwargs)
        saved.append(path)
        print(f"Saved: {path}")
    plt.close(fig)


def _cohort_axes(n: int, width_per_panel: float = 4.0, height: float = 3.4) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(
        1,
        max(n, 1),
        figsize=(max(n, 1) * width_per_panel, height),
        squeeze=False,
        layout="constrained",
    )
    return fig, axes.ravel()


def _cohort_ylim(values: pd.Series, margin: float = 0.04) -> tuple[float, float]:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return 0.0, 1.0
    lo = max(0.0, float(vals.min()) - margin)
    hi = min(1.0, float(vals.max()) + margin)
    if math.isclose(lo, hi):
        lo = max(0.0, lo - 0.05)
        hi = min(1.0, hi + 0.05)
    return lo, hi


def _common_ylim(*series: pd.Series, margin: float = 0.02) -> tuple[float, float]:
    values = [pd.to_numeric(s, errors="coerce") for s in series if s is not None]
    if not values:
        return 0.0, 1.0
    return _cohort_ylim(pd.concat(values, ignore_index=True), margin=margin)


def plot_fusion_slope(df: pd.DataFrame, metric_key: str, out_dir: Path, formats: list[str], title_prefix: str, saved: list[Path]) -> None:
    spec = METRICS[metric_key]
    cohorts = sorted(df["cohort"].unique())
    rows = []
    for _, row in df.iterrows():
        for method in METHOD_ORDER:
            rows.append(
                {
                    "cohort": row["cohort"],
                    "method_id": row["method_id"],
                    "model": row["model_label"],
                    "mode": row["mode"],
                    "aggregation_family": row["aggregation_family"],
                    "x_label": METHOD_LABELS[method],
                    "x_order": METHOD_ORDER.index(method),
                    "method": method,
                    "value": row[spec[method]],
                }
            )
    plot_df = pd.DataFrame(rows).dropna(subset=["value"])
    stem = f"figure_fusion_slope_{metric_key}"
    _save_plot_data(plot_df, out_dir, stem, saved)
    if plot_df.empty:
        return

    fig, axes = _cohort_axes(len(cohorts), width_per_panel=3.6, height=3.5)
    for ax, cohort in zip(axes, cohorts):
        cwide = df[df["cohort"] == cohort]
        cplot = plot_df[plot_df["cohort"] == cohort]
        for _, row in cwide.iterrows():
            vals = [row[spec[m]] for m in METHOD_ORDER]
            if pd.isna(vals).all():
                continue
            color = FAMILY_COLORS.get(row["aggregation_family"], "#808080")
            ax.plot(range(3), vals, color=color, alpha=0.22, linewidth=0.9)
        med = cwide[[spec[m] for m in METHOD_ORDER]].median(skipna=True).values
        ax.plot(range(3), med, color="black", linewidth=2.4, marker="o", markersize=4, label="Median")
        n = int(cwide[[spec["wsi"], spec["fusion"]]].dropna(how="all").shape[0])
        ax.set_title(f"{_display_cohort(cohort)} (n={n})")
        ax.set_xticks(range(3), [METHOD_LABELS[m] for m in METHOD_ORDER], rotation=25, ha="right")
        ax.set_ylabel(spec["label"])
        ax.set_ylim(_cohort_ylim(cplot["value"]))
        ax.grid(axis="y", alpha=0.25)
    for ax in axes[len(cohorts) :]:
        ax.axis("off")
    fig.suptitle(_title(title_prefix, f"Fusion Improvement ({spec['label']})"))
    _save_figure(fig, out_dir, stem, formats, saved)


def _jitter(center: float, n: int, width: float = 0.12) -> np.ndarray:
    if n <= 1:
        return np.array([center])
    return center + np.linspace(-width, width, n)


def plot_aggregator_distribution(
    df: pd.DataFrame,
    metric_key: str,
    out_dir: Path,
    formats: list[str],
    title_prefix: str,
    saved: list[Path],
    shared_y_by_metric: bool,
) -> None:
    spec = METRICS[metric_key]
    rows = []
    for _, row in df.iterrows():
        for method in ["wsi", "fusion"]:
            rows.append(
                {
                    "cohort": row["cohort"],
                    "method_id": row["method_id"],
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "aggregation_family": row["aggregation_family"],
                    "model": row["model_label"],
                    "value": row[spec[method]],
                    "clinical_reference": row[spec["clinical"]],
                }
            )
    plot_df = pd.DataFrame(rows).dropna(subset=["value"])
    stem = f"figure_aggregator_distribution_{metric_key}"
    _save_plot_data(plot_df, out_dir, stem, saved)
    if plot_df.empty:
        return
    shared_ylim = None
    if shared_y_by_metric:
        shared_ylim = _common_ylim(plot_df["value"], plot_df["clinical_reference"], margin=0.02)
        print(f"{stem} y-limits: {shared_ylim[0]:.3f}, {shared_ylim[1]:.3f}")

    cohorts = sorted(plot_df["cohort"].unique())
    fig, axes = plt.subplots(
        len(cohorts),
        2,
        figsize=(8.8, max(2.9 * len(cohorts), 3.2)),
        squeeze=False,
        sharey=False,
        layout="constrained",
    )
    for row_idx, cohort in enumerate(cohorts):
        for col_idx, method in enumerate(["wsi", "fusion"]):
            ax = axes[row_idx, col_idx]
            cdf = plot_df[(plot_df["cohort"] == cohort) & (plot_df["method"] == method)]
            families = [f for f in FAMILY_ORDER if f in set(cdf["aggregation_family"])]
            families += sorted(set(cdf["aggregation_family"]) - set(families))
            values_for_ylim = list(cdf["value"])
            box_values = []
            box_positions = []
            for pos, family in enumerate(families):
                vals = pd.to_numeric(cdf.loc[cdf["aggregation_family"] == family, "value"], errors="coerce").dropna().values
                if len(vals) == 0:
                    continue
                box_values.append(vals)
                box_positions.append(pos)
                ax.scatter(
                    _jitter(pos, len(vals)),
                    vals,
                    s=16,
                    alpha=0.62,
                    color=FAMILY_COLORS.get(family, "#666666"),
                    edgecolor="white",
                    linewidth=0.25,
                    zorder=3,
                )
            if box_values:
                ax.boxplot(
                    box_values,
                    positions=box_positions,
                    widths=0.55,
                    patch_artist=True,
                    showfliers=False,
                    medianprops={"color": "black", "linewidth": 1.3},
                    boxprops={"facecolor": "#F2F2F2", "edgecolor": "#606060", "linewidth": 0.8},
                    whiskerprops={"color": "#606060", "linewidth": 0.8},
                    capprops={"color": "#606060", "linewidth": 0.8},
                )
            clinical = pd.to_numeric(cdf["clinical_reference"], errors="coerce").dropna()
            if not clinical.empty:
                ref = float(clinical.median())
                values_for_ylim.append(ref)
                ax.axhline(ref, color=METHOD_COLORS["clinical"], linewidth=1.1, linestyle="--", alpha=0.8)
            ax.set_title(f"{_display_cohort(cohort)} - {METHOD_LABELS[method]}")
            ax.set_xticks(range(len(families)), families, rotation=35, ha="right")
            ax.set_ylabel(spec["label"] if col_idx == 0 else "")
            ax.set_ylim(shared_ylim or _cohort_ylim(pd.Series(values_for_ylim)))
            ax.grid(axis="y", alpha=0.22)
    fig.suptitle(_title(title_prefix, f"Aggregation Strategy Distribution ({spec['label']})"))
    _save_figure(fig, out_dir, stem, formats, saved)


def _pivot_heatmap(cdf: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    table = cdf.pivot_table(
        index="model_label",
        columns="aggregation_family",
        values=metric_col,
        aggfunc="mean",
    )
    ordered_cols = [c for c in TILE_FAMILY_ORDER if c in table.columns] + sorted(set(table.columns) - set(TILE_FAMILY_ORDER))
    table = table[ordered_cols]
    if not table.empty:
        row_order = table.mean(axis=1, skipna=True).sort_values(ascending=False).index
        table = table.loc[row_order]
    return table


def plot_fm_aggregator_heatmaps(
    df: pd.DataFrame,
    metric_key: str,
    out_dir: Path,
    formats: list[str],
    title_prefix: str,
    saved: list[Path],
    primary_value: str,
    include_wsi_heatmaps: bool,
    annotate_heatmaps: bool,
) -> None:
    spec = METRICS[metric_key]
    methods = [primary_value]
    if include_wsi_heatmaps and "wsi" not in methods:
        methods.append("wsi")

    tile_df = df[(df["mode"].str.lower() != "slide") & (df["aggregation_family"] != "Slide encoder")].copy()
    for method in methods:
        metric_col = spec[method]
        rows = tile_df[
            ["cohort", "model_label", "aggregation_family", "tile_aggregator", metric_col, "method_id"]
        ].rename(columns={metric_col: "value"})
        stem = f"figure_fm_aggregator_heatmap_{method}_{metric_key}_auc"
        _save_plot_data(rows.dropna(subset=["value"]), out_dir, stem, saved)
        if rows["value"].dropna().empty:
            continue

        cohorts = sorted(tile_df["cohort"].unique())
        all_values = pd.to_numeric(rows["value"], errors="coerce").dropna()
        vmin = max(0.0, float(all_values.min()) - 0.01) if not all_values.empty else 0.0
        vmax = min(1.0, float(all_values.max()) + 0.01) if not all_values.empty else 1.0
        print(f"{stem} color scale: {vmin:.3f}, {vmax:.3f}")
        for cohort in cohorts:
            table = _pivot_heatmap(tile_df[tile_df["cohort"] == cohort], metric_col)
            if table.empty:
                continue
            cohort_label = _display_cohort(cohort)
            cohort_stem = f"figure_fm_aggregator_heatmap_{_clean_string(cohort).lower()}_{method}_{metric_key}_auc"
            matrix_path = out_dir / f"{cohort_stem}_matrix.csv"
            table.to_csv(matrix_path)
            saved.append(matrix_path)
            print(f"Saved: {matrix_path}")

            height = max(3.4, 0.28 * table.shape[0] + 1.5)
            width = max(4.6, 0.8 * table.shape[1] + 3.0)
            fig, ax = plt.subplots(figsize=(width, height), layout="constrained")
            image = ax.imshow(table.values, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(
                _title(title_prefix, f"{cohort_label} FM x Aggregator - {METHOD_LABELS[method]} ({spec['label']})")
            )
            ax.set_xticks(range(table.shape[1]), table.columns, rotation=45, ha="right", rotation_mode="anchor")
            ax.set_yticks(range(table.shape[0]), table.index)
            ax.tick_params(axis="y", labelsize=8)
            if annotate_heatmaps and table.shape[0] * table.shape[1] <= 120:
                threshold = vmin + 0.55 * (vmax - vmin)
                for i in range(table.shape[0]):
                    for j in range(table.shape[1]):
                        val = table.iat[i, j]
                        if pd.notna(val):
                            color = "white" if val < threshold else "black"
                            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5, color=color)
            fig.colorbar(image, ax=ax, location="right", shrink=0.8, pad=0.02, label=spec["label"])
            _save_figure(fig, out_dir, cohort_stem, formats, saved)


def plot_slide_encoder_comparison(
    df: pd.DataFrame, metric_key: str, out_dir: Path, formats: list[str], title_prefix: str, saved: list[Path]
) -> None:
    spec = METRICS[metric_key]
    slide_df = df[df["mode"].str.lower() == "slide"].copy()
    rows = []
    for _, row in slide_df.iterrows():
        for method in ["wsi", "fusion"]:
            rows.append(
                {
                    "cohort": row["cohort"],
                    "slide_encoder": row["slide_encoder_label"],
                    "model": row["model_label"],
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "value": row[spec[method]],
                    "clinical_reference": row[spec["clinical"]],
                    "method_id": row["method_id"],
                }
            )
    plot_df = pd.DataFrame(rows).dropna(subset=["value"])
    stem = f"figure_slide_encoder_comparison_{metric_key}"
    _save_plot_data(plot_df, out_dir, stem, saved)
    if plot_df.empty:
        return
    shared_ylim = _common_ylim(plot_df["value"], plot_df["clinical_reference"], margin=0.02)
    print(f"{stem} y-limits: {shared_ylim[0]:.3f}, {shared_ylim[1]:.3f}")

    cohorts = sorted(plot_df["cohort"].unique())
    fig, axes = _cohort_axes(len(cohorts), width_per_panel=4.2, height=3.5)
    for ax, cohort in zip(axes, cohorts):
        cwide = slide_df[slide_df["cohort"] == cohort].copy()
        if cwide.empty:
            ax.axis("off")
            continue
        cwide["encoder_order"] = cwide["slide_encoder_label"].map(
            {name: idx for idx, name in enumerate(SLIDE_ENCODER_ORDER)}
        ).fillna(len(SLIDE_ENCODER_ORDER))
        cwide = cwide.sort_values(["encoder_order", "slide_encoder_label"])
        encoders = list(cwide["slide_encoder_label"])
        x = np.arange(len(encoders))
        for i, (_, row) in enumerate(cwide.iterrows()):
            vals = [row[spec["wsi"]], row[spec["fusion"]]]
            if pd.notna(vals).all():
                ax.plot([i - 0.12, i + 0.12], vals, color="#888888", linewidth=0.9, alpha=0.8, zorder=2)
            ax.scatter(i - 0.12, row[spec["wsi"]], color=METHOD_COLORS["wsi"], s=28, label=METHOD_LABELS["wsi"] if i == 0 else None, zorder=3)
            ax.scatter(i + 0.12, row[spec["fusion"]], color=METHOD_COLORS["fusion"], s=28, label=METHOD_LABELS["fusion"] if i == 0 else None, zorder=3)
        clinical = pd.to_numeric(cwide[spec["clinical"]], errors="coerce").dropna()
        if not clinical.empty:
            ax.axhline(float(clinical.median()), color=METHOD_COLORS["clinical"], linestyle="--", linewidth=1.1, label=METHOD_LABELS["clinical"])
        ax.set_title(_display_cohort(cohort))
        ax.set_xticks(x, encoders, rotation=35, ha="right")
        ax.set_ylabel(spec["label"])
        ax.set_ylim(shared_ylim)
        ax.grid(axis="y", alpha=0.22)
        ax.legend(loc="best", frameon=False)
    for ax in axes[len(cohorts) :]:
        ax.axis("off")
    fig.suptitle(_title(title_prefix, f"Slide-Level Encoder Comparison ({spec['label']})"))
    _save_figure(fig, out_dir, stem, formats, saved)


def _spearman_fallback(x: pd.Series, y: pd.Series) -> float:
    xr = x.rank(method="average")
    yr = y.rank(method="average")
    return float(xr.corr(yr, method="pearson"))


def _try_adjust_text(texts: list[object], ax: plt.Axes) -> None:
    try:
        from adjustText import adjust_text
    except Exception:
        return
    adjust_text(texts, ax=ax, arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.4})


def _draw_rank_stability_panel(
    paired: pd.DataFrame,
    cohort_x: str,
    cohort_y: str,
    spec: dict[str, str],
    primary_value: str,
    rho: float,
    title_prefix: str,
    label_top_n: int,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.4, 3.8), layout="constrained")
    ax.scatter(paired[cohort_x], paired[cohort_y], color="#4C78A8", alpha=0.72, s=24)

    lim = _common_ylim(paired[cohort_x], paired[cohort_y], margin=0.02)
    ax.plot(lim, lim, color="#777777", linestyle="--", linewidth=0.8, alpha=0.45, zorder=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    if label_top_n > 0:
        labeled = paired.sort_values("mean_cross_cohort", ascending=False).head(label_top_n)
        texts = []
        for _, row in labeled.iterrows():
            label = str(row["rank_key"]).replace("|", "\n")
            texts.append(
                ax.annotate(
                    label,
                    (row[cohort_x], row[cohort_y]),
                    xytext=(4, 3),
                    textcoords="offset points",
                    fontsize=6,
                )
            )
        _try_adjust_text(texts, ax)

    x_label = _display_cohort(cohort_x)
    y_label = _display_cohort(cohort_y)
    ax.set_xlabel(f"{x_label} {METHOD_LABELS[primary_value]} {spec['label']}")
    ax.set_ylabel(f"{y_label} {METHOD_LABELS[primary_value]} {spec['label']}")
    ax.set_title(f"{x_label} vs {y_label} (Spearman rho={rho:.2f}, n={len(paired)})", fontsize=9)
    ax.grid(alpha=0.25)
    fig.suptitle(_title(title_prefix, f"Cross-Cohort Rank Stability ({spec['label']})"), fontsize=10)
    return fig


def plot_rank_stability(
    df: pd.DataFrame,
    metric_key: str,
    out_dir: Path,
    formats: list[str],
    title_prefix: str,
    saved: list[Path],
    primary_value: str,
    label_top_n: int,
) -> None:
    spec = METRICS[metric_key]
    cohorts = sorted(df["cohort"].unique())
    if len(cohorts) < 2:
        return
    preferred = None
    for pair in [("lusc", "vulvar"), ("LUSC", "vulvar")]:
        if pair[0] in cohorts and pair[1] in cohorts:
            preferred = pair
            break
    cohort_x, cohort_y = preferred or (cohorts[0], cohorts[1])

    tmp = df.copy()
    tmp["rank_key"] = np.where(
        tmp["mode"].str.lower() == "slide",
        "Slide encoder|" + tmp["slide_encoder_label"],
        tmp["aggregation_family"] + "|" + tmp["model_label"],
    )
    tmp = tmp[["cohort", "rank_key", "aggregation_family", "model_label", "slide_encoder_label", spec[primary_value]]].rename(
        columns={spec[primary_value]: "value"}
    )
    tmp = tmp.dropna(subset=["value"])
    paired = tmp.pivot_table(index="rank_key", columns="cohort", values="value", aggfunc="mean")
    if cohort_x not in paired.columns or cohort_y not in paired.columns:
        return
    paired = paired[[cohort_x, cohort_y]].dropna().reset_index()
    if paired.empty:
        return
    paired["mean_cross_cohort"] = paired[[cohort_x, cohort_y]].mean(axis=1)
    paired["abs_cross_cohort_delta"] = (paired[cohort_x] - paired[cohort_y]).abs()
    paired[f"rank_{cohort_x}"] = paired[cohort_x].rank(ascending=False, method="average")
    paired[f"rank_{cohort_y}"] = paired[cohort_y].rank(ascending=False, method="average")
    rho = _spearman_fallback(paired[cohort_x], paired[cohort_y])
    corr = pd.DataFrame(
        [
            {
                "metric": metric_key,
                "primary_value": primary_value,
                "cohort_x": cohort_x,
                "cohort_y": cohort_y,
                "n_methods": int(len(paired)),
                "spearman_rho": rho,
            }
        ]
    )
    stem = f"figure_cross_cohort_rank_stability_{metric_key}"
    _save_plot_data(paired, out_dir, stem, saved)
    corr_path = out_dir / f"{stem}_correlation.csv"
    corr.to_csv(corr_path, index=False)
    saved.append(corr_path)
    print(f"Saved: {corr_path}")

    fig = _draw_rank_stability_panel(
        paired, cohort_x, cohort_y, spec, primary_value, rho, title_prefix, label_top_n
    )
    _save_figure(fig, out_dir, stem, formats, saved)
    unlabeled_fig = _draw_rank_stability_panel(
        paired, cohort_x, cohort_y, spec, primary_value, rho, title_prefix, 0
    )
    _save_figure(unlabeled_fig, out_dir, f"{stem}_unlabeled", formats, saved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csvs", nargs="+", required=True, help="One or more benchmark aggregate CSVs.")
    parser.add_argument("--out_dir", default="outputs/paper_figures", help="Directory for figures and plotted data CSVs.")
    parser.add_argument("--metric_set", choices=["roc", "pr", "both"], default="both")
    parser.add_argument("--primary_value", choices=["fusion", "wsi"], default="fusion", help="Metric source for FM heatmaps and rank stability.")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf", "svg"], help="Figure formats to write.")
    parser.add_argument("--title_prefix", default="", help="Optional prefix added to figure titles.")
    parser.add_argument("--include_wsi_heatmaps", action="store_true", help="Also write WSI-only FM x aggregator heatmaps.")
    parser.add_argument(
        "--shared_y_by_metric",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use common y-axis limits across panels for each metric figure.",
    )
    parser.add_argument(
        "--annotate_heatmaps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Annotate heatmap cells when the matrix is not too large.",
    )
    parser.add_argument("--label_top_n", type=int, default=8, help="Number of top cross-cohort methods to label in rank-stability plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _setup_matplotlib()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = [fmt.lower().lstrip(".") for fmt in args.formats]

    warnings_out: list[str] = []
    raw = _read_inputs(args.input_csvs, warnings_out)
    df = _normalize_results(raw)
    check_summary = _run_checks(df, warnings_out)

    saved: list[Path] = []
    for metric_key in _metric_keys(args.metric_set):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            plot_fusion_slope(df, metric_key, out_dir, formats, args.title_prefix, saved)
            plot_aggregator_distribution(
                df, metric_key, out_dir, formats, args.title_prefix, saved, args.shared_y_by_metric
            )
            plot_fm_aggregator_heatmaps(
                df,
                metric_key,
                out_dir,
                formats,
                args.title_prefix,
                saved,
                args.primary_value,
                args.include_wsi_heatmaps,
                args.annotate_heatmaps,
            )
            plot_slide_encoder_comparison(df, metric_key, out_dir, formats, args.title_prefix, saved)
            plot_rank_stability(
                df, metric_key, out_dir, formats, args.title_prefix, saved, args.primary_value, args.label_top_n
            )

    summary_path = out_dir / "benchmark_figure_checks.csv"
    rows = []
    for cohort, metric_info in check_summary["clinical_consistency"].items():
        for metric_key, info in metric_info.items():
            rows.append(
                {
                    "cohort": cohort,
                    "metric": metric_key,
                    "clinical_n_unique": info["n_unique"],
                    "clinical_consistent": info["consistent"],
                    "clinical_values": ";".join(f"{v:.6g}" for v in info["values"]),
                }
            )
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    saved.append(summary_path)
    print(f"Saved: {summary_path}")

    print(f"\nSaved {len(saved)} files under {out_dir}:")
    for path in saved:
        print(f"  {path}")
    if warnings_out:
        print("\nWarnings:")
        for message in warnings_out:
            print(f"  - {message}")
    else:
        print("\nWarnings: none")


if __name__ == "__main__":
    main()
