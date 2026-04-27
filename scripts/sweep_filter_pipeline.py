#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
import yaml

from wsi_recurrence.experiment import load_experiment
from wsi_recurrence.stamp_runner import build_stamp_config, find_preprocess_output_dir
from wsi_recurrence.tile_filter.features import compute_features
from wsi_recurrence.tile_filter.h5_filter import ROUND_DECIMALS, filter_h5
from wsi_recurrence.tile_filter.zip_tiles import (
    build_zip_coord_index,
    match_tile_name,
    try_parse_tile_coords,
)


@dataclass(frozen=True)
class ThresholdSpec:
    value: float
    tag: str


def _under_project(project_dir: Path, path_like: str | Path) -> Path:
    p = Path(str(path_like))
    return p if p.is_absolute() else (project_dir / p)


def _parse_thresholds(s: str) -> list[ThresholdSpec]:
    out: list[ThresholdSpec] = []
    for raw in [x.strip() for x in str(s).split(",") if x.strip()]:
        val = float(raw)
        tag = raw.replace(".", "p").replace("-", "m")
        out.append(ThresholdSpec(value=val, tag=tag))
    if not out:
        raise ValueError("No thresholds provided.")
    return out


def _parse_gpus(val: str) -> list[str]:
    return [x.strip() for x in str(val).split(",") if x.strip()]


def _find_zip(cache_dir: Path, slide_id: str) -> Path | None:
    matches = sorted(cache_dir.glob(f"{slide_id}.*.zip"))
    if not matches:
        return None
    if len(matches) > 1:
        print(f"WARNING: multiple zips for {slide_id}; using first: {matches[0]}")
    return matches[0]


def _build_tile_coord_map(zip_names: list[str]) -> dict[tuple[float, float], str]:
    out: dict[tuple[float, float], str] = {}
    for name in zip_names:
        coords = try_parse_tile_coords(name)
        if coords is None:
            continue
        x_um, y_um = coords
        key = (round(float(x_um), ROUND_DECIMALS), round(float(y_um), ROUND_DECIMALS))
        out[key] = name
    return out


def _compute_keep_map_for_h5_coords(
    *,
    zip_path: Path,
    coords_um: np.ndarray,
    threshold: float,
    model,
    feature_names: list[str],
    coord_tol_um: float = 2.0,
) -> dict[tuple[float, float], bool]:
    """
    Build a keep_map keyed by (rounded x_um, rounded y_um) for coords in an .h5.
    keep = (bad_prob < threshold), matching build_tile_keep_masks.py behavior.
    """
    keys = [
        (round(float(x), ROUND_DECIMALS), round(float(y), ROUND_DECIMALS))
        for x, y in coords_um
    ]

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        coord_map = _build_tile_coord_map(names)
        zip_df = build_zip_coord_index(names)

        feat_rows: list[dict] = []
        feat_keys: list[tuple[float, float]] = []
        n_direct = 0
        n_fallback = 0
        n_missing = 0

        for (x_key, y_key), (x_um, y_um) in zip(keys, coords_um):
            tile_name = coord_map.get((x_key, y_key))
            if tile_name is None:
                tile_name, _ = match_tile_name(
                    float(x_um),
                    float(y_um),
                    zip_df,
                    tol=float(coord_tol_um),
                )
                if tile_name is None:
                    n_missing += 1
                    continue
                n_fallback += 1
            else:
                n_direct += 1

            img_bytes = zf.read(tile_name)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            feats = compute_features(img)
            feat_rows.append(feats)
            feat_keys.append((x_key, y_key))

    if feat_rows:
        feat_df = pd.DataFrame(feat_rows)
        X = feat_df[feature_names].values
        bad_prob = model.predict_proba(X)[:, 1].astype(float)
        keep_arr = bad_prob < float(threshold)
        keep_map = {k: bool(v) for k, v in zip(feat_keys, keep_arr)}

        # Prevent "all removed" failures: keep the single best tile if needed.
        if len(keep_map) > 0 and (not any(keep_map.values())):
            i_best = int(np.argmin(bad_prob))
            keep_map[feat_keys[i_best]] = True
            print(f"WARNING: threshold {threshold} would remove all tiles; keeping 1 tile for stability")
    else:
        keep_map = {}

    # Default to keep=True for coords we couldn't score.
    for k in keys:
        keep_map.setdefault(k, True)

    if n_missing:
        print(f"WARNING: {zip_path.name}: {n_missing}/{len(keys)} coords not matched to tiles; keeping them")
    if n_fallback:
        print(f"NOTE: {zip_path.name}: {n_fallback} coords matched via tolerance search (direct={n_direct})")
    return keep_map


def _filter_feature_dir(
    *,
    input_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    threshold: float,
    model,
    feature_names: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_files = sorted(input_dir.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found under {input_dir}")

    for h5_path in h5_files:
        slide_id = h5_path.stem
        out_h5 = output_dir / h5_path.name
        if out_h5.exists():
            continue

        zip_path = _find_zip(cache_dir, slide_id)
        if zip_path is None:
            print(f"WARNING: no zip found for {slide_id}; copying unfiltered")
            out_h5.write_bytes(h5_path.read_bytes())
            continue

        import h5py

        with h5py.File(h5_path, "r") as f:
            coords_um = f["coords"][:]

        keep_map = _compute_keep_map_for_h5_coords(
            zip_path=zip_path,
            coords_um=coords_um,
            threshold=threshold,
            model=model,
            feature_names=feature_names,
        )
        try:
            n_total, n_kept = filter_h5(h5_path, out_h5, keep_map)
            print(f"{slide_id}: kept {n_kept}/{n_total} (thr={threshold})")
        except RuntimeError as exc:
            print(f"WARNING: {slide_id}: {exc}; copying unfiltered")
            out_h5.write_bytes(h5_path.read_bytes())

    for extra in input_dir.iterdir():
        if extra.suffix != ".h5" and extra.is_file():
            dst = output_dir / extra.name
            if not dst.exists():
                dst.write_bytes(extra.read_bytes())


def _run(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    print(f"RUN: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env, cwd=str(cwd) if cwd is not None else None)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, default=Path("configs/project_lusc.yaml"))
    ap.add_argument("--experiment", type=Path, default=Path("configs/experiments/example_stamp_qc03.yaml"))
    ap.add_argument("--model", type=str, required=True, help="Model name (e.g. ctranspath)")
    ap.add_argument("--thresholds", type=str, required=True, help="Comma-separated thresholds, e.g. 0.1,0.2,0.3")
    ap.add_argument("--gpus", type=str, default="", help="Optional comma-separated CUDA_VISIBLE_DEVICES list (uses first).")
    ap.add_argument("--out_dir", type=Path, required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = _parse_thresholds(args.thresholds)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    spec = load_experiment(args.project, args.experiment)
    cfg = spec.config
    project_dir = Path(str(cfg["paths"]["project_dir"]))

    outputs_cfg = cfg.get("outputs", {}) or {}
    preprocess_base = _under_project(project_dir, outputs_cfg.get("preprocess_base", "stamp_preprocess"))
    preprocess_out = find_preprocess_output_dir(args.model, preprocess_base)
    if preprocess_out is None:
        raise RuntimeError(f"No existing preprocess output detected under {preprocess_base} for model {args.model}")
    input_feature_dir = preprocess_out
    print(f"Using existing feature_dir (unfiltered): {input_feature_dir}")

    cache_dir = Path(str(cfg.get("paths", {}).get("cache_dir", "/tmp/image_cache")))

    tile_filter_cfg = cfg.get("tile_filter", {}) or {}
    model_path = tile_filter_cfg.get("model_path", None)
    if not model_path:
        raise ValueError("Missing tile_filter.model_path in merged config; cannot run filtering sweep.")
    bundle = joblib.load(str(model_path))
    tile_model = bundle["model"]
    feature_names = list(bundle["features"])

    gpus = _parse_gpus(args.gpus)
    env = os.environ.copy()
    if gpus:
        env["CUDA_VISIBLE_DEVICES"] = gpus[0]
        print(f"Using CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")

    results_rows: list[dict] = []

    for thr in thresholds:
        thr_dir = args.out_dir / f"thr_{thr.tag}"
        thr_dir.mkdir(parents=True, exist_ok=True)

        filtered_feature_dir = thr_dir / "filtered_feature_dir" / args.model / input_feature_dir.name
        if not filtered_feature_dir.exists() or not list(filtered_feature_dir.glob("*.h5")):
            print(f"\n=== Threshold {thr.value} (build filtered features) ===")
            _filter_feature_dir(
                input_dir=input_feature_dir,
                output_dir=filtered_feature_dir,
                cache_dir=cache_dir,
                threshold=thr.value,
                model=tile_model,
                feature_names=feature_names,
            )
        else:
            print(f"\n=== Threshold {thr.value} (reuse filtered features) ===")

        crossval_out_dir = thr_dir / "stamp_crossval" / args.model
        analysis_out_dir = thr_dir / "analysis" / args.model
        fusion_out_dir = analysis_out_dir / "fusion"

        stamp_cfg = build_stamp_config(cfg, args.model, run_dir=thr_dir)
        stamp_cfg["crossval"]["feature_dir"] = str(filtered_feature_dir)
        stamp_cfg["crossval"]["output_dir"] = str(crossval_out_dir)

        cfg_path = thr_dir / f"stamp_config_{args.model}.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with cfg_path.open("w") as f:
            yaml.safe_dump(stamp_cfg, f, sort_keys=False)

        print(f"\n=== Threshold {thr.value} (STAMP crossval) ===")
        _run(["stamp", "--config", str(cfg_path), "crossval"], env=env, cwd=Path.cwd())

        print(f"\n=== Threshold {thr.value} (analyze) ===")
        _run(
            [
                sys.executable,
                "scripts/analyze_stamp_cv.py",
                "--cv_root",
                str(crossval_out_dir),
                "--model_name",
                str(args.model),
                "--out_dir",
                str(analysis_out_dir),
            ],
            env=env,
            cwd=Path.cwd(),
        )

        predictions_csv = analysis_out_dir / f"all_predictions_{args.model}.csv"

        print(f"\n=== Threshold {thr.value} (fusion) ===")
        _run(
            [
                sys.executable,
                "scripts/evaluate_fusion.py",
                "--project",
                str(args.project),
                "--predictions",
                str(predictions_csv),
                "--out_dir",
                str(fusion_out_dir),
            ],
            env=env,
            cwd=Path.cwd(),
        )

        metrics_csv = fusion_out_dir / "fusion_metrics.csv"
        if not metrics_csv.exists():
            raise FileNotFoundError(f"Missing expected fusion metrics: {metrics_csv}")

        mdf = pd.read_csv(metrics_csv)
        wsi_row = mdf[mdf["method"] == "WSI"]
        fusion_row = mdf[mdf["method"] == "Fusion"]
        if wsi_row.empty or fusion_row.empty:
            raise ValueError(f"Missing WSI/Fusion rows in {metrics_csv}")

        wsi_auc = float(wsi_row["roc_auc"].iloc[0])
        fusion_auc = float(fusion_row["roc_auc"].iloc[0])
        fusion_pr_auc = float(fusion_row["pr_auc"].iloc[0])

        results_rows.append(
            {
                "threshold": float(thr.value),
                "wsi_auc": wsi_auc,
                "fusion_auc": fusion_auc,
                "fusion_pr_auc": fusion_pr_auc,
            }
        )

    results_df = pd.DataFrame(results_rows).sort_values("threshold").reset_index(drop=True)
    out_csv = args.out_dir / "filter_sweep_results.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()

