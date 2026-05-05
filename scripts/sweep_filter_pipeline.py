#!/usr/bin/env python
from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import os
import shlex
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from wsi_recurrence.clinical import fusion_enabled, validate_fusion_config
from wsi_recurrence.experiment import load_experiment
from wsi_recurrence.metrics import compute_auc, compute_pr_auc
from wsi_recurrence.stamp_runner import build_stamp_config, find_preprocess_output_dir
from wsi_recurrence.validation import validate_predictions_complete


_WORKER_MODEL = None
_WORKER_FEATURE_NAMES: list[str] = []


def _init_filter_worker(model, feature_names: list[str]) -> None:
    global _WORKER_MODEL, _WORKER_FEATURE_NAMES
    _WORKER_MODEL = model
    _WORKER_FEATURE_NAMES = list(feature_names)


def _filter_one_h5(
    *,
    h5_path: Path,
    out_h5: Path,
    cache_dir: Path,
    bad_prob_threshold: float,
    allow_extreme_filtering: bool,
) -> tuple[str, int, int]:
    """
    Worker unit: process exactly one slide (.h5) -> one output file.
    Returns (slide_id, n_total, n_kept).
    """
    from wsi_recurrence.tile_filter.h5_filter import filter_h5

    slide_id = h5_path.stem
    if out_h5.exists():
        return slide_id, 0, 0

    zip_path = _find_zip(cache_dir, slide_id)
    if zip_path is None:
        out_h5.parent.mkdir(parents=True, exist_ok=True)
        out_h5.write_bytes(h5_path.read_bytes())
        return slide_id, 0, 0

    if _WORKER_MODEL is None:
        raise RuntimeError(
            "Internal error: filter worker model is unset (expected _init_filter_worker to run)."
        )

    import h5py

    with h5py.File(h5_path, "r") as f:
        coords_um = f["coords"][:]

    keep_map = _compute_keep_map_for_h5_coords(
        zip_path=zip_path,
        coords_um=coords_um,
        bad_prob_threshold=bad_prob_threshold,
        model=_WORKER_MODEL,
        feature_names=_WORKER_FEATURE_NAMES,
    )
    n_total = int(len(coords_um))
    n_kept = int(sum(keep_map.values()))
    frac_kept = (n_kept / n_total) if n_total else 0.0
    if (frac_kept < 0.5) and (not allow_extreme_filtering):
        raise RuntimeError(
            f"{slide_id}: extreme filtering at bad_prob_threshold={bad_prob_threshold} "
            f"(kept {n_kept}/{n_total} = {frac_kept:.3f}). "
            "Pass --allow_extreme_filtering to override."
        )
    try:
        out_h5.parent.mkdir(parents=True, exist_ok=True)
        n_total, n_kept = filter_h5(h5_path, out_h5, keep_map)
        return slide_id, int(n_total), int(n_kept)
    except RuntimeError:
        out_h5.parent.mkdir(parents=True, exist_ok=True)
        out_h5.write_bytes(h5_path.read_bytes())
        return slide_id, 0, 0


@dataclass(frozen=True)
class ThresholdSpec:
    token: str
    value: float | None
    tag: str

    @property
    def is_raw(self) -> bool:
        return self.token == "raw"


def _under_project(project_dir: Path, path_like: str | Path) -> Path:
    p = Path(str(path_like))
    return p if p.is_absolute() else (project_dir / p)


def _parse_thresholds(s: str) -> list[ThresholdSpec]:
    out: list[ThresholdSpec] = []
    for raw in [x.strip() for x in str(s).split(",") if x.strip()]:
        if raw.lower() == "raw":
            out.append(ThresholdSpec(token="raw", value=None, tag="raw"))
            continue
        val = float(raw)
        tag = raw.replace(".", "p").replace("-", "m")
        out.append(ThresholdSpec(token=raw, value=val, tag=tag))
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
    from wsi_recurrence.tile_filter.h5_filter import ROUND_DECIMALS
    from wsi_recurrence.tile_filter.zip_tiles import try_parse_tile_coords

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
    bad_prob_threshold: float,
    model,
    feature_names: list[str],
    coord_tol_um: float = 2.0,
) -> dict[tuple[float, float], bool]:
    """
    Build a keep_map keyed by (rounded x_um, rounded y_um) for coords in an .h5.
    keep = (bad_prob < bad_prob_threshold), matching build_tile_keep_masks.py behavior.
    """
    from PIL import Image
    from wsi_recurrence.tile_filter.features import compute_features
    from wsi_recurrence.tile_filter.h5_filter import ROUND_DECIMALS
    from wsi_recurrence.tile_filter.zip_tiles import build_zip_coord_index, match_tile_name

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
        keep_arr = bad_prob < float(bad_prob_threshold)
        keep_map = {k: bool(v) for k, v in zip(feat_keys, keep_arr)}

        # Prevent "all removed" failures: keep the single best tile if needed.
        if len(keep_map) > 0 and (not any(keep_map.values())):
            i_best = int(np.argmin(bad_prob))
            keep_map[feat_keys[i_best]] = True
            print(
                "WARNING: bad_prob_threshold would remove all tiles; keeping 1 tile for stability "
                f"(bad_prob_threshold={bad_prob_threshold})"
            )
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
    bad_prob_threshold: float,
    model,
    feature_names: list[str],
    allow_extreme_filtering: bool,
    workers: int = 1,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_files = sorted(input_dir.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found under {input_dir}")

    workers = int(workers)
    if workers < 1:
        raise ValueError("--workers must be >= 1")

    todo: list[tuple[Path, Path]] = []
    for h5_path in h5_files:
        out_h5 = output_dir / h5_path.name
        if not out_h5.exists():
            todo.append((h5_path, out_h5))

    if not todo:
        pass
    elif workers == 1:
        _init_filter_worker(model, feature_names)
        for h5_path, out_h5 in todo:
            slide_id, n_total, n_kept = _filter_one_h5(
                h5_path=h5_path,
                out_h5=out_h5,
                cache_dir=cache_dir,
                bad_prob_threshold=bad_prob_threshold,
                allow_extreme_filtering=allow_extreme_filtering,
            )
            if n_total == 0 and n_kept == 0:
                if _find_zip(cache_dir, slide_id) is None:
                    print(f"WARNING: no zip found for {slide_id}; copying unfiltered")
                else:
                    print(f"WARNING: {slide_id}: filter_h5 failed; copying unfiltered")
            else:
                print(f"{slide_id}: kept {n_kept}/{n_total} (bad_prob_threshold={bad_prob_threshold})")
    else:
        print(f"Filtering {len(todo)} slide(s) with workers={workers}")
        done = 0
        with cf.ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_filter_worker,
            initargs=(model, feature_names),
        ) as ex:
            fut_to_slide: dict[cf.Future, str] = {}
            for h5_path, out_h5 in todo:
                slide_id = h5_path.stem
                fut = ex.submit(
                    _filter_one_h5,
                    h5_path=h5_path,
                    out_h5=out_h5,
                    cache_dir=cache_dir,
                    bad_prob_threshold=bad_prob_threshold,
                    allow_extreme_filtering=allow_extreme_filtering,
                )
                fut_to_slide[fut] = slide_id

            for fut in cf.as_completed(fut_to_slide):
                slide_id = fut_to_slide[fut]
                try:
                    slide_id_out, n_total, n_kept = fut.result()
                except Exception as exc:
                    raise RuntimeError(f"{slide_id}: filtering failed") from exc
                done += 1
                if n_total == 0 and n_kept == 0:
                    print(f"[{done}/{len(todo)}] {slide_id_out}: copied (unfiltered)")
                else:
                    print(f"[{done}/{len(todo)}] {slide_id_out}: kept {n_kept}/{n_total}")

    for extra in input_dir.iterdir():
        if extra.suffix != ".h5" and extra.is_file():
            dst = output_dir / extra.name
            if not dst.exists():
                dst.write_bytes(extra.read_bytes())


def _infer_pred_col(df: pd.DataFrame) -> str:
    if "pred" in df.columns:
        return "pred"
    pred_cols = [c for c in df.columns if str(c).startswith("pred_")]
    if len(pred_cols) == 1:
        return pred_cols[0]
    raise ValueError("Could not infer prediction column (expected 'pred' or a single 'pred_*').")


def _compute_wsi_metrics(pred_csv: Path, *, label_col: str) -> tuple[float, float]:
    df = pd.read_csv(pred_csv)
    pred_col = _infer_pred_col(df)
    if label_col not in df.columns:
        raise ValueError(f"Missing label column {label_col!r} in {pred_csv}")

    y_true = pd.to_numeric(df[label_col], errors="coerce")
    y_pred = pd.to_numeric(df[pred_col], errors="coerce")
    keep = y_true.notna() & y_pred.notna()
    if int(keep.sum()) == 0:
        raise ValueError(f"No valid rows to score in {pred_csv} (after dropping NaNs).")

    y_true_arr = y_true[keep].astype(int).values
    y_pred_arr = y_pred[keep].astype(float).values
    return float(compute_auc(y_true_arr, y_pred_arr)), float(compute_pr_auc(y_true_arr, y_pred_arr))


def _run(
    cmd: list[str],
    *,
    dry_run: bool,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> None:
    if dry_run:
        print(shlex.join(cmd))
        return
    subprocess.run(cmd, check=True, env=env, cwd=str(cwd) if cwd is not None else None)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, default=Path("configs/project_lusc.yaml"))
    ap.add_argument("--experiment", type=Path, default=Path("configs/experiments/example_stamp_qc03.yaml"))
    ap.add_argument("--model", type=str, required=True, help="Model name (e.g. ctranspath)")
    ap.add_argument(
        "--thresholds",
        type=str,
        required=True,
        help="Comma-separated thresholds: 'raw' or numeric bad_prob_threshold values (e.g. raw,0.2,0.3).",
    )
    ap.add_argument("--gpus", type=str, default="", help="Optional comma-separated CUDA_VISIBLE_DEVICES list (uses first).")
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="CPU workers for per-slide tile filtering (default: 1).",
    )
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow_extreme_filtering", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = _parse_thresholds(args.thresholds)
    if not args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"CPU workers: {int(args.workers)}")

    spec = load_experiment(args.project, args.experiment)
    cfg = spec.config
    project_dir = Path(str(cfg["paths"]["project_dir"]))
    try:
        run_fusion = fusion_enabled(cfg)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if run_fusion:
        validate_fusion_config(cfg, project_path=args.project)
    else:
        print("Fusion disabled in project config; skipping clinical fusion step.")

    outputs_cfg = cfg.get("outputs", {}) or {}
    preprocess_base = _under_project(project_dir, outputs_cfg.get("preprocess_base", "stamp_preprocess"))
    preprocess_out = find_preprocess_output_dir(args.model, preprocess_base)
    if preprocess_out is None:
        if args.dry_run:
            input_feature_dir = preprocess_base / args.model / "FEATURE_DIR_PLACEHOLDER"
            print(
                f"WARNING: No existing preprocess output detected under {preprocess_base} for model {args.model}. "
                "Using a placeholder feature_dir for dry-run."
            )
        else:
            raise RuntimeError(f"No existing preprocess output detected under {preprocess_base} for model {args.model}")
    else:
        input_feature_dir = preprocess_out
    print(f"Input feature_dir (unfiltered): {input_feature_dir}")

    cache_dir = Path(str(cfg.get("paths", {}).get("cache_dir", "/tmp/image_cache")))

    tile_model = None
    feature_names: list[str] = []
    tile_filter_cfg = cfg.get("tile_filter", {}) or {}
    model_path = tile_filter_cfg.get("model_path", None)
    if not model_path:
        if args.dry_run:
            print("WARNING: Missing tile_filter.model_path in merged config; filtering step will be a no-op in dry-run.")
        else:
            raise ValueError("Missing tile_filter.model_path in merged config; cannot run filtering sweep.")
    if not args.dry_run:
        import joblib

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
        if not args.dry_run:
            thr_dir.mkdir(parents=True, exist_ok=True)

        if thr.is_raw:
            filtered_feature_dir = input_feature_dir
        else:
            filtered_feature_dir = thr_dir / "filtered_feature_dir" / args.model / input_feature_dir.name

        print("\n---")
        if thr.is_raw:
            print("bad_prob_threshold: raw (unfiltered baseline)")
            print("rule: raw = no tile filtering (use original feature_dir)")
        else:
            print(f"bad_prob_threshold: {thr.value}")
            print("rule: keep tiles where bad_prob < bad_prob_threshold")
        print(f"output directory: {thr_dir}")
        print(f"input feature_dir: {input_feature_dir}")
        print(f"filtered feature_dir: {filtered_feature_dir}")

        if args.dry_run:
            if thr.is_raw:
                print("(dry-run) Raw mode: would not build filtered features")
            else:
                print(f"(dry-run) Would build filtered features at: {filtered_feature_dir} (workers={int(args.workers)})")
        else:
            if thr.is_raw:
                pass
            else:
                if not filtered_feature_dir.exists() or not list(filtered_feature_dir.glob("*.h5")):
                    print(f"\n=== bad_prob_threshold {thr.value} (build filtered features) ===")
                    _filter_feature_dir(
                        input_dir=input_feature_dir,
                        output_dir=filtered_feature_dir,
                        cache_dir=cache_dir,
                        bad_prob_threshold=float(thr.value),
                        model=tile_model,
                        feature_names=feature_names,
                        allow_extreme_filtering=bool(args.allow_extreme_filtering),
                        workers=int(args.workers),
                    )
                else:
                    print(f"\n=== bad_prob_threshold {thr.value} (reuse filtered features) ===")

        crossval_out_dir = thr_dir / "stamp_crossval" / args.model
        analysis_out_dir = thr_dir / "analysis" / args.model
        fusion_out_dir = analysis_out_dir / "fusion"

        stamp_cfg = build_stamp_config(cfg, args.model, run_dir=thr_dir)
        stamp_cfg["crossval"]["feature_dir"] = str(filtered_feature_dir)
        stamp_cfg["crossval"]["output_dir"] = str(crossval_out_dir)

        cfg_path = thr_dir / f"stamp_config_{args.model}.yaml"
        if args.dry_run:
            print(f"(dry-run) Would write STAMP config: {cfg_path}")
        else:
            import yaml

            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            with cfg_path.open("w") as f:
                yaml.safe_dump(stamp_cfg, f, sort_keys=False)

        label = "raw" if thr.is_raw else str(thr.value)
        print(f"\n=== bad_prob_threshold {label} (STAMP crossval) ===")
        _run(
            ["stamp", "--config", str(cfg_path), "crossval"],
            dry_run=args.dry_run,
            env=env,
            cwd=Path.cwd(),
        )

        print(f"\n=== bad_prob_threshold {label} (analyze) ===")
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
            dry_run=args.dry_run,
            env=env,
            cwd=Path.cwd(),
        )

        predictions_csv = analysis_out_dir / f"all_predictions_{args.model}.csv"

        cohort_stats = None
        if not args.dry_run:
            try:
                cohort_stats = validate_predictions_complete(predictions_csv, cfg)
            except Exception as exc:
                threshold_label = "raw" if thr.is_raw else str(thr.value)
                msg = (
                    f"Threshold run failed validation (threshold={threshold_label}) for predictions CSV:\n"
                    f"  {predictions_csv}\n"
                    f"{exc}"
                )
                raise SystemExit(msg) from exc

        if run_fusion:
            print(f"\n=== bad_prob_threshold {label} (fusion) ===")
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
                dry_run=args.dry_run,
                env=env,
                cwd=Path.cwd(),
            )
        else:
            print(f"\n=== bad_prob_threshold {label} (fusion skipped) ===")

        if args.dry_run:
            continue

        if run_fusion:
            metrics_csv = fusion_out_dir / "fusion_metrics.csv"
            if not metrics_csv.exists():
                raise FileNotFoundError(f"Missing expected fusion metrics: {metrics_csv}")

            mdf = pd.read_csv(metrics_csv)
            wsi_row = mdf[mdf["method"] == "WSI"]
            fusion_row = mdf[mdf["method"] == "Fusion"]
            if wsi_row.empty or fusion_row.empty:
                raise ValueError(f"Missing WSI/Fusion rows in {metrics_csv}")

            wsi_auc = float(wsi_row["roc_auc"].iloc[0])
            wsi_pr_auc = float(wsi_row["pr_auc"].iloc[0]) if "pr_auc" in wsi_row.columns else float("nan")
            fusion_auc = float(fusion_row["roc_auc"].iloc[0])
            fusion_pr_auc = float(fusion_row["pr_auc"].iloc[0])
        else:
            label_col = str((cfg.get("columns", {}) or {}).get("label", "recur"))
            wsi_auc, wsi_pr_auc = _compute_wsi_metrics(predictions_csv, label_col=label_col)
            fusion_auc = float("nan")
            fusion_pr_auc = float("nan")

        threshold_label = "raw" if thr.is_raw else str(thr.value)
        bad_prob_threshold_num = float("nan") if thr.is_raw else float(thr.value)
        results_rows.append(
            {
                "threshold": threshold_label,
                "bad_prob_threshold": bad_prob_threshold_num,
                "n_patients": int(cohort_stats.n_patients) if cohort_stats is not None else float("nan"),
                "n_positive": int(cohort_stats.n_positive) if cohort_stats is not None else float("nan"),
                "n_negative": int(cohort_stats.n_negative) if cohort_stats is not None else float("nan"),
                "wsi_auc": wsi_auc,
                "wsi_pr_auc": wsi_pr_auc,
                "fusion_auc": fusion_auc,
                "fusion_pr_auc": fusion_pr_auc,
            }
        )

    out_csv = args.out_dir / "filter_sweep_results.csv"
    if args.dry_run:
        print(f"\n(dry-run) Would save: {out_csv}")
        return

    results_df = pd.DataFrame(results_rows)
    results_df["__sort_key"] = results_df["bad_prob_threshold"].fillna(-1.0)
    results_df = results_df.sort_values("__sort_key").drop(columns=["__sort_key"]).reset_index(drop=True)
    if results_df.empty:
        raise RuntimeError("No sweep results recorded (unexpected).")
    else:
        results_df.to_csv(out_csv, index=False)
        print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
