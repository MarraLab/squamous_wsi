#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import subprocess
import shlex
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed


from wsi_recurrence.experiment import (
    build_manifest,
    create_run_dir,
    dump_yaml,
    load_experiment,
)
from wsi_recurrence.clinical import fusion_enabled, validate_fusion_config
from wsi_recurrence.stamp_runner import (
    build_stamp_config,
    find_preprocess_output_dir,
    find_slide_encoding_output_dir,
    slide_encoding_output_dir,
    update_stamp_config_feature_dir,
    update_stamp_config_slide_encoding,
    write_stamp_configs,
)
from wsi_recurrence.validation import validate_predictions_complete
from wsi_recurrence.slide_encoding import parse_slide_encoding_config, validate_slide_encoding_pairing


def _as_path(value) -> Path:
    return Path(str(value))


def _under_project(project_dir: Path, path_like) -> Path:
    p = _as_path(path_like)
    return p if p.is_absolute() else (project_dir / p)

def _parse_csv_list(val: str) -> list[str]:
    return [x.strip() for x in str(val).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--project",
        type=Path,
        default=Path("configs/project_lusc.yaml"),
        help="Project YAML under wsi_recurrence/ (paths, defaults).",
    )
    ap.add_argument(
        "--experiment",
        type=Path,
        default=Path("configs/experiments/example_stamp_qc03.yaml"),
        help="Experiment YAML under wsi_recurrence/ (models, overrides).",
    )
    ap.add_argument(
        "--out_root",
        type=Path,
        default=Path("outputs/runs"),
        help="Run output root (under wsi_recurrence/).",
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands only (default).",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Actually run preprocess/crossval commands.",
    )
    ap.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated subset of models to process (e.g. ctranspath,virchow-full).",
    )
    ap.add_argument(
        "--gpus",
        type=str,
        default="0,1",
        help="Comma-separated GPU ids for CUDA_VISIBLE_DEVICES (e.g. 0,1).",
    )
    ap.add_argument(
        "--parallel-models",
        action="store_true",
        help="Run selected models in parallel across GPUs (requires --execute unless --dry-run).",
    )
    pp = ap.add_mutually_exclusive_group()
    pp.add_argument(
        "--run-preprocess",
        action="store_true",
        help="Always run STAMP preprocess before crossval.",
    )
    pp.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Never run STAMP preprocess; require an existing detected feature_dir.",
    )
    ap.add_argument(
        "--skip-crossval",
        action="store_true",
        help="Do not run STAMP crossval; require existing crossval outputs under the chosen cv_root.",
    )
    ap.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Shorthand for --skip-preprocess and --skip-crossval (reuse existing STAMP outputs).",
    )
    se = ap.add_mutually_exclusive_group()
    se.add_argument(
        "--skip-slide-encoding",
        action="store_true",
        help="Do not run STAMP encode_slides; require an existing detected slide-level feature_dir.",
    )
    se.add_argument(
        "--reuse-slide-encoding",
        action="store_true",
        help="If slide-encoding output is detected, skip encode_slides; otherwise run it.",
    )
    se.add_argument(
        "--run-slide-encoding",
        action="store_true",
        help=(
            "Force STAMP encode_slides to run. Existing slide-encoding outputs for the selected "
            "model are removed first so downstream crossval uses newly generated slide features."
        ),
    )
    ap.add_argument(
        "--existing-run-dir",
        type=Path,
        default=None,
        help=(
            "Existing STAMP crossval run directory containing per-model folders. "
            "When set, inferred per-model cv_root is <existing-run-dir>/<model_name>."
        ),
    )
    ap.add_argument(
        "--analyze",
        action="store_true",
        help="After crossval, run analyze_stamp_cv.py into the run's analysis folder.",
    )
    ap.add_argument(
        "--fusion",
        action="store_true",
        help="After analysis, run evaluate_fusion.py into the run's analysis folder.",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="After fusion, run plot_results.py into the run's analysis folder.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    dry_run = True if args.dry_run else (not args.execute)
    if args.reuse_existing:
        args.skip_preprocess = True
        args.skip_crossval = True
    if args.existing_run_dir is not None and (not args.skip_crossval):
        raise ValueError("--existing-run-dir is only supported with --skip-crossval/--reuse-existing.")
    if args.fusion and not args.analyze:
        raise ValueError("--fusion requires --analyze (fusion needs model predictions from analysis).")
    if args.plot and not args.analyze:
        raise ValueError("--plot requires --analyze.")

    spec = load_experiment(args.project, args.experiment)
    try:
        run_fusion_cfg = fusion_enabled(spec.config)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if run_fusion_cfg:
        validate_fusion_config(spec.config, project_path=args.project)
    if args.fusion and (not run_fusion_cfg):
        raise ValueError(
            f"Project config disables fusion ({args.project}); remove --fusion or set analysis.run_fusion=true."
        )
    run_dir = create_run_dir(args.out_root, spec.name)

    manifest = build_manifest(spec, run_dir, cli_argv=sys.argv)
    dump_yaml(manifest, run_dir / "manifest.yaml")

    config_paths = write_stamp_configs(spec.config, spec.models(), run_dir=run_dir)
    print(f"Run dir: {run_dir}")
    print(f"Manifest: {run_dir / 'manifest.yaml'}")

    merged = spec.config
    se_cfg = parse_slide_encoding_config(merged)
    if se_cfg is not None and se_cfg.enabled:
        try:
            validate_slide_encoding_pairing(se_cfg)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    project_dir = _as_path(merged["paths"]["project_dir"])
    preprocess_base = _under_project(project_dir, merged.get("outputs", {}).get("preprocess_base", "stamp_preprocess"))
    crossval_runs_base = project_dir / "stamp_crossval_runs" / run_dir.name

    tile_filter_cfg = merged.get("tile_filter", {}) or {}
    tile_filter_enabled = bool(tile_filter_cfg.get("enabled", False))
    filtered_preprocess_base = _under_project(
        project_dir,
        tile_filter_cfg.get("filtered_preprocess_base", project_dir / "stamp_preprocess_filtered"),
    )

    all_models = spec.models()
    if args.models.strip():
        wanted = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = sorted(set(wanted) - set(all_models))
        if unknown:
            raise ValueError(f"Unknown model(s) not in experiment config: {unknown}")
        models = wanted
    else:
        models = all_models

    if args.parallel_models and (not args.execute) and (not dry_run):
        raise ValueError("--parallel-models requires --execute (or use --dry-run to preview scheduling).")

    header = "Planned STAMP workflow (dry-run):" if dry_run else "Planned STAMP workflow (execute):"
    print(f"\n{header}")
    if args.models.strip():
        print(f"Selected models (CLI): {', '.join(models)}")
    else:
        print(f"Selected models (experiment config): {', '.join(models)}")
        if args.execute:
            print(f"Executing all models from experiment config: {', '.join(models)}")
    clinical_cfg = merged.get("clinical", {}) or {}
    clinical_id_col = clinical_cfg.get("id_col", None)
    clinical_stage_col = clinical_cfg.get("stage_col", None)

    gpus = _parse_csv_list(args.gpus)
    if args.parallel_models and not gpus:
        raise ValueError("--parallel-models requires at least one GPU id via --gpus.")
    if not gpus:
        gpus = [""]  # sentinel: no CUDA_VISIBLE_DEVICES override

    model_to_gpu: dict[str, str] = {m: gpus[i % len(gpus)] for i, m in enumerate(models)}

    def run_one_model(model_name: str) -> None:
        gpu_id = model_to_gpu.get(model_name, "")
        gpu_label = gpu_id if gpu_id != "" else "default"
        prefix = f"[{model_name} | GPU {gpu_label}]"

        cfg_path = config_paths[model_name]
        preprocess_cmd = ["stamp", "--config", str(cfg_path), "preprocess"]
        encode_slides_cmd = ["stamp", "--config", str(cfg_path), "encode_slides"]
        crossval_cmd = ["stamp", "--config", str(cfg_path), "crossval"]
        crossval_out_dir = crossval_runs_base / model_name
        analysis_out_dir = run_dir / "analysis" / model_name
        cv_root = (args.existing_run_dir / model_name) if args.existing_run_dir is not None else crossval_out_dir
        analyze_cmd = [
            sys.executable,
            "scripts/analyze_stamp_cv.py",
            "--project",
            str(args.project),
            "--cv_root",
            str(cv_root),
            "--model_name",
            str(model_name),
            "--out_dir",
            str(analysis_out_dir),
        ]
        predictions_csv = analysis_out_dir / f"all_predictions_{model_name}.csv"
        fusion_out_dir = analysis_out_dir / "fusion"
        fusion_predictions_csv = fusion_out_dir / "fusion_predictions.csv"
        figures_out_dir = analysis_out_dir / "figures"
        fusion_cmd = [
            sys.executable,
            "scripts/evaluate_fusion.py",
            "--project",
            str(args.project),
            "--predictions",
            str(predictions_csv),
            "--out_dir",
            str(fusion_out_dir),
        ]
        if clinical_id_col:
            fusion_cmd += ["--clinical_id_col", str(clinical_id_col)]
        if clinical_stage_col:
            fusion_cmd += ["--clinical_stage_col", str(clinical_stage_col)]

        if run_fusion_cfg:
            plot_cmd = [
                sys.executable,
                "scripts/plot_results.py",
                "--project",
                str(args.project),
                "--predictions",
                str(fusion_predictions_csv),
                "--out_dir",
                str(figures_out_dir),
            ]
        else:
            plot_cmd = [
                sys.executable,
                "scripts/plot_results.py",
                "--predictions",
                str(predictions_csv),
                "--project",
                str(args.project),
                "--out_dir",
                str(figures_out_dir),
            ]

        env = os.environ.copy()
        if gpu_id != "":
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        print(f"\n{prefix}")
        print(f"{prefix} 1) preprocess: {shlex.join(preprocess_cmd)}")
        print(f"{prefix}    crossval.output_dir: {crossval_out_dir}")
        if args.skip_crossval:
            print(f"{prefix}    crossval decision: SKIP (flag set)")
        if args.existing_run_dir is not None:
            print(f"{prefix}    existing cv_root: {cv_root}")

        if args.skip_crossval:
            if not cv_root.exists():
                raise SystemExit(f"Requested --skip-crossval but CV root not found for model {model_name}: {cv_root}")
            if args.skip_preprocess:
                print(f"{prefix}    preprocess decision: SKIP (flag set)")
            elif args.run_preprocess:
                print(f"{prefix}    preprocess decision: SKIP (crossval skipped; preprocess not needed)")
            else:
                print(f"{prefix}    preprocess decision: SKIP (crossval skipped; preprocess not needed)")
            if args.analyze:
                print(f"{prefix} 5) analyze: {shlex.join(analyze_cmd)}")
                if not dry_run:
                    analysis_out_dir.mkdir(parents=True, exist_ok=True)
                    subprocess.run(analyze_cmd, check=True)
                    try:
                        _ = validate_predictions_complete(predictions_csv, merged)
                    except Exception as exc:
                        raise SystemExit(
                            f"{prefix} Prediction completeness check failed for {predictions_csv}:\n{exc}"
                        ) from exc
                if args.fusion and run_fusion_cfg:
                    print(f"{prefix} 6) fusion: {shlex.join(fusion_cmd)}")
                    if not dry_run:
                        fusion_out_dir.mkdir(parents=True, exist_ok=True)
                        subprocess.run(fusion_cmd, check=True)
                if args.plot:
                    print(f"{prefix} 7) plot: {shlex.join(plot_cmd)}")
                    if not dry_run:
                        figures_out_dir.mkdir(parents=True, exist_ok=True)
                        subprocess.run(plot_cmd, check=True)
            else:
                print(f"{prefix} analyze/fusion/plot: (not requested)")
            return

        # -----------------------------
        # Optional slide-level encoding
        # -----------------------------
        if se_cfg is not None and se_cfg.enabled:
            expected_key = se_cfg.model_key
            if model_name != expected_key:
                raise SystemExit(
                    f"{prefix} slide_encoding is enabled (encoder={se_cfg.encoder}, feat_model={se_cfg.feat_model}, "
                    f"agg_feat_model={se_cfg.agg_feat_model}), but selected model is {model_name!r}. "
                    f"Expected model key: {expected_key!r}."
                )

            try:
                def _ensure_tile_features(tile_model: str) -> Path | None:
                    tile_cfg_path = run_dir / "configs" / "stamp" / f"config_preprocess_{tile_model}.yaml"
                    if not tile_cfg_path.exists():
                        tile_cfg = build_stamp_config(merged, tile_model, run_dir=run_dir)
                        dump_yaml(tile_cfg, tile_cfg_path)
                    cmd = ["stamp", "--config", str(tile_cfg_path), "preprocess"]

                    out = find_preprocess_output_dir(tile_model, preprocess_base)
                    if out is None and args.skip_preprocess:
                        raise ValueError(
                            f"{tile_model}: --skip-preprocess set but no feature_dir detected under {preprocess_base}"
                        )
                    should_run = False
                    if args.run_preprocess:
                        should_run = True
                    elif args.skip_preprocess:
                        should_run = False
                    else:
                        should_run = out is None

                    print(f"{prefix} 2) tile features ({tile_model}): {str(out) if out is not None else '(none yet)'}")
                    if should_run:
                        if dry_run:
                            print(f"{prefix}    preprocess({tile_model}) decision: WILL RUN (dry-run; not executing)")
                        else:
                            print(f"{prefix}    preprocess({tile_model}) decision: running preprocess now...")
                            subprocess.run(cmd, check=True, env=env)
                        out = find_preprocess_output_dir(tile_model, preprocess_base)
                        if out is None:
                            if dry_run:
                                print(f"{prefix}    preprocess({tile_model}) result: feature_dir not detected yet (dry-run)")
                            else:
                                raise RuntimeError(
                                    f"{tile_model}: preprocess finished but no output dir found under {preprocess_base}"
                                )
                    else:
                        print(f"{prefix}    preprocess({tile_model}) decision: SKIP")

                    if out is None:
                        if dry_run and should_run:
                            return None
                        raise RuntimeError(f"{tile_model}: feature_dir not detected under {preprocess_base}")
                    return out

                feat_dir = _ensure_tile_features(se_cfg.feat_model)
                agg_feat_dir = None
                if se_cfg.agg_feat_model:
                    agg_feat_dir = _ensure_tile_features(se_cfg.agg_feat_model)
                elif se_cfg.encoder.strip().lower() == "eagle":
                    raise ValueError("slide_encoding.encoder=eagle requires slide_encoding.agg_feat_model.")
            except Exception as exc:
                raise SystemExit(f"{prefix} {exc}") from exc

            out_dir = slide_encoding_output_dir(se_cfg.output_base, model_key=model_name)
            if feat_dir is not None and (agg_feat_dir is not None or (not se_cfg.agg_feat_model)):
                update_stamp_config_slide_encoding(
                    cfg_path,
                    encoder=se_cfg.encoder,
                    feat_dir=feat_dir,
                    agg_feat_dir=agg_feat_dir,
                    output_dir=out_dir,
                    device=se_cfg.device,
                    generate_hash=se_cfg.generate_hash,
                )

            detected_slide_dir = find_slide_encoding_output_dir(out_dir, encoder=se_cfg.encoder)
            print(f"{prefix} 3) slide_encoding:")
            print(f"{prefix}    encoder: {se_cfg.encoder}")
            print(f"{prefix}    feat_model: {se_cfg.feat_model}")
            print(f"{prefix}    feat_dir: {feat_dir if feat_dir is not None else '(none yet)'}")
            if agg_feat_dir is not None:
                print(f"{prefix}    agg_feat_model: {se_cfg.agg_feat_model}")
                print(f"{prefix}    agg_feat_dir: {agg_feat_dir}")
            print(f"{prefix}    output_dir: {out_dir}")
            print(f"{prefix}    detected slide feature_dir: {str(detected_slide_dir) if detected_slide_dir else '(none yet)'}")

            should_run_encode = False
            if args.skip_slide_encoding:
                should_run_encode = False
                print(f"{prefix}    encode_slides decision: SKIP (flag set)")
            elif args.run_slide_encoding:
                should_run_encode = True
                print(f"{prefix}    encode_slides decision: WILL RUN (force flag set)")
            elif args.reuse_slide_encoding:
                should_run_encode = detected_slide_dir is None
                print(f"{prefix}    encode_slides decision: {'WILL RUN' if should_run_encode else 'SKIP'} (reuse-slide-encoding)")
            else:
                should_run_encode = detected_slide_dir is None
                print(f"{prefix}    encode_slides decision: {'WILL RUN' if should_run_encode else 'SKIP'}")

            if should_run_encode:
                if dry_run:
                    if args.run_slide_encoding and out_dir.exists():
                        print(f"{prefix}    remove existing slide_encoding output: {out_dir} (dry-run; not removing)")
                    print(f"{prefix} 4) encode_slides: {shlex.join(encode_slides_cmd)}")
                else:
                    if args.run_slide_encoding and out_dir.exists():
                        print(f"{prefix}    removing existing slide_encoding output: {out_dir}")
                        shutil.rmtree(out_dir)
                    print(f"{prefix} 4) encode_slides: running now...")
                    subprocess.run(encode_slides_cmd, check=True, env=env)

            slide_out = find_slide_encoding_output_dir(out_dir, encoder=se_cfg.encoder)
            if slide_out is None:
                if dry_run and should_run_encode:
                    print(f"{prefix} 5) crossval.feature_dir (slide-level): (not ready yet; would be {out_dir}/<encoder>-slide-*/)")
                    print(f"{prefix} 6) crossval: {shlex.join(crossval_cmd)}")
                    print(f"{prefix}    crossval decision: NOT READY (encode_slides not executed in dry-run)")
                    if args.analyze:
                        print(f"{prefix} 7) analyze: (skipped; crossval not run)")
                        print(f"{prefix}    analyze command: {shlex.join(analyze_cmd)}")
                    return
                raise RuntimeError(
                    f"{prefix} slide encoding output not detected under {out_dir} "
                    f"(encoder={se_cfg.encoder}); use --execute to run encode_slides or provide existing output."
                )

            final_feature_dir = slide_out
            print(f"{prefix} 5) crossval.feature_dir (slide-level): {final_feature_dir}")
            update_stamp_config_feature_dir(cfg_path, final_feature_dir)

            print(f"{prefix} 6) crossval: {shlex.join(crossval_cmd)}")
            if dry_run:
                if args.analyze:
                    print(f"{prefix} 7) analyze: {shlex.join(analyze_cmd)}")
                    if args.fusion and run_fusion_cfg:
                        print(f"{prefix} 8) fusion: {shlex.join(fusion_cmd)}")
                    if args.plot:
                        print(f"{prefix} 9) plot: {shlex.join(plot_cmd)}")
                return

            subprocess.run(crossval_cmd, check=True, env=env)
            if args.analyze:
                print(f"{prefix} 7) analyze: {shlex.join(analyze_cmd)}")
                subprocess.run(analyze_cmd, check=True)
                try:
                    _ = validate_predictions_complete(predictions_csv, merged)
                except Exception as exc:
                    raise SystemExit(
                        f"{prefix} Prediction completeness check failed for {predictions_csv}:\n{exc}"
                    ) from exc
                if args.fusion and run_fusion_cfg:
                    print(f"{prefix} 8) fusion: {shlex.join(fusion_cmd)}")
                    subprocess.run(fusion_cmd, check=True)
                if args.plot:
                    print(f"{prefix} 9) plot: {shlex.join(plot_cmd)}")
                    subprocess.run(plot_cmd, check=True)
            return

        # -----------------------------
        # Standard tile-feature pipeline
        # -----------------------------
        preprocess_out = find_preprocess_output_dir(model_name, preprocess_base)
        detected_str = str(preprocess_out) if preprocess_out is not None else "(none yet)"
        print(f"{prefix} 2) detected feature_dir: {detected_str}")

        should_run_preprocess = False
        if args.run_preprocess:
            should_run_preprocess = True
        elif args.skip_preprocess:
            should_run_preprocess = False
        else:
            should_run_preprocess = preprocess_out is None

        if should_run_preprocess:
            if args.skip_preprocess:
                raise ValueError("--skip-preprocess conflicts with preprocess execution.")
            if dry_run:
                print(f"{prefix}    preprocess decision: WILL RUN (dry-run; not executing)")
            else:
                print(f"{prefix}    preprocess decision: running preprocess now...")
                subprocess.run(preprocess_cmd, check=True, env=env)
            preprocess_out = find_preprocess_output_dir(model_name, preprocess_base)
            if preprocess_out is None:
                if args.skip_preprocess or dry_run:
                    print(f"{prefix}    preprocess result: feature_dir still not detected")
                else:
                    raise RuntimeError(
                        f"{model_name}: preprocess finished but no output dir found under {preprocess_base}"
                    )
        else:
            print(f"{prefix}    preprocess decision: SKIP")

        if preprocess_out is None:
            if args.skip_preprocess:
                raise RuntimeError(f"{model_name}: --skip-preprocess set but no feature_dir detected under {preprocess_base}")
            print(f"{prefix} 3) filtered feature_dir: (n/a)")
            print(f"{prefix} 4) crossval: NOT READY (feature_dir not detected yet)")
            print(f"{prefix}    crossval command: {shlex.join(crossval_cmd)}")
            if args.analyze:
                print(f"{prefix} 5) analyze: (skipped; crossval not run)")
                print(f"{prefix}    analyze command: {shlex.join(analyze_cmd)}")
                if args.fusion:
                    print(f"{prefix} 6) fusion: (skipped; analysis not run)")
                    print(f"{prefix}    fusion command: {shlex.join(fusion_cmd)}")
                    if args.plot:
                        print(f"{prefix} 7) plot: (skipped; fusion not run)")
                        print(f"{prefix}    plot command: {shlex.join(plot_cmd)}")
            return

        print(f"{prefix} 2) detected feature_dir (final): {preprocess_out}")

        final_feature_dir: Path = preprocess_out
        if tile_filter_enabled:
            filtered_dir = filtered_preprocess_base / model_name / preprocess_out.name
            print(f"{prefix} 3) filtered feature_dir: {filtered_dir}")
            final_feature_dir = filtered_dir
        else:
            print(f"{prefix} 3) filtered feature_dir: (tile_filter disabled)")

        update_stamp_config_feature_dir(cfg_path, final_feature_dir)
        print(f"{prefix} 4) crossval: {shlex.join(crossval_cmd)}")
        print(f"{prefix}    using feature_dir: {final_feature_dir}")

        if not dry_run:
            subprocess.run(crossval_cmd, check=True, env=env)
            if args.analyze:
                print(f"{prefix} 5) analyze: {shlex.join(analyze_cmd)}")
                subprocess.run(analyze_cmd, check=True)
                try:
                    _ = validate_predictions_complete(predictions_csv, merged)
                except Exception as exc:
                    raise SystemExit(
                        f"{prefix} Prediction completeness check failed for {predictions_csv}:\n{exc}"
                    ) from exc
                if args.fusion and run_fusion_cfg:
                    print(f"{prefix} 6) fusion: {shlex.join(fusion_cmd)}")
                    subprocess.run(fusion_cmd, check=True)
                if args.plot:
                    print(f"{prefix} 7) plot: {shlex.join(plot_cmd)}")
                    subprocess.run(plot_cmd, check=True)
        else:
            if args.analyze:
                print(f"{prefix} 5) analyze: {shlex.join(analyze_cmd)}")
                if args.fusion and run_fusion_cfg:
                    print(f"{prefix} 6) fusion: {shlex.join(fusion_cmd)}")
                if args.plot:
                    print(f"{prefix} 7) plot: {shlex.join(plot_cmd)}")

    if args.parallel_models:
        print(f"\nScheduling {len(models)} model(s) across GPU(s): {', '.join(gpus)}")
        for m in models:
            gpu_id = model_to_gpu[m]
            gpu_label = gpu_id if gpu_id != "" else "default"
            print(f"  - {m} -> GPU {gpu_label}")

        if dry_run:
            for m in models:
                run_one_model(m)
        else:
            max_workers = len(gpus)
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = {ex.submit(run_one_model, m): m for m in models}
                for fut in as_completed(futs):
                    fut.result()
    else:
        for model_name in models:
            run_one_model(model_name)

if __name__ == "__main__":
    main()
