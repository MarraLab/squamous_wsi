#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import subprocess
import shlex
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


from wsi_recurrence.experiment import (
    build_manifest,
    create_run_dir,
    dump_yaml,
    load_experiment,
)
from wsi_recurrence.stamp_runner import (
    find_preprocess_output_dir,
    update_stamp_config_feature_dir,
    write_stamp_configs,
)


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
    if args.fusion and not args.analyze:
        raise ValueError("--fusion requires --analyze (fusion needs model predictions from analysis).")
    if args.plot and not args.fusion:
        raise ValueError("--plot requires --fusion.")

    spec = load_experiment(args.project, args.experiment)
    run_dir = create_run_dir(args.out_root, spec.name)

    manifest = build_manifest(spec, run_dir)
    dump_yaml(manifest, run_dir / "manifest.yaml")

    config_paths = write_stamp_configs(spec.config, spec.models(), run_dir=run_dir)
    print(f"Run dir: {run_dir}")
    print(f"Manifest: {run_dir / 'manifest.yaml'}")

    merged = spec.config
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

    if args.execute and not args.models.strip():
        raise ValueError("Refusing to --execute for all models by default. Pass --models.")
    if args.parallel_models and (not args.execute) and (not dry_run):
        raise ValueError("--parallel-models requires --execute (or use --dry-run to preview scheduling).")

    header = "Planned STAMP workflow (dry-run):" if dry_run else "Planned STAMP workflow (execute):"
    print(f"\n{header}")
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
        crossval_cmd = ["stamp", "--config", str(cfg_path), "crossval"]
        crossval_out_dir = crossval_runs_base / model_name
        analysis_out_dir = run_dir / "analysis" / model_name
        analyze_cmd = [
            sys.executable,
            "scripts/analyze_stamp_cv.py",
            "--cv_root",
            str(crossval_out_dir),
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

        plot_cmd = [
            sys.executable,
            "scripts/plot_results.py",
            "--fusion_predictions",
            str(fusion_predictions_csv),
            "--out_dir",
            str(figures_out_dir),
        ]

        env = os.environ.copy()
        if gpu_id != "":
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        print(f"\n{prefix}")
        print(f"{prefix} 1) preprocess: {shlex.join(preprocess_cmd)}")
        print(f"{prefix}    crossval.output_dir: {crossval_out_dir}")

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
            print(f"{prefix} 4) crossval: NOT READY (feature_dir still /tmp/)")
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
                if args.fusion:
                    print(f"{prefix} 6) fusion: {shlex.join(fusion_cmd)}")
                    subprocess.run(fusion_cmd, check=True)
                    if args.plot:
                        print(f"{prefix} 7) plot: {shlex.join(plot_cmd)}")
                        subprocess.run(plot_cmd, check=True)
        else:
            if args.analyze:
                print(f"{prefix} 5) analyze: {shlex.join(analyze_cmd)}")
                if args.fusion:
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
