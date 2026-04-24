# wsi_recurrence

Modular STAMP/WSI recurrence pipeline (refactor of `CLAM/run_stamp_pipeline.py` behavior; `CLAM/` is left untouched).

## Purpose
- Run STAMP preprocessing + cross-validation (optionally with tile QC filtering).
- Analyze CV outputs and produce per-model predictions.
- Fit/evaluate fusion models (WSI-only, clinical-only, fusion) and generate ROC/PR (and optional KM if `lifelines` is installed and time columns exist).

## Setup
From `wsi_recurrence/`:
- `pip install -e .`

## Config files
- `configs/project_lusc.yaml`: cohort-specific paths and column mappings
- `configs/experiments/example_stamp_qc03.yaml`: experiment definition (models, tile-filter settings)

## Clinical config (project YAML)
This pipeline distinguishes:
- `paths.stamp_table`: the table STAMP uses for `clini_table`/`slide_table` (often a minimal table with `patient`, `filename`, labels, etc.)
- `paths.clinical_features_table`: the richer clinical table used for fusion + time-to-event derivation

Key fields under `clinical:`:
- `clinical.id_col`: clinical table ID column to merge with predictions (mapped to prediction `patient` by default)
- `clinical.stage_col`: clinical numeric stage feature (used in clinical-only + fusion LR)
- `clinical.date_surgery_col`: surgery date column (optional; used to derive `time_to_event`)
- `clinical.date_followup_col`: followup/recurrence date column (optional)
- `clinical.event_col`: event indicator column (optional)
- `clinical.event_positive_value`: value in `event_col` treated as an event (default `1`)
- `clinical.drop_missing_time`: if `true`, drop rows missing `time_to_event/event` from written `fusion_predictions.csv` (fusion fit still runs)

## Running experiments
Dry-run (prints commands only):
- `python scripts/run_experiment.py --dry-run --models ctranspath --analyze --fusion --plot`

One-model execute:
- `python scripts/run_experiment.py --execute --models ctranspath --analyze --fusion --plot`

Notes:
- `--execute` is intentionally gated behind `--models` to avoid launching all models accidentally.
- Preprocess control:
  - default: preprocess only if no hash-named output dir is detected
  - `--run-preprocess`: always preprocess
  - `--skip-preprocess`: require existing preprocess outputs

## Output structure
Runs are written under:
- `outputs/runs/<run_id>/`

Per model:
- `outputs/runs/<run_id>/analysis/<model>/` (analysis artifacts from `analyze_stamp_cv.py`)
- `outputs/runs/<run_id>/analysis/<model>/fusion/` (`fusion_predictions.csv`, `fusion_metrics.csv`)
- `outputs/runs/<run_id>/analysis/<model>/figures/` (ROC/PR figures; optional KM figures)

STAMP crossval outputs are isolated per-run:
- `<project_dir>/stamp_crossval_runs/<run_id>/<model>/`

## Tile QC filtering (overview)
Optional tile filtering is supported via keep-mask CSVs built from cached tile zips (see scripts in `scripts/`):
- `scripts/build_tile_keep_masks.py`: builds per-slide keep masks (`*_keep_mask.csv`)
- `scripts/apply_keep_masks_to_h5_dir.py`: applies keep masks to STAMP `.h5` feature dirs

The experiment config can point to:
- a trained `tile_filter_model.joblib`
- a `keep_mask_dir`
- a `filtered_preprocess_base` for filtered features

## Manual fusion + plotting
If you already have per-model predictions:
- Fusion evaluation:
  - `python scripts/evaluate_fusion.py --project configs/project_lusc.yaml --predictions <analysis_dir>/all_predictions_<model>.csv --out_dir <analysis_dir>/fusion`
- Plotting:
  - `python scripts/plot_results.py --fusion_predictions <analysis_dir>/fusion/fusion_predictions.csv --out_dir <analysis_dir>/figures`

## Adapting to a new cohort
- Create a new project YAML (copy `configs/project_lusc.yaml`) and update:
  - `paths.project_dir`, `paths.wsi_dir`
  - `paths.stamp_table` and STAMP-required columns
  - `paths.clinical_features_table` and `clinical.*` mappings
- Create a new experiment YAML and adjust:
  - `models`
  - tile filter settings (or disable tile filtering)
- Validate the merge key (`clinical.id_col` vs prediction `patient`) on a small subset before running a full sweep.
