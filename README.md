# Squamous WSI recurrence

This repository contains a modular pipeline for whole-slide-image (WSI) recurrence experiments in squamous carcinoma cohorts. It wraps STAMP-style preprocessing/cross-validation, summarizes model predictions, evaluates clinical/WSI fusion models, and generates manuscript-ready benchmark figures.

The code was originally developed for LUSC and vulvar squamous carcinoma recurrence experiments, but the project and experiment YAML files are intended to make the pipeline adaptable to new cohorts with similar slide-level labels.

## What the pipeline does

- Runs STAMP preprocessing and cross-validation for one or more foundation models or slide encoders.
- Optionally applies tile quality-control filtering before model training.
- Analyzes cross-validation outputs and writes per-patient predictions.
- Evaluates WSI-only, clinical-only, and WSI+clinical fusion models when clinical features are configured.
- Aggregates benchmark metrics across runs.
- Produces ROC AUC, PR AUC, aggregation-strategy, heatmap, slide-encoder, and cross-cohort rank-stability figures.

## Repository layout

```text
configs/
  project_lusc.yaml              # LUSC project paths and column mappings
  project_vulvar.yaml            # Vulvar project paths and column mappings
  project_wsi_only_example.yaml  # Minimal WSI-only template
  experiments/                   # Model lists and run-level options
scripts/
  run_experiment.py              # Main STAMP/analyze/fusion/plot driver
  analyze_stamp_cv.py            # Parse STAMP CV outputs into predictions
  evaluate_fusion.py             # WSI/clinical/fusion evaluation
  aggregate_results.py           # Combine per-run summary metrics
  plot_benchmark_figures.py      # Manuscript benchmark figures
src/wsi_recurrence/
  clinical.py, fusion.py, metrics.py, stamp_runner.py, ...
tests/
  Unit/synthetic tests for core workflow pieces
```

## Installation

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

This package intentionally keeps `pyproject.toml` minimal. In practice, the pipeline uses common scientific Python packages such as `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `pyyaml`, and `joblib`. STAMP itself and any WSI backend dependencies must be installed in the environment where experiments are run.

To verify the local install:

```bash
python -m pytest tests
```

Some tests or workflows may require optional WSI dependencies and local data paths. Synthetic tests are intended to cover most code paths without requiring the full slide dataset.

## Inputs you need

Before running a real experiment, prepare:

1. A project YAML, usually copied from `configs/project_lusc.yaml` or `configs/project_wsi_only_example.yaml`.
2. A STAMP table with at least patient IDs, slide filenames, and outcome labels.
3. WSI files reachable from the paths in the project YAML.
4. Optionally, a richer clinical-feature table for clinical-only and fusion models.
5. An experiment YAML listing the foundation models or slide encoders to run.

The project config separates:

- `paths.stamp_table`: the table passed to STAMP.
- `paths.clinical_features_table`: the table used for fusion modeling.
- `columns.*`: labels and prediction ID columns.
- `clinical.*`: merge keys, stage column, event indicator, and optional time-to-event columns.

For a new cohort, copy a project YAML and update all absolute paths and column names before running anything expensive.

## Running an experiment

Start with a dry run. This prints the commands and creates a run manifest without launching the full computation:

```bash
python scripts/run_experiment.py \
  --project configs/project_lusc.yaml \
  --experiment configs/experiments/example_stamp_qc03.yaml \
  --dry-run \
  --models ctranspath \
  --analyze \
  --fusion \
  --plot
```

Run the experiment once the planned commands look correct:

```bash
python scripts/run_experiment.py \
  --project configs/project_lusc.yaml \
  --experiment configs/experiments/example_stamp_qc03.yaml \
  --execute \
  --models ctranspath \
  --analyze \
  --fusion \
  --plot
```

To run every model listed in the experiment YAML, omit `--models`. To reuse existing STAMP outputs and only rerun downstream analysis:

```bash
python scripts/run_experiment.py \
  --project configs/project_lusc.yaml \
  --experiment configs/experiments/example_stamp_qc03.yaml \
  --execute \
  --reuse-existing \
  --analyze \
  --fusion \
  --plot
```

## Expected run outputs

Each run is written under:

```text
outputs/runs/<experiment_name>_<timestamp>/
  manifest.yaml
  stamp_configs/
  analysis/
    <model>/
      all_predictions_<model>.csv
      fusion/
        fusion_predictions.csv
        fusion_metrics.csv
      figures/
        summary_metrics.csv
        roc_curve.png/.pdf
        pr_curve.png/.pdf
```

The exact files depend on whether analysis, fusion, plotting, tile filtering, and slide encoding are enabled. For a typical fusion-enabled run, the most important outputs are:

- `all_predictions_<model>.csv`: per-patient WSI predictions.
- `fusion/fusion_predictions.csv`: WSI, clinical, and fusion predictions.
- `figures/summary_metrics.csv`: ROC AUC and PR AUC for WSI-only, clinical-only, and fusion models.

## Aggregating model results

After a run finishes, combine per-model summary metrics:

```bash
python scripts/aggregate_results.py \
  --run_dir outputs/runs/<run_id> \
  --project configs/project_lusc.yaml
```

Expected outputs:

```text
outputs/runs/<run_id>/analysis/model_summary/
  combined_metrics.csv
  ranked_models.csv
  grouped_roc_auc.png
  grouped_pr_auc.png
```

`ranked_models.csv` is the most useful quick check. It ranks models by the configured primary metric, usually fusion ROC AUC when fusion is enabled.

## Manuscript benchmark figures

Use `scripts/plot_benchmark_figures.py` with one or more wide aggregate CSVs. For cross-cohort figures, pass both cohort aggregates:

```bash
python scripts/plot_benchmark_figures.py \
  --input_csvs outputs/benchmarks/lusc_all_results_wide.csv \
               outputs/benchmarks/vulvar_all_results_wide.csv \
  --out_dir outputs/paper_figures/current \
  --metric_set both \
  --formats png pdf
```

Normal execution writes only final manuscript-ready PNG/PDF files:

```text
outputs/paper_figures/current/
  figures/main/
    aggregation_distribution_roc_main.png/.pdf
    aggregation_distribution_pr_main.png/.pdf
    fusion_improvement_roc_main.png/.pdf
    fusion_improvement_pr_main.png/.pdf
    fm_aggregator_heatmap_lusc_fusion_roc_main.png/.pdf
    fm_aggregator_heatmap_vulvar_fusion_roc_main.png/.pdf
    fm_aggregator_heatmap_lusc_fusion_pr_main.png/.pdf
    fm_aggregator_heatmap_vulvar_fusion_pr_main.png/.pdf
    slide_encoder_comparison_roc_main.png/.pdf
    slide_encoder_comparison_pr_main.png/.pdf
    cross_cohort_rank_stability_roc_main_clean.png/.pdf
    cross_cohort_rank_stability_pr_main_clean.png/.pdf
  figure_data/
    *_data.csv
    *_matrix.csv
    *_correlation.csv
  figures/supplementary/
    benchmark_figure_checks.csv
    cross_cohort_rank_stability_point_lookup.csv
    model_combination_performance_table.csv
    model_combination_performance_table.pdf
```

The cross-cohort rank-stability plots use equal x/y axis limits within each metric, color points by aggregation method, and leave individual model identities in the supplementary lookup table rather than cluttering the main scatterplot.

Experimental variants are disabled by default. To write debug/legacy figure variants under `figures/debug/`, run with:

```bash
python scripts/plot_benchmark_figures.py ... --save_debug_plots
```

## Interpreting expected benchmark results

For each model or model combination, the benchmark tables report:

- ROC AUC: discrimination over recurrence status across thresholds.
- PR AUC: precision-recall performance, useful when recurrence prevalence is low or imbalanced.
- WSI-only, clinical-only, and fusion performance when clinical fusion is configured.

In the cross-cohort rank-stability plots:

- Each point is one foundation-model by aggregation-strategy combination.
- X-axis is LUSC fusion performance.
- Y-axis is vulvar fusion performance.
- The dashed line marks equal performance in both cohorts.
- Color indicates aggregation method.

The intended visual summary is that exact model rankings can vary by cohort, while higher-performing regions are often enriched for context-aware aggregation strategies such as ViT, TransMIL, or native slide encoders.

## Tile QC filtering

Tile QC filtering is optional and controlled in the experiment YAML. The helper scripts are:

```text
scripts/build_tile_keep_masks.py
scripts/apply_keep_masks_to_h5_dir.py
scripts/train_filtering.py
scripts/filter_threshold_sweep.py
```

The experiment YAML can point to a trained `tile_filter_model.joblib`, `keep_mask_dir`, and `filtered_preprocess_base`. Disable tile filtering in the experiment YAML when running a baseline experiment.

## Adapting to a new cohort

1. Copy `configs/project_wsi_only_example.yaml` or an existing cohort YAML.
2. Update `paths.project_dir`, `paths.wsi_dir`, `paths.stamp_table`, and `paths.clinical_features_table`.
3. Update `columns.label`, `columns.pred_id`, and `clinical.*` mappings.
4. Copy an experiment YAML and choose the model list and tile-filter settings.
5. Run a single-model dry run.
6. Run one small execute job and inspect `summary_metrics.csv`.
7. Scale to the full model list once merge keys and outputs are validated.

## Troubleshooting

- If fusion fails, check that `clinical.id_col` matches the prediction `patient` IDs after any filename normalization.
- If no STAMP feature directory is found, use `--run-preprocess` or verify the configured `outputs.preprocess_base`.
- If you already have STAMP cross-validation outputs, use `--reuse-existing` or `--existing-run-dir`.
- If benchmark plotting cannot find expected metric columns, regenerate or inspect the wide aggregate CSVs before plotting.
