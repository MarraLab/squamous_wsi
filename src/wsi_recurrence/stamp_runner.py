from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml


def _dump_yaml(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at {path}, got {type(data).__name__}")
    return data


def _project_dir(cfg: Dict[str, Any]) -> Path:
    project_dir = cfg.get("paths", {}).get("project_dir", None)
    if not project_dir:
        raise ValueError("Missing paths.project_dir in merged config.")
    return Path(str(project_dir))


def _join_under_project(project_dir: Path, maybe_rel: str | Path) -> Path:
    p = Path(str(maybe_rel))
    if p.is_absolute():
        return p
    return project_dir / p


def build_stamp_config(cfg: Dict[str, Any], model_name: str, *, run_dir: Path) -> Dict[str, Any]:
    """
    Build a STAMP YAML config equivalent to the current `CLAM/run_stamp_pipeline.py` template.
    This does not run STAMP; it only emits config content.
    """
    paths = cfg.get("paths", {})
    outputs = cfg.get("outputs", {})
    stamp = cfg.get("stamp", {})
    crossval = cfg.get("crossval", {})
    advanced = cfg.get("advanced_config", {})

    project_dir = Path(str(paths["project_dir"]))
    wsi_dir = Path(str(paths.get("wsi_dir", project_dir)))
    stamp_table = paths.get("stamp_table") or paths.get("clinical_table")
    if not stamp_table:
        raise ValueError("Missing paths.stamp_table (or legacy paths.clinical_table) in merged config.")
    stamp_table = Path(str(stamp_table))
    cache_dir = Path(str(paths.get("cache_dir", "/tmp/image_cache")))

    preprocess_base = _join_under_project(project_dir, outputs.get("preprocess_base", "stamp_preprocess"))
    crossval_base = _join_under_project(project_dir, outputs.get("crossval_base", "stamp_crossval"))

    output_base = preprocess_base / model_name / "wsi"
    # Crossval outputs should be unique per run, but stay under project_dir.
    run_id = run_dir.name
    crossval_run_dir = project_dir / "stamp_crossval_runs" / run_id / model_name
    cfg_out = {
        "preprocessing": {
            "output_dir": str(output_base),
            "wsi_dir": str(wsi_dir),
            "extractor": str(model_name),
            "device": str(stamp.get("device", "cuda")),
            "cache_dir": str(cache_dir),
            "max_workers": int(stamp.get("max_workers", 16)),
        },
        "crossval": {
            "output_dir": str(crossval_run_dir),
            "clini_table": str(stamp_table),
            "feature_dir": "/tmp/",
            "slide_table": str(stamp_table),
            "ground_truth_label": str(crossval.get("ground_truth_label", "recur")),
            "patient_label": str(crossval.get("patient_label", "patient")),
            "filename_label": str(crossval.get("filename_label", "filename")),
            "n_splits": int(crossval.get("n_splits", 5)),
            "task": str(crossval.get("task", "classification")),
        },
        "advanced_config": advanced,
    }
    return cfg_out


def write_stamp_configs(cfg: Dict[str, Any], models: List[str], *, run_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for model in models:
        config = build_stamp_config(cfg, model, run_dir=run_dir)
        path = run_dir / "configs" / "stamp" / f"config_{model}.yaml"
        _dump_yaml(config, path)
        out[model] = path
    return out


def planned_stamp_commands(config_paths: Dict[str, Path]) -> List[str]:
    cmds: List[str] = []
    for model, path in config_paths.items():
        cmds.append(f"stamp --config {path} preprocess")
        cmds.append(f"stamp --config {path} crossval")
    return cmds


def find_preprocess_output_dir(model_name: str, preprocess_base: Path) -> Path | None:
    """
    Mirror the old `run_stamp_pipeline.py` behavior:
      - Look under: preprocess_base/model_name/wsi
      - Find directories named f"{model_name}-*"
      - Return the newest/sorted latest directory
    """
    base = preprocess_base / model_name / "wsi"
    if not base.exists():
        return None
    subdirs = [p for p in base.glob(f"{model_name}-*") if p.is_dir()]
    if not subdirs:
        return None
    return sorted(subdirs)[-1]


def update_stamp_config_feature_dir(config_path: Path, feature_dir: Path | str) -> None:
    """
    Rewrite only `crossval.feature_dir` in a STAMP YAML config.
    """
    cfg = _load_yaml(config_path)
    cv = cfg.get("crossval", {})
    if not isinstance(cv, dict):
        cv = {}
    cv["feature_dir"] = str(feature_dir)
    cfg["crossval"] = cv
    _dump_yaml(cfg, config_path)
