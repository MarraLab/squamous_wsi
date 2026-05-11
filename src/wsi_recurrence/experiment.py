from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shlex
from typing import Any, Dict, Tuple

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at {path}, got {type(data).__name__}")
    return data


def dump_yaml(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def utc_timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


@dataclass(frozen=True)
class ExperimentSpec:
    project_path: Path
    experiment_path: Path
    config: Dict[str, Any]

    @property
    def name(self) -> str:
        exp = self.config.get("experiment", {})
        name = exp.get("name", None)
        if not name or not isinstance(name, str):
            raise ValueError("Missing experiment.name in experiment YAML.")
        return name

    def models(self) -> list[str]:
        models = self.config.get("models", [])
        if not isinstance(models, list):
            raise ValueError("Expected `models` to be a list.")
        return [str(m) for m in models]


def load_experiment(project_yaml: Path, experiment_yaml: Path) -> ExperimentSpec:
    project_cfg = load_yaml(project_yaml)
    exp_cfg = load_yaml(experiment_yaml)
    merged = _deep_merge(project_cfg, exp_cfg)
    return ExperimentSpec(
        project_path=project_yaml,
        experiment_path=experiment_yaml,
        config=merged,
    )


def create_run_dir(out_root: Path, experiment_name: str, ts: str | None = None) -> Path:
    stamp = ts or utc_timestamp()
    run_dir = out_root / f"{experiment_name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_manifest(spec: ExperimentSpec, run_dir: Path, cli_argv: list[str] | None = None) -> Dict[str, Any]:
    provenance: Dict[str, Any] = {
        "project_yaml": str(spec.project_path),
        "experiment_yaml": str(spec.experiment_path),
        "run_dir": str(run_dir),
    }
    if cli_argv is not None:
        argv = [str(x) for x in cli_argv]
        provenance["cli"] = {
            "argv": argv,
            "command": shlex.join(argv),
        }

    return {
        "experiment": spec.config.get("experiment", {}),
        "project": spec.config.get("project", {}),
        "paths": spec.config.get("paths", {}),
        "outputs": spec.config.get("outputs", {}),
        "stamp": spec.config.get("stamp", {}),
        "crossval": spec.config.get("crossval", {}),
        "advanced_config": spec.config.get("advanced_config", {}),
        "tile_filter": spec.config.get("tile_filter", {}),
        "slide_encoding": spec.config.get("slide_encoding", {}),
        "models": spec.models(),
        "provenance": provenance,
    }
