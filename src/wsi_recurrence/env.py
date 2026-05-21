from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _find_upwards(filename: str, start: Path | None = None) -> Path | None:
    current = (start or Path.cwd()).resolve()
    for parent in [current, *current.parents]:
        candidate = parent / filename
        if candidate.exists():
            return candidate
    return None


def load_local_env(filename: str = ".env.local") -> None:
    """
    Load simple KEY=VALUE entries from a local env file without overriding
    variables already set by the user's shell or scheduler.
    """
    env_path = _find_upwards(filename)
    if env_path is None:
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, os.path.expanduser(os.path.expandvars(value)))


def expand_env_vars(value: Any) -> Any:
    """
    Recursively expand environment variables in YAML-derived structures.
    Missing variables are left unchanged by os.path.expandvars, making the
    unresolved setting visible in downstream errors.
    """
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env_vars(item) for key, item in value.items()}
    return value
