from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class FusionModelParams:
    C: float = 0.01
    class_weight: str | None = "balanced"
    solver: str = "lbfgs"
    max_iter: int = 5000


def _norm_str(v: object) -> str:
    return str(v).strip()


def parse_class_weight(value: object) -> str | None:
    raw = _norm_str(value).lower()
    if raw in ("", "none", "null"):
        return None
    if raw == "balanced":
        return "balanced"
    raise ValueError('Invalid class_weight (expected "balanced" or "none").')


def resolve_fusion_model_params(
    *,
    project_cfg: Mapping[str, Any],
    cli_C: float | None = None,
    cli_class_weight: object | None = None,
    cli_solver: str | None = None,
    cli_max_iter: int | None = None,
) -> FusionModelParams:
    """
    Resolve LogisticRegression hyperparameters for clinical-only and fusion models.

    Precedence (highest to lowest):
      1) CLI value (when provided)
      2) project_cfg['analysis']['fusion_model']
      3) built-in defaults (matches historic behavior)
    """
    analysis = project_cfg.get("analysis", {}) or {}
    fusion_model = analysis.get("fusion_model", {}) or {}
    if fusion_model is None:
        fusion_model = {}
    if not isinstance(fusion_model, Mapping):
        raise ValueError("analysis.fusion_model must be a mapping (e.g. {C: 0.3, class_weight: balanced}).")

    # Defaults (historic)
    C = 0.01
    class_weight: str | None = "balanced"
    solver = "lbfgs"
    max_iter = 5000

    # Config layer
    if "C" in fusion_model and fusion_model["C"] is not None:
        C = float(fusion_model["C"])
    if "class_weight" in fusion_model:
        class_weight = parse_class_weight(fusion_model.get("class_weight"))
    if "solver" in fusion_model and fusion_model["solver"] is not None and _norm_str(fusion_model["solver"]):
        solver = _norm_str(fusion_model["solver"])
    if "max_iter" in fusion_model and fusion_model["max_iter"] is not None:
        max_iter = int(fusion_model["max_iter"])

    # CLI overrides
    if cli_C is not None:
        C = float(cli_C)
    if cli_class_weight is not None:
        # Treat empty/none/null as explicit override to None.
        class_weight = parse_class_weight(cli_class_weight)
    if cli_solver is not None and _norm_str(cli_solver):
        solver = _norm_str(cli_solver)
    if cli_max_iter is not None:
        max_iter = int(cli_max_iter)

    return FusionModelParams(C=C, class_weight=class_weight, solver=solver, max_iter=max_iter)

