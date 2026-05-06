from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


def _as_bool(v: Any, *, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "1", "yes", "y", "on"}:
            return True
        if s in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"Expected boolean, got {type(v).__name__}: {v!r}")


def _as_str(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    return s


def _under_project(project_dir: Path, maybe_rel: str | Path) -> Path:
    p = Path(str(maybe_rel))
    return p if p.is_absolute() else (project_dir / p)


def fs_safe_key(*parts: str) -> str:
    cleaned: list[str] = []
    for p in parts:
        s = str(p).strip().lower().replace(" ", "_")
        s = "".join(ch if (ch.isalnum() or ch in {"-", "_", "."}) else "_" for ch in s)
        s = "_".join([x for x in s.split("_") if x])
        if s:
            cleaned.append(s)
    return "_".join(cleaned)


@dataclass(frozen=True)
class SlideEncodingConfig:
    enabled: bool
    encoder: str
    feat_model: str
    agg_feat_model: str | None
    output_base: Path
    device: str
    generate_hash: bool
    allow_unvalidated_pairing: bool

    @property
    def model_key(self) -> str:
        if self.agg_feat_model:
            return fs_safe_key(self.encoder, self.feat_model, self.agg_feat_model)
        return fs_safe_key(self.encoder, self.feat_model)


def parse_slide_encoding_config(cfg: Mapping[str, Any]) -> SlideEncodingConfig | None:
    raw = cfg.get("slide_encoding", None)
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("slide_encoding must be a mapping (YAML dict) when provided.")

    enabled = _as_bool(raw.get("enabled", False), default=False)
    paths = cfg.get("paths", {}) or {}
    project_dir = paths.get("project_dir", None)
    if not project_dir:
        raise ValueError("Missing paths.project_dir in merged config (required for slide_encoding).")
    project_dir = Path(str(project_dir))

    if not enabled:
        return SlideEncodingConfig(
            enabled=False,
            encoder="",
            feat_model="",
            agg_feat_model=None,
            output_base=_under_project(project_dir, "stamp_slide_encoding"),
            device=str(cfg.get("stamp", {}).get("device", "cuda")),
            generate_hash=True,
            allow_unvalidated_pairing=_as_bool(raw.get("allow_unvalidated_pairing", False), default=False),
        )

    encoder = _as_str(raw.get("encoder", None))
    feat_model = _as_str(raw.get("feat_model", None))
    agg_feat_model = _as_str(raw.get("agg_feat_model", None)) or None

    if not encoder:
        raise ValueError("slide_encoding.enabled=true requires slide_encoding.encoder")
    if not feat_model:
        raise ValueError("slide_encoding.enabled=true requires slide_encoding.feat_model")

    output_base_raw = raw.get("output_base", "stamp_slide_encoding")
    output_base = _under_project(project_dir, output_base_raw)

    device = _as_str(raw.get("device", cfg.get("stamp", {}).get("device", "cuda"))) or "cuda"
    generate_hash = _as_bool(raw.get("generate_hash", True), default=True)
    allow_unvalidated_pairing = _as_bool(raw.get("allow_unvalidated_pairing", False), default=False)

    return SlideEncodingConfig(
        enabled=True,
        encoder=encoder,
        feat_model=feat_model,
        agg_feat_model=agg_feat_model,
        output_base=output_base,
        device=device,
        generate_hash=generate_hash,
        allow_unvalidated_pairing=allow_unvalidated_pairing,
    )


def validate_slide_encoding_pairing(se: SlideEncodingConfig) -> None:
    if not se.enabled:
        return
    if se.allow_unvalidated_pairing:
        return

    enc = se.encoder.strip().lower()
    feat = se.feat_model.strip().lower()
    agg = (se.agg_feat_model or "").strip().lower()

    if enc == "eagle":
        if not se.agg_feat_model:
            raise ValueError("slide_encoding.encoder=eagle requires slide_encoding.agg_feat_model (e.g. virchow2).")
        if agg != "virchow2":
            raise ValueError(
                "slide_encoding.encoder=eagle expects slide_encoding.agg_feat_model=virchow2; "
                "set slide_encoding.allow_unvalidated_pairing=true to override."
            )
    elif enc == "gigapath":
        if feat != "gigapath":
            raise ValueError(
                "slide_encoding.encoder=gigapath expects slide_encoding.feat_model=gigapath; "
                "set slide_encoding.allow_unvalidated_pairing=true to override."
            )
        if se.agg_feat_model:
            raise ValueError("slide_encoding.encoder=gigapath does not use agg_feat_model.")
    elif enc == "titan":
        if feat not in {"conchv1.5", "conchv15"}:
            raise ValueError(
                "slide_encoding.encoder=titan expects slide_encoding.feat_model=conchv1.5; "
                "set slide_encoding.allow_unvalidated_pairing=true to override."
            )
    elif enc == "prism":
        if feat not in {"virchow-full", "virchow_full"}:
            raise ValueError(
                "slide_encoding.encoder=prism expects slide_encoding.feat_model=virchow-full; "
                "set slide_encoding.allow_unvalidated_pairing=true to override."
            )
    # cobra2 / unknown encoders: allow (advanced users can set allow_unvalidated_pairing if needed)

