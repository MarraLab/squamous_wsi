from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def resolve_cv_dir(cv_root: Path, cv_tag: Optional[str] = None) -> Path:
    if cv_tag:
        cv_dir = cv_root / cv_tag
        if not cv_dir.exists():
            raise FileNotFoundError(f"cv_tag directory not found: {cv_dir}")
        return cv_dir

    if list(cv_root.glob("split-*")):
        return cv_root

    cv_dirs = sorted(cv_root.glob("cv*"))
    if len(cv_dirs) == 1:
        return cv_dirs[0]
    if not cv_dirs:
        raise FileNotFoundError(f"No split-* or cv* dirs found under: {cv_root}")
    raise ValueError(
        f"Multiple cv* dirs found under {cv_root}. "
        "Pass --cv_tag or a direct cv_dir."
    )


def load_patient_predictions(cv_dir: Path) -> pd.DataFrame:
    """
    Load and concatenate patient-level predictions from all cross-validation splits.

    Normalization behavior (preserved from prior scripts):
    - If `recur_1` exists, use it as the probability prediction stored in `pred`.
    - If `pred` existed already, preserve it into `pred_label` before overwriting.
    - Always add `split` extracted from `split-{k}` directory name.
    """
    all_preds: list[pd.DataFrame] = []
    split_dirs = sorted(cv_dir.glob("split-*"))

    for split_dir in split_dirs:
        pred_file = split_dir / "patient-preds.csv"
        if not pred_file.exists():
            continue

        df = pd.read_csv(pred_file)
        if "recur_1" in df.columns:
            if "pred" in df.columns:
                df["pred_label"] = df["pred"]
            df["pred"] = df["recur_1"]

        df["split"] = int(split_dir.name.split("-")[-1])
        all_preds.append(df)

    if not all_preds:
        return pd.DataFrame()
    return pd.concat(all_preds, ignore_index=True)

