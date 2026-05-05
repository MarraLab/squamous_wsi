#!/usr/bin/env python
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from wsi_recurrence.validation import validate_predictions_complete


PROJECT_CFG = {
    "columns": {"pred_id": "patient", "label": "recur"},
    "validation": {
        "expected_n_patients": 152,
        "expected_n_positive": 47,
        "expected_n_negative": 105,
        "patient_col": "patient",
        "label_col": "recur",
    },
}


def _write(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _expect_pass(name: str, path: Path) -> None:
    stats = validate_predictions_complete(path, PROJECT_CFG)
    assert stats.n_patients == 152, (name, stats)
    assert stats.n_positive == 47, (name, stats)
    assert stats.n_negative == 105, (name, stats)
    print(f"PASS: {name}")


def _expect_fail(name: str, path: Path, contains: str) -> None:
    try:
        validate_predictions_complete(path, PROJECT_CFG)
    except Exception as exc:
        msg = str(exc)
        if contains not in msg:
            raise AssertionError(f"{name}: expected error containing {contains!r}, got: {msg}") from exc
        print(f"PASS (expected fail): {name} -> {contains!r}")
        return
    raise AssertionError(f"{name}: expected failure but validation passed")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="wsi_recurrence_validation_") as td:
        base = Path(td)

        # A) Valid synthetic predictions: 152 unique patients; 47 positives
        patients = [f"p{i:03d}" for i in range(152)]
        labels = [1] * 47 + [0] * 105
        df_ok = pd.DataFrame({"patient": patients, "recur": labels, "pred_ctranspath": [0.5] * 152})
        _expect_pass("valid_152", _write(df_ok, base / "ok.csv"))

        # B) Invalid synthetic predictions: only 11 patients
        df_11 = df_ok.iloc[:11].copy()
        _expect_fail("invalid_11_patients", _write(df_11, base / "p11.csv"), "expected 152 unique patients, found 11")

        # C) Duplicate patient predictions
        df_dup = df_ok.copy()
        df_dup.loc[1, "patient"] = df_dup.loc[0, "patient"]
        _expect_fail("duplicate_patient", _write(df_dup, base / "dup.csv"), "Duplicate patients found")

        # D) Missing label column
        df_nolabel = df_ok.drop(columns=["recur"]).copy()
        _expect_fail("missing_label_col", _write(df_nolabel, base / "nolabel.csv"), "missing label column")

    print("All validation tests passed.")


if __name__ == "__main__":
    main()

