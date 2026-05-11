import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "analyze_stamp_cv.py"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class TestAnalyzeStampCvSynthetic(unittest.TestCase):
    def test_lusc_like_uses_label_1(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cv_dir = Path(td) / "cv"
            _write_csv(
                cv_dir / "split-0" / "patient-preds.csv",
                [
                    {"patient": "P1", "recur": 0, "pred": 0, "recur_0": 0.9, "recur_1": 0.1, "loss": 0.0},
                    {"patient": "P2", "recur": 1, "pred": 1, "recur_0": 0.1, "recur_1": 0.9, "loss": 0.0},
                ],
            )
            out_dir = Path(td) / "out"
            cmd = [
                sys.executable,
                str(SCRIPT),
                "--cv_root",
                str(cv_dir),
                "--model_name",
                "ctranspath",
                "--out_dir",
                str(out_dir),
                "--no_plots",
            ]
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)

            combined = pd.read_csv(out_dir / "all_predictions_ctranspath.csv")
            self.assertEqual(list(combined.columns), ["patient", "recur", "fold", "pred_ctranspath"])
            self.assertEqual(int(combined.loc[combined["patient"] == "P1", "fold"].iloc[0]), 0)
            self.assertAlmostEqual(float(combined.loc[combined["patient"] == "P1", "pred_ctranspath"].iloc[0]), 0.1)

            auc = pd.read_csv(out_dir / "auc_summary_ctranspath.csv")
            self.assertIn("roc_auc", auc.columns)
            self.assertIn("pr_auc", auc.columns)

    def test_vulvar_like_prefers_label_1_over_pred(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cv_dir = Path(td) / "cv"
            _write_csv(
                cv_dir / "split-0" / "patient-preds.csv",
                [
                    {
                        "patient": "P1",
                        "has_recurrence": 0,
                        "pred": 0,
                        "has_recurrence_0": 0.8,
                        "has_recurrence_1": 0.2,
                        "loss": 0.0,
                    },
                    {
                        "patient": "P2",
                        "has_recurrence": 1,
                        "pred": 1,
                        "has_recurrence_0": 0.2,
                        "has_recurrence_1": 0.8,
                        "loss": 0.0,
                    },
                ],
            )
            out_dir = Path(td) / "out"
            cmd = [
                sys.executable,
                str(SCRIPT),
                "--project",
                "configs/project_vulvar.yaml",
                "--cv_root",
                str(cv_dir),
                "--model_name",
                "ctranspath",
                "--out_dir",
                str(out_dir),
                "--no_plots",
            ]
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)

            combined = pd.read_csv(out_dir / "all_predictions_ctranspath.csv")
            self.assertEqual(list(combined.columns), ["patient", "has_recurrence", "fold", "pred_ctranspath"])
            self.assertEqual(int(combined.loc[combined["patient"] == "P1", "fold"].iloc[0]), 0)
            self.assertAlmostEqual(float(combined.loc[combined["patient"] == "P1", "pred_ctranspath"].iloc[0]), 0.2)

    def test_missing_probability_column_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cv_dir = Path(td) / "cv"
            _write_csv(
                cv_dir / "split-0" / "patient-preds.csv",
                [
                    {"patient": "P1", "has_recurrence": 0, "pred": 2.0},
                    {"patient": "P2", "has_recurrence": 1, "pred": -1.0},
                ],
            )
            out_dir = Path(td) / "out"
            cmd = [
                sys.executable,
                str(SCRIPT),
                "--cv_root",
                str(cv_dir),
                "--model_name",
                "ctranspath",
                "--out_dir",
                str(out_dir),
                "--label_col",
                "has_recurrence",
                "--patient_col",
                "patient",
                "--no_plots",
            ]
            proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("Could not infer probability column", (proc.stderr or "") + (proc.stdout or ""))

    def test_no_valid_folds_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cv_dir = Path(td) / "cv"
            (cv_dir / "split-0").mkdir(parents=True, exist_ok=True)  # no patient-preds.csv
            out_dir = Path(td) / "out"
            cmd = [
                sys.executable,
                str(SCRIPT),
                "--cv_root",
                str(cv_dir),
                "--model_name",
                "ctranspath",
                "--out_dir",
                str(out_dir),
                "--no_plots",
            ]
            proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)
            self.assertNotEqual(proc.returncode, 0)
            msg = (proc.stderr or "") + (proc.stdout or "")
            self.assertIn("No valid prediction files found", msg)


if __name__ == "__main__":
    unittest.main()
