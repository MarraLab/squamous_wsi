import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "plot_results.py"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class TestPlotResultsSynthetic(unittest.TestCase):
    def test_lusc_default_recur_and_pred_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pred_path = Path(td) / "all_predictions_ctranspath.csv"
            _write_csv(
                pred_path,
                [
                    {"patient": "P1", "recur": 0, "pred_ctranspath": 0.1},
                    {"patient": "P2", "recur": 1, "pred_ctranspath": 0.9},
                ],
            )
            out_dir = Path(td) / "figs"
            cmd = [sys.executable, str(SCRIPT), "--predictions", str(pred_path), "--out_dir", str(out_dir)]
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)
            self.assertTrue((out_dir / "summary_metrics.csv").exists())
            summary = pd.read_csv(out_dir / "summary_metrics.csv")
            self.assertIn("roc_auc", summary.columns)

    def test_vulvar_project_infers_label(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pred_path = Path(td) / "fusion_predictions.csv"
            _write_csv(
                pred_path,
                [
                    {"patient": "P1", "has_recurrence": 0, "pred": 0.2},
                    {"patient": "P2", "has_recurrence": 1, "pred": 0.8},
                ],
            )
            out_dir = Path(td) / "figs"
            cmd = [
                sys.executable,
                str(SCRIPT),
                "--project",
                "configs/project_vulvar.yaml",
                "--predictions",
                str(pred_path),
                "--out_dir",
                str(out_dir),
            ]
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)
            self.assertTrue((out_dir / "summary_metrics.csv").exists())

    def test_fusion_column_variants_plots_present_cols(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pred_path = Path(td) / "fusion_predictions.csv"
            _write_csv(
                pred_path,
                [
                    {"patient": "P1", "recur": 0, "pred_wsi": 0.1, "pred_clinical": 0.2, "pred_fusion": 0.3},
                    {"patient": "P2", "recur": 1, "pred_wsi": 0.9, "pred_clinical": 0.8, "pred_fusion": 0.7},
                ],
            )
            out_dir = Path(td) / "figs"
            cmd = [sys.executable, str(SCRIPT), "--predictions", str(pred_path), "--out_dir", str(out_dir)]
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)
            summary = pd.read_csv(out_dir / "summary_metrics.csv")
            self.assertEqual(set(summary["method"].tolist()), {"WSI", "Clinical", "Fusion"})

    def test_missing_survival_columns_skips_km(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pred_path = Path(td) / "all_predictions_ctranspath.csv"
            _write_csv(
                pred_path,
                [
                    {"patient": "P1", "recur": 0, "pred": 0.1},
                    {"patient": "P2", "recur": 1, "pred": 0.9},
                ],
            )
            out_dir = Path(td) / "figs"
            cmd = [sys.executable, str(SCRIPT), "--predictions", str(pred_path), "--out_dir", str(out_dir)]
            proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0)
            msg = (proc.stdout or "") + (proc.stderr or "")
            self.assertIn("skipping KM plotting", msg)


if __name__ == "__main__":
    unittest.main()

