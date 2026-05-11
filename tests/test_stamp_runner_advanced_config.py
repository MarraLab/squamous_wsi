import tempfile
import unittest
from pathlib import Path

from wsi_recurrence.stamp_runner import build_stamp_config


def _base_cfg(project_dir: Path) -> dict:
    return {
        "paths": {"project_dir": str(project_dir), "stamp_table": str(project_dir / "clin.csv")},
        "outputs": {"preprocess_base": "stamp_preprocess", "crossval_base": "stamp_crossval"},
        "stamp": {"device": "cpu", "max_workers": 1},
        "crossval": {"ground_truth_label": "recur", "patient_label": "patient", "filename_label": "filename", "n_splits": 2, "task": "classification"},
    }


class TestStampRunnerAdvancedConfig(unittest.TestCase):
    def test_vit_defaults_intact(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_dir = Path(td)
            cfg = _base_cfg(project_dir)
            cfg["advanced_config"] = {"model_name": "vit"}
            out = build_stamp_config(cfg, "ctranspath", run_dir=project_dir / "run")
            adv = out["advanced_config"]
            self.assertEqual(adv["model_name"], "vit")
            self.assertIn("vit", adv["model_params"])
            self.assertIn("dim_model", adv["model_params"]["vit"])

    def test_trans_mil_adds_empty_params(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_dir = Path(td)
            cfg = _base_cfg(project_dir)
            cfg["advanced_config"] = {"model_name": "trans_mil"}
            out = build_stamp_config(cfg, "ctranspath", run_dir=project_dir / "run")
            adv = out["advanced_config"]
            self.assertEqual(adv["model_name"], "trans_mil")
            self.assertIn("trans_mil", adv["model_params"])
            self.assertEqual(adv["model_params"]["trans_mil"], {})

    def test_mlp_adds_empty_params(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_dir = Path(td)
            cfg = _base_cfg(project_dir)
            cfg["advanced_config"] = {"model_name": "mlp"}
            out = build_stamp_config(cfg, "ctranspath", run_dir=project_dir / "run")
            adv = out["advanced_config"]
            self.assertEqual(adv["model_name"], "mlp")
            self.assertIn("mlp", adv["model_params"])
            self.assertEqual(adv["model_params"]["mlp"], {})

    def test_linear_adds_empty_params(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_dir = Path(td)
            cfg = _base_cfg(project_dir)
            cfg["advanced_config"] = {"model_name": "linear"}
            out = build_stamp_config(cfg, "ctranspath", run_dir=project_dir / "run")
            adv = out["advanced_config"]
            self.assertEqual(adv["model_name"], "linear")
            self.assertIn("linear", adv["model_params"])
            self.assertEqual(adv["model_params"]["linear"], {})


if __name__ == "__main__":
    unittest.main()

