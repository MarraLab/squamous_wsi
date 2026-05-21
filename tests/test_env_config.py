import os
import tempfile
import unittest
from pathlib import Path

from wsi_recurrence.experiment import load_yaml


class TestEnvConfig(unittest.TestCase):
    def test_load_yaml_expands_env_vars_from_local_env(self) -> None:
        old_cwd = Path.cwd()
        old_value = os.environ.pop("WSI_TEST_ROOT", None)
        try:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                os.chdir(root)
                (root / ".env.local").write_text("WSI_TEST_ROOT=data/test_cohort\n")
                cfg_path = root / "project.yaml"
                cfg_path.write_text(
                    "paths:\n"
                    "  project_dir: ${WSI_TEST_ROOT}\n"
                    "  stamp_table: ${WSI_TEST_ROOT}/stamp_table.csv\n"
                )

                cfg = load_yaml(cfg_path)

                self.assertEqual(cfg["paths"]["project_dir"], "data/test_cohort")
                self.assertEqual(cfg["paths"]["stamp_table"], "data/test_cohort/stamp_table.csv")
        finally:
            os.chdir(old_cwd)
            if old_value is not None:
                os.environ["WSI_TEST_ROOT"] = old_value
            else:
                os.environ.pop("WSI_TEST_ROOT", None)

    def test_shell_env_overrides_local_env(self) -> None:
        old_cwd = Path.cwd()
        old_value = os.environ.get("WSI_TEST_ROOT")
        try:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                os.chdir(root)
                os.environ["WSI_TEST_ROOT"] = "from_shell"
                (root / ".env.local").write_text("WSI_TEST_ROOT=from_file\n")
                cfg_path = root / "project.yaml"
                cfg_path.write_text("paths:\n  project_dir: ${WSI_TEST_ROOT}\n")

                cfg = load_yaml(cfg_path)

                self.assertEqual(cfg["paths"]["project_dir"], "from_shell")
        finally:
            os.chdir(old_cwd)
            if old_value is not None:
                os.environ["WSI_TEST_ROOT"] = old_value
            else:
                os.environ.pop("WSI_TEST_ROOT", None)


if __name__ == "__main__":
    unittest.main()
