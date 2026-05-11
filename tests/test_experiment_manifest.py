from pathlib import Path
import shlex
import unittest

from wsi_recurrence.experiment import ExperimentSpec, build_manifest


class ExperimentManifestTests(unittest.TestCase):
    def test_build_manifest_records_cli_argv_and_command(self) -> None:
        spec = ExperimentSpec(
            project_path=Path("configs/project.yaml"),
            experiment_path=Path("configs/experiments/example.yaml"),
            config={"experiment": {"name": "demo"}, "models": ["ctranspath"]},
        )
        argv = [
            "scripts/run_experiment.py",
            "--project",
            "configs/project.yaml",
            "--models",
            "ctranspath,virchow-full",
        ]

        manifest = build_manifest(spec, Path("outputs/runs/demo_20260511_120000"), cli_argv=argv)

        self.assertEqual(manifest["provenance"]["cli"]["argv"], argv)
        self.assertEqual(manifest["provenance"]["cli"]["command"], shlex.join(argv))


if __name__ == "__main__":
    unittest.main()
