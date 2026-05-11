import tempfile
import unittest
from pathlib import Path

import yaml

from wsi_recurrence.slide_encoding import parse_slide_encoding_config, validate_slide_encoding_pairing
from wsi_recurrence.stamp_runner import (
    build_stamp_config,
    find_slide_encoding_output_dir,
    update_stamp_config_slide_encoding,
)


def _base_cfg(project_dir: Path) -> dict:
    return {
        "paths": {"project_dir": str(project_dir), "stamp_table": str(project_dir / "clin.csv")},
        "outputs": {"preprocess_base": "stamp_preprocess", "crossval_base": "stamp_crossval"},
        "stamp": {"device": "cpu", "max_workers": 1},
        "crossval": {
            "ground_truth_label": "recur",
            "patient_label": "patient",
            "filename_label": "filename",
            "n_splits": 2,
            "task": "classification",
        },
        "advanced_config": {"model_name": "vit"},
    }


class TestSlideEncodingSupport(unittest.TestCase):
    def test_eagle_requires_agg_feat_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_dir = Path(td)
            cfg = _base_cfg(project_dir)
            cfg["slide_encoding"] = {"enabled": True, "encoder": "eagle", "feat_model": "ctranspath"}
            se = parse_slide_encoding_config(cfg)
            assert se is not None
            with self.assertRaises(ValueError):
                validate_slide_encoding_pairing(se)

    def test_find_slide_encoding_output_dir_prefers_encoder(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            (base / "other-slide-aaaa").mkdir(parents=True)
            (base / "other-slide-aaaa" / "a.h5").write_bytes(b"")
            (base / "eagle-slide-bbbb").mkdir(parents=True)
            (base / "eagle-slide-bbbb" / "b.h5").write_bytes(b"")

            out = find_slide_encoding_output_dir(base, encoder="eagle")
            self.assertIsNotNone(out)
            self.assertEqual(out.name, "eagle-slide-bbbb")

    def test_update_stamp_config_slide_encoding_writes_section(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_dir = Path(td)
            run_dir = project_dir / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            cfg = _base_cfg(project_dir)

            config = build_stamp_config(cfg, "eagle_ctranspath_virchow2", run_dir=run_dir)
            cfg_path = run_dir / "config.yaml"
            with cfg_path.open("w") as f:
                yaml.safe_dump(config, f, sort_keys=False)

            feat_dir = project_dir / "feat"
            feat_dir.mkdir(parents=True, exist_ok=True)
            out_dir = project_dir / "slide_out"
            out_dir.mkdir(parents=True, exist_ok=True)
            update_stamp_config_slide_encoding(
                cfg_path,
                encoder="eagle",
                feat_dir=feat_dir,
                agg_feat_dir=project_dir / "agg",
                output_dir=out_dir,
                device="cpu",
                generate_hash=True,
            )

            with cfg_path.open("r") as f:
                updated = yaml.safe_load(f)
            self.assertIn("slide_encoding", updated)
            self.assertEqual(updated["slide_encoding"]["encoder"], "eagle")
            self.assertEqual(updated["slide_encoding"]["feat_dir"], str(feat_dir))
            self.assertEqual(updated["slide_encoding"]["output_dir"], str(out_dir))


if __name__ == "__main__":
    unittest.main()

