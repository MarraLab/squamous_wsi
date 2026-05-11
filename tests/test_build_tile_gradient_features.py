import importlib.util
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_tile_gradient_features.py"

SPEC = importlib.util.spec_from_file_location("build_tile_gradient_features", SCRIPT)
gradient = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["build_tile_gradient_features"] = gradient
SPEC.loader.exec_module(gradient)


def write_h5(path: Path, coords: np.ndarray, feats: np.ndarray) -> None:
    with h5py.File(path, "w") as h5:
        h5.create_dataset("coords", data=coords)
        h5.create_dataset("feats", data=feats)


class TestBuildTileGradientFeatures(unittest.TestCase):
    def test_append_norm_output_shape_and_finite_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_dir = tmp_path / "in"
            out_dir = tmp_path / "out"
            in_dir.mkdir()
            coords = np.array([[x, y] for y in [0.0, 256.0, 512.0] for x in [0.0, 256.0, 512.0]], dtype=float)
            feats = np.arange(9 * 4, dtype=float).reshape(9, 4)
            write_h5(in_dir / "slide1.h5", coords, feats)
            args = Namespace(
                output_feature_dir=out_dir,
                mode="append_norm",
                neighbors=None,
                neighbor_method="kdtree_radius",
                radius_factor=1.5,
                max_neighbors=8,
                min_neighbors=1,
                neighbor_tolerance_frac=0.1,
                round_decimals=3,
            )

            row = gradient.process_one(in_dir / "slide1.h5", args)
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["input_dim"], 4)
            self.assertEqual(row["output_dim"], 5)
            self.assertEqual(row["zero_neighbor_count"], 0)
            self.assertEqual(row["fraction_8_neighbor"], 1 / 9)

            with h5py.File(out_dir / "slide1.h5", "r") as h5:
                self.assertIn("coords", h5)
                self.assertIn("feats", h5)
                out_feats = h5["feats"][:]
                self.assertEqual(out_feats.shape, (9, 5))
                self.assertTrue(np.isfinite(out_feats).all())
                np.testing.assert_allclose(h5["coords"][:], coords)
                self.assertEqual(h5.attrs["gradient_neighbor_method"], "kdtree_radius")

    def test_isolated_tile_zero_neighbor_no_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_dir = tmp_path / "in"
            out_dir = tmp_path / "out"
            in_dir.mkdir()
            coords = np.array([[0.0, 0.0], [256.0, 0.0], [5000.0, 5000.0]], dtype=float)
            feats = np.array([[1.0, 0.0], [2.0, 0.0], [10.0, 10.0]], dtype=float)
            write_h5(in_dir / "slide2.h5", coords, feats)
            args = Namespace(
                output_feature_dir=out_dir,
                mode="append_norm",
                neighbors=None,
                neighbor_method="kdtree_radius",
                radius_factor=1.5,
                max_neighbors=8,
                min_neighbors=1,
                neighbor_tolerance_frac=0.1,
                round_decimals=3,
            )

            row = gradient.process_one(in_dir / "slide2.h5", args)
            self.assertGreaterEqual(row["zero_neighbor_count"], 1)
            with h5py.File(out_dir / "slide2.h5", "r") as h5:
                out_feats = h5["feats"][:]
                self.assertTrue(np.isfinite(out_feats).all())
                self.assertEqual(float(out_feats[-1, -1]), 0.0)


if __name__ == "__main__":
    unittest.main()
