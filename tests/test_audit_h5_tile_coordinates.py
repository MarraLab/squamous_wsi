import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_h5_tile_coordinates.py"

SPEC = importlib.util.spec_from_file_location("audit_h5_tile_coordinates", SCRIPT)
audit_h5_tile_coordinates = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["audit_h5_tile_coordinates"] = audit_h5_tile_coordinates
SPEC.loader.exec_module(audit_h5_tile_coordinates)


def write_h5(path: Path, coords=None, feats=None, include_coords=True, include_feats=True) -> None:
    with h5py.File(path, "w") as h5:
        if include_coords:
            h5.create_dataset("coords", data=np.asarray(coords, dtype=float))
        if include_feats:
            h5.create_dataset("feats", data=np.asarray(feats, dtype=float))


class TestAuditH5TileCoordinates(unittest.TestCase):
    def test_regular_3x3_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "slideA.h5"
            coords = np.array([[x, y] for y in [0.0, 10.0, 20.0] for x in [0.0, 10.0, 20.0]])
            feats = np.ones((9, 4))
            write_h5(path, coords, feats)

            row = audit_h5_tile_coordinates.audit_h5(path)
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["n_tiles"], 9)
            self.assertEqual(row["feature_dim"], 4)
            self.assertEqual(row["n_duplicate_coords"], 0)
            self.assertEqual(row["dx_mode_approx"], 10.0)
            self.assertEqual(row["dy_mode_approx"], 10.0)
            self.assertAlmostEqual(row["frac_tiles_with_8_neighbors"], 1 / 9)

    def test_duplicated_coordinate_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "slide_dup.h5"
            coords = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 0.0]])
            feats = np.ones((3, 2))
            write_h5(path, coords, feats)

            row = audit_h5_tile_coordinates.audit_h5(path)
            self.assertEqual(row["status"], "warning")
            self.assertEqual(row["n_duplicate_coords"], 1)
            self.assertIn("duplicate", row["warning"])

    def test_missing_coords_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "slide_missing.h5"
            write_h5(path, coords=None, feats=np.ones((3, 2)), include_coords=False)

            row = audit_h5_tile_coordinates.audit_h5(path)
            self.assertEqual(row["status"], "fail")
            self.assertIn("missing required dataset", row["warning"])

    def test_coords_features_length_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "slide_mismatch.h5"
            write_h5(path, coords=np.ones((3, 2)), feats=np.ones((2, 5)))

            row = audit_h5_tile_coordinates.audit_h5(path)
            self.assertEqual(row["status"], "fail")
            self.assertIn("length mismatch", row["warning"])

    def test_mask_alignment_perfect_and_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mask_dir = tmp_path / "masks"
            mask_dir.mkdir()
            coords = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
            feats = np.ones((4, 3))

            perfect_h5 = tmp_path / "perfect.h5"
            write_h5(perfect_h5, coords, feats)
            pd.DataFrame(
                {
                    "x_um": coords[:, 0],
                    "y_um": coords[:, 1],
                    "keep": [True, True, False, True],
                }
            ).to_csv(mask_dir / "perfect.csv", index=False)
            perfect = audit_h5_tile_coordinates.audit_h5(perfect_h5, mask_dir=mask_dir)
            self.assertEqual(perfect["mask_match_fraction"], 1.0)
            self.assertEqual(perfect["mask_h5_unmatched_count"], 0)
            self.assertAlmostEqual(perfect["mask_keep_fraction"], 0.75)

            partial_h5 = tmp_path / "partial.h5"
            write_h5(partial_h5, coords, feats)
            pd.DataFrame(
                {
                    "x_um": [0.0, 10.0],
                    "y_um": [0.0, 0.0],
                    "keep": [True, False],
                }
            ).to_csv(mask_dir / "partial.csv", index=False)
            partial = audit_h5_tile_coordinates.audit_h5(partial_h5, mask_dir=mask_dir)
            self.assertEqual(partial["status"], "warning")
            self.assertEqual(partial["mask_match_fraction"], 0.5)
            self.assertEqual(partial["mask_h5_unmatched_count"], 2)
            self.assertIn("low mask match fraction", partial["warning"])


if __name__ == "__main__":
    unittest.main()
