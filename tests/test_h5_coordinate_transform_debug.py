import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VIS_SCRIPT = ROOT / "scripts" / "visualize_h5_tile_alignment.py"

SPEC = importlib.util.spec_from_file_location("visualize_h5_tile_alignment", VIS_SCRIPT)
viz = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["visualize_h5_tile_alignment"] = viz
SPEC.loader.exec_module(viz)


class TestH5CoordinateTransformDebug(unittest.TestCase):
    def test_regular_grid_exact_neighbors(self):
        coords = np.array([[x, y] for y in [0.0, 256.0, 512.0] for x in [0.0, 256.0, 512.0]])
        counts = viz.exact_neighbor_counts(coords)
        self.assertEqual(int(counts[4]), 8)
        self.assertEqual(int(np.sum(counts == 3)), 4)
        self.assertEqual(int(np.sum(counts == 5)), 4)

    def test_regular_grid_kdtree_radius_neighbors(self):
        coords = np.array([[x, y] for y in [0.0, 256.0, 512.0] for x in [0.0, 256.0, 512.0]])
        counts = viz.kdtree_radius_neighbor_counts(coords, radius_factor=1.5, max_neighbors=8)
        self.assertEqual(int(counts[4]), 8)
        self.assertEqual(int(np.sum(counts == 3)), 4)
        self.assertEqual(int(np.sum(counts == 5)), 4)

    def test_jittered_grid_tolerant_neighbors_recover_interior(self):
        rng = np.random.default_rng(3)
        coords = np.array([[x, y] for y in [0.0, 256.0, 512.0] for x in [0.0, 256.0, 512.0]], dtype=float)
        jittered = coords + rng.normal(0, 1.0, size=coords.shape)
        exact = viz.exact_neighbor_counts(jittered)
        tolerant = viz.tolerant_neighbor_counts(jittered, tolerance_frac=0.1)
        self.assertLess(int(exact[4]), 8)
        self.assertEqual(int(tolerant[4]), 8)
        kdtree = viz.kdtree_radius_neighbor_counts(jittered, radius_factor=1.5, max_neighbors=8)
        self.assertEqual(int(kdtree[4]), 8)

    def test_missing_corner_and_isolated_tile_kdtree_counts(self):
        coords = np.array([[x, y] for y in [0.0, 256.0, 512.0] for x in [0.0, 256.0, 512.0]], dtype=float)
        coords = coords[1:]  # remove one corner
        coords = np.vstack([coords, [5000.0, 5000.0]])
        counts = viz.kdtree_radius_neighbor_counts(coords, radius_factor=1.5, max_neighbors=8)
        self.assertEqual(int(counts[-1]), 0)
        self.assertTrue(np.max(counts[:-1]) <= 8)

    def test_swap_xy_transform(self):
        coords = np.array([[10.0, 20.0], [30.0, 40.0]])
        transformed = viz.apply_coordinate_transform(coords, "swap_xy", 100.0, 200.0)
        np.testing.assert_allclose(transformed, np.array([[20.0, 10.0], [40.0, 30.0]]))

    def test_y_inverted_transform(self):
        coords = np.array([[10.0, 20.0], [30.0, 40.0]])
        transformed = viz.apply_coordinate_transform(coords, "invert_y", 100.0, 200.0)
        np.testing.assert_allclose(transformed, np.array([[10.0, 180.0], [30.0, 160.0]]))

    def test_pixel_vs_micron_mapping(self):
        coords = np.array([[100.0, 200.0]])
        thumb_shape = (100, 200, 3)
        x_pixel, y_pixel, _, units_pixel, _ = viz.map_coords_to_thumbnail(
            coords,
            thumb_shape,
            1000,
            1000,
            coord_units="pixel",
            transform="raw",
            mpp_x=0.25,
            mpp_y=0.25,
        )
        x_micron, y_micron, _, units_micron, _ = viz.map_coords_to_thumbnail(
            coords,
            thumb_shape,
            1000,
            1000,
            coord_units="micron",
            transform="raw",
            mpp_x=0.25,
            mpp_y=0.25,
        )
        self.assertEqual(units_pixel, "pixel")
        self.assertEqual(units_micron, "micron")
        self.assertAlmostEqual(float(x_pixel[0]), 20.0)
        self.assertAlmostEqual(float(y_pixel[0]), 20.0)
        self.assertAlmostEqual(float(x_micron[0]), 80.0)
        self.assertAlmostEqual(float(y_micron[0]), 80.0)


if __name__ == "__main__":
    unittest.main()
