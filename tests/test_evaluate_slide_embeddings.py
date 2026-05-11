import subprocess
import sys
import tempfile
import unittest
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_slide_embeddings.py"

SPEC = importlib.util.spec_from_file_location("evaluate_slide_embeddings", SCRIPT)
evaluate_slide_embeddings = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["evaluate_slide_embeddings"] = evaluate_slide_embeddings
SPEC.loader.exec_module(evaluate_slide_embeddings)


class TestEvaluateSlideEmbeddings(unittest.TestCase):
    def test_synthetic_logreg_and_categorical_fusion_without_umap(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            emb_dir = tmp_path / "eagle-slide-test"
            out_dir = tmp_path / "out"
            emb_dir.mkdir()

            rows = []
            rng = np.random.default_rng(7)
            for i in range(20):
                patient = f"P{i:03d}"
                label = i % 2
                filename = f"{patient}.ndpi"
                rows.append(
                    {
                        "patient": patient,
                        "filename": filename,
                        "recur": label,
                        "stage_cont": float(i % 4),
                        "hpv_p53_group": "hpv" if i % 3 == 0 else "p53",
                    }
                )
                vec = rng.normal(0, 0.1, size=6)
                vec[0] += 1.5 * label
                with h5py.File(emb_dir / f"{patient}.h5", "w") as h5:
                    h5.create_dataset("features", data=vec.reshape(1, -1))

            clinical_path = tmp_path / "clinical.csv"
            pd.DataFrame(rows).to_csv(clinical_path, index=False)
            project_path = tmp_path / "project.yaml"
            project = {
                "analysis": {
                    "run_fusion": True,
                    "clinical_path": str(clinical_path),
                    "clinical_features": ["stage_cont", "hpv_p53_group"],
                    "fusion_model": {"C": 0.1, "class_weight": "none", "solver": "lbfgs", "max_iter": 5000},
                },
                "columns": {"pred_id": "patient", "label": "recur"},
                "crossval": {"patient_label": "patient", "filename_label": "filename", "ground_truth_label": "recur"},
                "paths": {"stamp_table": str(clinical_path)},
            }
            project_path.write_text(yaml.safe_dump(project))

            cmd = [
                sys.executable,
                str(SCRIPT),
                "--project",
                str(project_path),
                "--embedding_dirs",
                str(emb_dir),
                "--out_dir",
                str(out_dir),
                "--fusion",
                "--no_umap",
                "--n_splits",
                "4",
                "--C_values",
                "0.1",
                "--class_weights",
                "none",
            ]
            subprocess.run(cmd, check=True, cwd=ROOT)

            enc_out = out_dir / "eagle-slide-test"
            self.assertTrue((enc_out / "embedding_manifest.csv").exists())
            self.assertTrue((enc_out / "logreg_predictions.csv").exists())
            self.assertTrue((enc_out / "logreg_sweep.csv").exists())
            self.assertTrue((enc_out / "fusion_predictions.csv").exists())
            self.assertTrue((enc_out / "fusion_sweep.csv").exists())
            self.assertTrue((out_dir / "slide_embedding_logreg_summary.csv").exists())
            self.assertTrue((out_dir / "slide_embedding_fusion_summary.csv").exists())
            self.assertTrue((out_dir / "slide_embedding_wide_summary.csv").exists())
            self.assertFalse((enc_out / "umap_coordinates.csv").exists())

            pred = pd.read_csv(enc_out / "logreg_predictions.csv")
            self.assertEqual({"patient", "label", "pred_embedding", "fold"}.issubset(pred.columns), True)
            self.assertEqual(len(pred), 20)
            self.assertEqual(pred["fold"].nunique(), 4)

            fusion_pred = pd.read_csv(enc_out / "fusion_predictions.csv")
            expected = {"patient", "label", "stage_cont", "hpv_p53_group", "pred_embedding", "pred_clinical", "pred_fusion", "fold"}
            self.assertTrue(expected.issubset(fusion_pred.columns))

            fusion_sweep = pd.read_csv(enc_out / "fusion_sweep.csv")
            self.assertEqual(set(fusion_sweep["method"]), {"embedding-only", "clinical-only", "fusion"})

    def test_cluster_associations_are_global_chi_square_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            enc_out = tmp_path / "encoder_a"
            enc_out.mkdir()
            coords = pd.DataFrame(
                {
                    "patient": [f"P{i:03d}" for i in range(12)],
                    "umap1": [-5.0, -5.2, -4.8, -4.9, 0.0, 0.2, -0.2, 0.1, 5.0, 5.2, 4.8, 5.1],
                    "umap2": [0.0, 0.1, -0.1, 0.2, 5.0, 5.1, 4.9, 5.2, 0.0, 0.2, -0.2, 0.1],
                    "recur": [0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0],
                    "hpv_p53_group": ["hpv", "hpv", "p53", "other", "hpv", "p53", "other", "other", "p53", "p53", "hpv", "other"],
                    "age": list(range(12)),
                }
            )
            coords["pred_embedding"] = np.linspace(0.05, 0.95, len(coords))
            coords.to_csv(enc_out / "umap_coordinates.csv", index=False)
            evaluate_slide_embeddings.plot_umap(coords, "recur", enc_out / "umap_recur.png")
            evaluate_slide_embeddings.plot_umap(
                coords,
                "pred_embedding",
                enc_out / "umap_predicted_recurrence_probability.png",
                categorical=False,
            )

            assoc = evaluate_slide_embeddings.cluster_associations(
                "encoder_a",
                coords,
                ["recur", "hpv_p53_group", "age"],
                [3],
                enc_out,
                42,
            )
            assoc = evaluate_slide_embeddings.apply_cluster_fdr([assoc])
            assoc.loc[assoc["variable"] == "recur", ["status", "p_adj_bh", "cramers_v"]] = ["ok", 0.001, 0.5]
            assoc.loc[assoc["variable"] == "hpv_p53_group", "p_adj_bh"] = 0.5
            assoc.to_csv(enc_out / "cluster_association_tests.csv", index=False)
            sig_plots = evaluate_slide_embeddings.plot_significant_cluster_associations(assoc, tmp_path, 0.05)
            sig_plots.to_csv(tmp_path / "significant_cluster_plots.csv", index=False)

            out = pd.read_csv(enc_out / "cluster_association_tests.csv")
            self.assertEqual(len(out), 3)
            self.assertEqual(set(out["variable"]), {"recur", "hpv_p53_group", "age"})
            self.assertEqual(set(out["test"]), {"chi2_contingency"})
            self.assertFalse((out["test"] == "fisher_exact_cluster_vs_rest").any())
            self.assertIn("p_adj_bh", out.columns)
            self.assertIn("cramers_v", out.columns)
            self.assertIn("contingency_table_json", out.columns)
            self.assertIn("cluster_space", out.columns)
            self.assertEqual(set(out["cluster_space"]), {"umap"})

            hpv_row = out[out["variable"] == "hpv_p53_group"].iloc[0]
            self.assertEqual(hpv_row["status"], "ok")
            self.assertEqual(int(hpv_row["n_clusters"]), 3)
            self.assertEqual(int(hpv_row["n_categories"]), 3)
            self.assertGreater(int(hpv_row["n_cells_expected_lt5"]), 0)
            self.assertIn("expected contingency cells are < 5", hpv_row["warning"])

            age_row = out[out["variable"] == "age"].iloc[0]
            self.assertEqual(age_row["status"], "skipped")
            self.assertEqual(age_row["warning"], "continuous numeric variable not tested as categorical")

            self.assertTrue((enc_out / "umap_recur.png").exists())
            self.assertFalse((enc_out / "umap_label.png").exists())
            self.assertTrue((enc_out / "umap_predicted_recurrence_probability.png").exists())
            self.assertTrue((enc_out / "umap_clusters_k3.png").exists())
            self.assertTrue((enc_out / "umap_significant_recur_k3.png").exists())
            self.assertFalse((enc_out / "umap_significant_hpv_p53_group_k3.png").exists())
            self.assertEqual(len(sig_plots), 1)
            self.assertEqual(sig_plots.iloc[0]["variable"], "recur")
            self.assertTrue((tmp_path / "significant_cluster_plots.csv").exists())


if __name__ == "__main__":
    unittest.main()
