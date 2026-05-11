import unittest

import numpy as np
import pandas as pd

from wsi_recurrence.clinical import merge_predictions_with_clinical
from wsi_recurrence.fusion import evaluate_fusion_groupkfold


def _make_base_df(n_groups: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    patient = np.array([f"P{i:03d}" for i in range(n_groups)])
    # Balanced labels across groups to avoid single-class folds.
    y = (np.arange(n_groups) % 2).astype(int)
    pred = np.clip(0.2 + 0.6 * y + rng.normal(0, 0.15, size=n_groups), 0, 1)
    return pd.DataFrame({"patient": patient, "recur": y, "pred": pred})


class TestFusionSynthetic(unittest.TestCase):
    def test_numeric_only_lusc_like(self):
        df = _make_base_df()
        df["stage_cont"] = np.linspace(0, 1, len(df))

        res = evaluate_fusion_groupkfold(
            df,
            id_col="patient",
            label_col="recur",
            pred_col="pred",
            clinical_features=["stage_cont"],
            n_splits=5,
        )

        self.assertIn("clinical_pred", res.predictions.columns)
        self.assertIn("fusion_pred", res.predictions.columns)
        self.assertIn("stage_cont", res.predictions.columns)
        self.assertEqual(set(res.metrics["method"]), {"WSI", "Clinical", "Fusion"})
        self.assertFalse(res.metrics["roc_auc"].isna().any())
        self.assertFalse(res.metrics["pr_auc"].isna().any())

    def test_mixed_categorical_vulvar_like(self):
        df = _make_base_df()
        rng = np.random.default_rng(1)
        df["has_lymph_node"] = (rng.random(len(df)) > 0.6).astype(int)
        df["has_invasion"] = (rng.random(len(df)) > 0.5).astype(int)
        df["hpv_p53_group"] = rng.choice(["hpv", "p53", "other"], size=len(df), replace=True)

        res = evaluate_fusion_groupkfold(
            df.rename(columns={"recur": "has_recurrence"}),
            id_col="patient",
            label_col="has_recurrence",
            pred_col="pred",
            clinical_features=["has_lymph_node", "has_invasion", "hpv_p53_group"],
            n_splits=5,
        )

        self.assertIn("clinical_pred", res.predictions.columns)
        self.assertIn("fusion_pred", res.predictions.columns)
        for c in ["has_lymph_node", "has_invasion", "hpv_p53_group"]:
            self.assertIn(c, res.predictions.columns)
        self.assertEqual(set(res.metrics["method"]), {"WSI", "Clinical", "Fusion"})
        self.assertFalse(res.metrics["roc_auc"].isna().any())
        self.assertFalse(res.metrics["pr_auc"].isna().any())

    def test_missing_clinical_feature_fails_clearly(self):
        pred_df = _make_base_df()
        clinical_df = pd.DataFrame({"patient": pred_df["patient"], "has_lymph_node": 1})
        with self.assertRaisesRegex(ValueError, r"Clinical feature\(s\) missing from clinical table: hpv_p53_group"):
            merge_predictions_with_clinical(
                pred_df,
                clinical_df,
                pred_id_col="patient",
                pred_col="pred",
                label_col="recur",
                clinical_id_col="patient",
                clinical_features=["has_lymph_node", "hpv_p53_group"],
                extra_cols=[],
            )

    def test_no_survival_columns_ok(self):
        df = _make_base_df()
        df["stage_cont"] = np.linspace(0, 1, len(df))
        # No time_to_event/event columns; fusion should still run.
        res = evaluate_fusion_groupkfold(
            df,
            id_col="patient",
            label_col="recur",
            pred_col="pred",
            clinical_features=["stage_cont"],
            n_splits=5,
        )
        self.assertIn("fusion_pred", res.predictions.columns)

    def test_fixed_fold_column_is_reused(self):
        df = _make_base_df(n_groups=40)
        df["stage_cont"] = np.linspace(0, 1, len(df))
        df["fold"] = np.arange(len(df)) % 5

        res = evaluate_fusion_groupkfold(
            df,
            id_col="patient",
            label_col="recur",
            pred_col="pred",
            clinical_features=["stage_cont"],
            n_splits=3,
            fold_col="fold",
        )

        self.assertEqual(res.predictions["fold"].tolist(), df["fold"].tolist())
        self.assertEqual(set(res.metrics["split_source"]), {"fixed:fold"})
        self.assertEqual(set(res.metrics["n_splits"]), {5})


if __name__ == "__main__":
    unittest.main()
