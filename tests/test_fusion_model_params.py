import unittest

from wsi_recurrence.fusion_config import resolve_fusion_model_params


class TestFusionModelParams(unittest.TestCase):
    def test_config_used_when_cli_absent(self):
        project_cfg = {
            "analysis": {
                "fusion_model": {
                    "C": 0.3,
                    "class_weight": "balanced",
                    "solver": "lbfgs",
                    "max_iter": 123,
                }
            }
        }
        p = resolve_fusion_model_params(project_cfg=project_cfg)
        self.assertEqual(p.C, 0.3)
        self.assertEqual(p.class_weight, "balanced")
        self.assertEqual(p.solver, "lbfgs")
        self.assertEqual(p.max_iter, 123)

    def test_cli_overrides_config_C(self):
        project_cfg = {"analysis": {"fusion_model": {"C": 0.3}}}
        p = resolve_fusion_model_params(project_cfg=project_cfg, cli_C=0.1)
        self.assertEqual(p.C, 0.1)

    def test_class_weight_none_becomes_None(self):
        project_cfg = {"analysis": {"fusion_model": {"class_weight": "balanced"}}}
        p = resolve_fusion_model_params(project_cfg=project_cfg, cli_class_weight="none")
        self.assertIsNone(p.class_weight)


if __name__ == "__main__":
    unittest.main()

