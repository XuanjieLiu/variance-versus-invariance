import copy
from pathlib import Path
import unittest

import yaml

from model.rank_regularization import v3_method_specs
from model.v3_loss import V3Loss


ROOT = Path(__file__).resolve().parents[1]


class HexRankProtocolTest(unittest.TestCase):
    def test_only_rank_objective_changes_from_mean_baseline(self):
        baseline = yaml.safe_load((ROOT / "configs/hex/v2/cfg_vvi_hex_k16_repo_mean_seed0.yaml").read_text())
        for rank in (4, 2):
            config = yaml.safe_load((ROOT / f"configs/hex/v2/cfg_vvi_hex_k16_mean_rank{rank}_seed0.yaml").read_text())
            self.assertEqual(config["method"], "v3_rank")
            self.assertEqual(v3_method_specs(config), ["V3", "rank"])
            self.assertTrue(V3Loss(config["loss_config"], config["model_config"]).rank_enabled)
            objective = config["loss_config"].pop("rank_regularization")
            self.assertEqual(objective, {"enabled": True, "target_rank": rank, "weight": .01, "eps": 1e-12})
            config["name"] = baseline["name"]
            config["method"] = baseline["method"]
            self.assertEqual(config, baseline)

    def test_alias_requires_enabled_objective_and_legacy_still_works(self):
        config = {"method": "v3_rank", "loss_config": {}}
        with self.assertRaisesRegex(ValueError, "requires"): v3_method_specs(config)
        config["loss_config"]["rank_regularization"] = {"enabled": False}
        with self.assertRaisesRegex(ValueError, "requires"): v3_method_specs(config)
        config["method"] = "V3"
        self.assertEqual(v3_method_specs(config), ["V3"])


if __name__ == "__main__": unittest.main()
