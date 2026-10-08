"""Small real-Hex GPU train/resume/evaluation smoke; deletes all run artifacts."""
import copy
import csv
import logging
from pathlib import Path
import socket
import tempfile
import unittest

import torch
import yaml

from model.rank_regularization import RANK_LOG_COLUMNS
from trainer import Trainer
from run_codebook_health import evaluate


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(torch.cuda.is_available(), "GPU allocation required")
class RankSmokeTest(unittest.TestCase):
    def test_real_hex_train_resume_and_health(self):
        self.assertTrue(socket.gethostname().startswith("ws-"), "Never run on login")
        config = yaml.safe_load((ROOT / "configs/hex/v2/cfg_vvi_hex_k16_repo_seed0.yaml").read_text())
        config.update(debug=True, debug_portion=.0004, batch_size=4, num_workers=0,
                      wandb=False, save_best_validation_loss=False, snapshot_every_n_epochs=0,
                      macro_best_health_gate={}, epochs=1)
        config["data_dir"] = str((ROOT / config["data_dir"]).resolve())
        # Only rank/loss/codebook/V3 outputs are in scope for this small smoke.
        for key in ("disentanglement_probes", "bn_diagnostic", "optimization_diagnostics",
                    "per_style_codebook_monitor", "training_usage_monitor"):
            config[key] = {"enabled": False}
        config["loss_config"]["rank_regularization"] = {
            "enabled": True, "target_rank": 4, "weight": .01, "eps": 1e-12}
        with tempfile.TemporaryDirectory(prefix="smoke-v3-rank-", dir=ROOT / "logs") as directory:
            config["log_dir"] = directory
            for name, expected_epoch in (("initial", 0), ("resume", 1)):
                config["name"] = name
                if expected_epoch:
                    config["load_checkpoint"] = str(Path(directory) / "initial/cp_current_epoch0.pt")
                    config["inherit_resume_history"] = True
                trainer = Trainer(copy.deepcopy(config))
                trainer.prepare_data()
                trainer.build_model()
                self.assertEqual(trainer.start_epoch, expected_epoch)
                trainer.train()
                run = Path(trainer.log_dir)
                for filename in ("loss_history.csv", "loss_epoch_history.csv", "loss_curves.png",
                                 "codebook_epoch_history.csv", "codebook_metrics.png",
                                 "v3_ratio_epoch_history.csv", "v3_ratios.png"):
                    self.assertTrue((run / filename).is_file(), filename)
                with (run / "loss_epoch_history.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                epoch_rows = [row for row in rows if int(row["epoch"]) == expected_epoch]
                self.assertEqual({row["partition"] for row in epoch_rows}, {"train", "val"})
                for row in epoch_rows:
                    for key in RANK_LOG_COLUMNS:
                        self.assertNotEqual(row[key], "")
                        self.assertTrue(torch.isfinite(torch.tensor(float(row[key]))))
                    self.assertAlmostEqual(float(row["weighted_rank_loss"]), .01 * float(row["rank_loss"]))
                checkpoint = run / f"cp_current_epoch{expected_epoch}.pt"
                self.assertTrue(checkpoint.is_file())
                self.assertLessEqual(len(list(run.glob("cp*.pt"))), 2)
                if expected_epoch:
                    self.assertEqual({int(row["epoch"]) for row in rows}, {0, 1})
                    # Health evaluation uses the same native objective, not the
                    # lifted decoder vector. Also exercise an existing evaluator.
                    report = evaluate(config, checkpoint, subset_size=16,
                                      subset_seed=0, subset_strategy="style_stratified")
                    self.assertEqual(report["sample_count"], 16)
                    for key in RANK_LOG_COLUMNS: self.assertIn(key, report["losses"])
                    losses = report["losses"]
                    expected = sum(losses[k] * w for k, w in config["loss_config"]["weights"].items())
                    self.assertAlmostEqual(losses["total_loss"], expected + losses["weighted_rank_loss"], places=5)
                del trainer
                torch.cuda.empty_cache()
            # Close file handles before TemporaryDirectory removes this exact,
            # newly created smoke directory. Historical logs are never targeted.
            for handler in list(logging.getLogger().handlers):
                if isinstance(handler, logging.FileHandler) and str(handler.baseFilename).startswith(directory + "/"):
                    handler.close()
                    logging.getLogger().removeHandler(handler)
        self.assertFalse(Path(directory).exists())


if __name__ == "__main__":
    unittest.main()
