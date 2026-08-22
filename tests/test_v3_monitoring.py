import json
import tempfile
import unittest
from pathlib import Path

import torch

from model.v3_loss import V3Loss, V3_RATIO_KEYS
from trainer import Trainer


class V3MonitoringTest(unittest.TestCase):
    def test_v3_loss_exposes_all_raw_ratios(self):
        loss = V3Loss(
            {
                "relativity": 15,
                "weights": {
                    "recon_loss": 1,
                    "content_loss": 1,
                    "style_loss": 1,
                    "sample_loss": 1,
                    "fragment_loss": 1,
                    "commit_loss": 0.1,
                },
            }
        )
        generator = torch.Generator().manual_seed(0)
        z_c = torch.randn(3, 4, 5, generator=generator)
        z_c_vq = torch.randn(3, 4, 5, generator=generator)
        z_s = torch.randn(3, 4, 6, generator=generator)
        x = torch.randn(3, 4, 2, generator=generator)

        outputs = loss.compute_loss(x, z_c, z_c_vq, torch.tensor(0.2), z_s, x)

        for name in V3_RATIO_KEYS:
            self.assertIn(name, outputs)
            self.assertEqual(outputs[name].ndim, 0)
            self.assertTrue(torch.isfinite(outputs[name]))

    def test_macro_checkpoint_ordering_and_tie_break(self):
        self.assertTrue(Trainer._is_better_macro_checkpoint(0.8, 0.2, None, None))
        self.assertTrue(Trainer._is_better_macro_checkpoint(0.8, 0.1, 0.8, 0.2))
        self.assertFalse(Trainer._is_better_macro_checkpoint(0.8, 0.3, 0.8, 0.2))
        self.assertFalse(Trainer._is_better_macro_checkpoint(0.7, 0.1, 0.8, 0.2))

    def test_macro_checkpoint_is_replaced_only_by_a_better_candidate(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            trainer = Trainer.__new__(Trainer)
            trainer.log_dir = temporary_dir
            trainer.model = torch.nn.Linear(2, 2)
            trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
            trainer.scheduler = torch.optim.lr_scheduler.StepLR(
                trainer.optimizer, step_size=1
            )
            trainer.scaler = torch.cuda.amp.GradScaler(enabled=False)
            trainer.best_macro_atom_purity = None
            trainer.best_macro_val_loss = None
            trainer.best_macro_epoch = None
            trainer.best_macro_checkpoint_path = None

            trainer._save_best_macro_checkpoint(1, 0.2, 0.8)
            first_path = Path(trainer.best_macro_checkpoint_path)
            trainer._save_current_checkpoint(1, 0.2, 0.8)
            trainer._save_current_checkpoint(2, 0.15, 0.7)
            checkpoints = list(Path(temporary_dir).glob("cp*.pt"))
            self.assertEqual(len(checkpoints), 2)
            self.assertTrue(
                (Path(temporary_dir) / "cp_current_epoch2.pt").exists()
            )
            trainer._save_best_macro_checkpoint(2, 0.1, 0.7)
            self.assertEqual(Path(trainer.best_macro_checkpoint_path), first_path)
            trainer._save_best_macro_checkpoint(3, 0.1, 0.8)

            final_path = Path(trainer.best_macro_checkpoint_path)
            self.assertFalse(first_path.exists())
            self.assertTrue(final_path.name.endswith("epoch3.pt"))
            self.assertEqual(len(list(Path(temporary_dir).glob("cp*.pt"))), 2)
            metadata = json.loads(
                (Path(temporary_dir) / "best_macro_atom_purity.json").read_text()
            )
            self.assertEqual(metadata["epoch"], 3)
            state = torch.load(final_path, weights_only=False)
            self.assertEqual(state["best_macro_epoch"], 3)
            self.assertAlmostEqual(state["best_macro_atom_purity"], 0.8)

    def test_health_gate_rejects_degenerate_macro_candidate(self):
        failures = Trainer._gate_failures(
            {"active_codes": 10, "usage_perplexity": 5.0},
            {"min": {"active_codes": 20, "usage_perplexity": 15}},
        )
        self.assertEqual(len(failures), 2)

    def test_initialization_checkpoint_uses_distinct_name(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            trainer = Trainer.__new__(Trainer)
            trainer.config = {}
            trainer.log_dir = temporary_dir
            trainer.model = torch.nn.Linear(2, 2)
            trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
            trainer.scheduler = torch.optim.lr_scheduler.StepLR(
                trainer.optimizer, step_size=1
            )
            trainer.scaler = torch.cuda.amp.GradScaler(enabled=False)
            trainer.best_macro_atom_purity = None
            trainer.best_macro_val_loss = None
            trainer.best_macro_epoch = None
            trainer.best_macro_stage = None
            trainer.best_macro_checkpoint_path = None

            trainer._save_best_macro_checkpoint(
                -1, 0.2, 0.9, metrics={}, stage="initialization"
            )

            self.assertTrue(
                (Path(temporary_dir) / "cp_best_macro_atom_purity_epochinit.pt").exists()
            )
            metadata = json.loads(
                (Path(temporary_dir) / "best_macro_atom_purity.json").read_text()
            )
            self.assertEqual(metadata["stage"], "initialization")


if __name__ == "__main__":
    unittest.main()
