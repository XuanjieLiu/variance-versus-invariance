import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from run_codebook_health import compute_codebook_metrics, select_best_checkpoint


class CodebookHealthTest(unittest.TestCase):
    def test_codebook_metrics_for_perfect_alignment(self):
        confusion = np.eye(3, dtype=np.int64) * 10
        codebook = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        metrics = compute_codebook_metrics(confusion, codebook)

        self.assertEqual(metrics["active_codes"], 3)
        self.assertAlmostEqual(metrics["usage_perplexity"], 3.0)
        self.assertAlmostEqual(metrics["codebook_purity"], 1.0)
        self.assertAlmostEqual(metrics["one_to_one_accuracy"], 1.0)
        self.assertEqual(metrics["dominant_label_coverage"], 3)
        self.assertAlmostEqual(metrics["min_atom_distance"], 1.0)

    def test_codebook_metrics_for_collapsed_usage(self):
        confusion = np.array([[4, 4, 4], [0, 0, 0], [0, 0, 0]])
        codebook = torch.tensor([[0.0], [0.0], [1.0]])

        metrics = compute_codebook_metrics(confusion, codebook)

        self.assertEqual(metrics["active_codes"], 1)
        self.assertAlmostEqual(metrics["usage_perplexity"], 1.0)
        self.assertAlmostEqual(metrics["codebook_purity"], 1 / 3)
        self.assertAlmostEqual(metrics["one_to_one_accuracy"], 1 / 3)
        self.assertEqual(metrics["dominant_label_coverage"], 1)
        self.assertAlmostEqual(metrics["min_atom_distance"], 0.0)

    def test_select_best_retained_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir)
            history_path = run_dir / "loss_epoch_history.csv"
            with history_path.open("w", newline="") as history_file:
                writer = csv.DictWriter(
                    history_file, fieldnames=["partition", "epoch", "total_loss"]
                )
                writer.writeheader()
                writer.writerow(
                    {"partition": "val", "epoch": 1, "total_loss": 0.1}
                )
                writer.writerow(
                    {"partition": "val", "epoch": 2, "total_loss": 0.2}
                )
            (run_dir / "cp_epoch2.pt").touch()

            checkpoint, epoch, val_loss = select_best_checkpoint(str(run_dir))

            self.assertTrue(checkpoint.endswith("cp_epoch2.pt"))
            self.assertEqual(epoch, 2)
            self.assertAlmostEqual(val_loss, 0.2)


if __name__ == "__main__":
    unittest.main()
