import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from run_codebook_health import (
    compute_codebook_metrics,
    select_best_checkpoint,
    select_best_macro_checkpoint,
)
from utils.codebook_metrics import (
    batch_confusion_counts,
    compute_alias_geometry_metrics,
    compute_assignment_metrics,
)


class CodebookHealthTest(unittest.TestCase):
    def test_codebook_metrics_for_perfect_alignment(self):
        confusion = np.eye(3, dtype=np.int64) * 10
        codebook = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        metrics = compute_codebook_metrics(confusion, codebook)

        self.assertEqual(metrics["active_codes"], 3)
        self.assertAlmostEqual(metrics["usage_perplexity"], 3.0)
        self.assertAlmostEqual(metrics["codebook_purity"], 1.0)
        self.assertAlmostEqual(metrics["macro_atom_purity"], 1.0)
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

    def test_legacy_accuracy_does_not_require_one_to_one_matching(self):
        confusion = np.array([[10, 0], [10, 0]], dtype=np.int64)

        metrics = compute_assignment_metrics(confusion)

        self.assertAlmostEqual(metrics["legacy_codebook_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["macro_atom_purity"], 1.0)
        self.assertAlmostEqual(metrics["codebook_purity"], 1.0)
        self.assertAlmostEqual(metrics["one_to_one_accuracy"], 0.5)
        self.assertEqual(metrics["dominant_label_coverage"], 1)

    def test_macro_purity_allows_multiple_codes_per_content_and_ignores_inactive(self):
        confusion = np.array(
            [[10, 0], [5, 0], [0, 7], [0, 3], [0, 0]], dtype=np.int64
        )

        metrics = compute_assignment_metrics(confusion)

        self.assertEqual(metrics["active_codes"], 4)
        self.assertEqual(metrics["dominant_label_coverage"], 2)
        self.assertEqual(metrics["dominant_label_code_counts"], [2, 2])
        self.assertEqual(metrics["dominant_label_code_count_min"], 2)
        self.assertEqual(metrics["dominant_label_code_count_max"], 2)
        self.assertAlmostEqual(metrics["dominant_label_code_count_cv"], 0.0)
        self.assertAlmostEqual(metrics["macro_atom_purity"], 1.0)
        self.assertAlmostEqual(metrics["legacy_codebook_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["codebook_purity"], 1.0)
        self.assertAlmostEqual(metrics["one_to_one_accuracy"], 17 / 25)

    def test_redundant_codebook_reports_imbalanced_dominant_code_counts(self):
        confusion = np.array(
            [
                [9, 0, 0],
                [8, 0, 0],
                [7, 0, 0],
                [0, 6, 0],
                [0, 0, 5],
                [0, 0, 0],
            ],
            dtype=np.int64,
        )

        metrics = compute_assignment_metrics(confusion)

        self.assertEqual(metrics["active_codes"], 5)
        self.assertEqual(metrics["dominant_label_code_counts"], [3, 1, 1])
        self.assertEqual(metrics["dominant_label_code_count_min"], 1)
        self.assertEqual(metrics["dominant_label_code_count_max"], 3)
        self.assertAlmostEqual(metrics["macro_atom_purity"], 1.0)
        self.assertLess(metrics["one_to_one_accuracy"], 1.0)

    def test_rectangular_hungarian_matching_and_inactive_atom(self):
        confusion = np.array([[8, 0], [0, 7], [0, 0]], dtype=np.int64)

        metrics = compute_assignment_metrics(confusion)

        self.assertEqual(metrics["active_codes"], 2)
        self.assertAlmostEqual(metrics["one_to_one_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["legacy_codebook_accuracy"], 1.0)

    def test_batch_confusion_counts_is_vectorized(self):
        indices = torch.tensor([[0, 1, 1], [2, 0, 1]])
        labels = torch.tensor([[0, 1, 0], [1, 0, 1]])

        counts = batch_confusion_counts(indices, labels, 3, 2)

        np.testing.assert_array_equal(
            counts.numpy(), np.array([[2, 0], [1, 2], [0, 1]])
        )

    def test_alias_geometry_rewards_tight_same_content_codes(self):
        confusion = np.array(
            [[10, 0], [9, 0], [0, 8], [0, 7]], dtype=np.int64
        )
        codebook = torch.tensor(
            [[0.0, 0.0], [0.1, 0.0], [4.0, 0.0], [4.1, 0.0]]
        )

        metrics = compute_alias_geometry_metrics(confusion, codebook)

        self.assertAlmostEqual(metrics["alias_nearest_same_closer_fraction"], 1.0)
        self.assertLess(metrics["alias_within_between_ratio"], 0.1)
        self.assertLess(
            metrics["alias_nearest_same_distance_median"],
            metrics["alias_nearest_other_distance_median"],
        )

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

    def test_select_best_macro_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir)
            checkpoint = run_dir / "cp_best_macro_atom_purity_epoch7.pt"
            checkpoint.touch()
            metadata = {
                "epoch": 7,
                "macro_atom_purity": 0.81,
                "validation_total_loss": 0.14,
                "checkpoint": checkpoint.name,
            }
            (run_dir / "best_macro_atom_purity.json").write_text(
                json.dumps(metadata)
            )

            selected = select_best_macro_checkpoint(str(run_dir))

            self.assertEqual(selected[0], str(checkpoint))
            self.assertEqual(selected[1], 7)
            self.assertAlmostEqual(selected[2], 0.81)
            self.assertAlmostEqual(selected[3], 0.14)


if __name__ == "__main__":
    unittest.main()
