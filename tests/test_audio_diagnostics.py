import csv
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from utils.audio_diagnostics import AudioDiagnostics


class FakeAudioDataset:
    training = False
    split = "val"
    n_fragments = 12
    groups_per_take = 10
    s_list = [f"instrument{i:02d}" for i in range(12)]
    c_list = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    manifest_sha256 = "fixture"

    def __init__(self, root):
        self.npy_paths = [str(Path(root) / f"ins{style:02d}_{take:03d}.npy")
                          for style in range(12) for take in range(10)]
        self.eval_items = [(style * 10 + take, group)
                           for group in range(10) for take in range(10)
                           for style in range(12)]

    def __len__(self):
        return 1200

    def __getitem__(self, index):
        raise AssertionError("Diagnostics must not read or resample the dataset")


def synthetic_batch(dataset, start, end):
    items = dataset.eval_items[start:end]
    content = torch.arange(12).expand(len(items), 12).clone()
    style = torch.tensor([file_id // 10 for file_id, _ in items])[:, None].expand_as(content).clone()
    ec = torch.nn.functional.one_hot(content, 12).float()
    zs = torch.nn.functional.one_hot(style, 12).float()
    codes = (content + 1) % 12
    # Values deliberately exceed the RGB range and identify each source group.
    values = torch.tensor([file_id + group / 10 for file_id, group in items])
    originals = (values[:, None, None, None] + content[:, :, None, None]
                 + torch.arange(6).reshape(1, 1, 2, 3)).float()
    outputs = originals + 0.25
    return originals, outputs, ec, zs, codes, content, style


class AudioDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = FakeAudioDataset(self.root / "val")
        self.config = {"enabled": True, "every_n_epochs": 50, "final_epoch": 1999,
                       "pca_components": 12, "plot_content_ids": [0]}

    def monitor(self, dataset=None, config=None):
        return AudioDiagnostics(self.root / "run", dataset or self.dataset,
                                self.config if config is None else config, 12, self.dataset.c_list)

    def collect_all(self, monitor, batch_size=37):
        for start in range(0, len(self.dataset), batch_size):
            batch = synthetic_batch(self.dataset, start, min(start + batch_size, len(self.dataset)))
            monitor.collect(*batch, {"content_fragment_mpd": torch.tensor(float(start))})
        return monitor.counts.numpy().copy()

    def assert_numpy_rng_equal(self, before):
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_recording_split_uses_explicit_round_robin_items(self):
        monitor = self.monitor()
        fit_files = {monitor.items[i][0] for i in monitor.fit_ids}
        score_files = {monitor.items[i][0] for i in monitor.score_ids}
        self.assertFalse(fit_files & score_files)
        self.assertEqual(len(fit_files), 60)
        self.assertEqual(len(score_files), 60)
        self.assertEqual(len(monitor.fit_ids), 600)
        self.assertEqual(len(monitor.score_ids), 600)
        for style in range(12):
            self.assertEqual(fit_files & set(range(style * 10, style * 10 + 10)),
                             set(range(style * 10, style * 10 + 5)))
            index = monitor.grid_ids[style]
            self.assertEqual(monitor.items[index], (style * 10 + 5, 0))
        for file_id in fit_files | score_files:
            ids = monitor.fit_ids if file_id in fit_files else monitor.score_ids
            self.assertEqual({monitor.items[i][1] for i in ids if monitor.items[i][0] == file_id},
                             set(range(10)))
        # In round-robin order index 1 belongs to instrument 1, not recording 0.
        self.assertEqual(monitor.items[1], (10, 0))
        protocol = json.loads((monitor.root / "audio_diagnostic_protocol.json").read_text())
        self.assertEqual(protocol["expected_fragment_count"], 14400)
        self.assertFalse(set(protocol["fit_recordings"]) & set(protocol["score_recordings"]))
        self.assertIn("not disentanglement", protocol["interpretation"])

    def test_recording_index_method_fallback_preserves_round_robin_split(self):
        items = self.dataset.eval_items
        del self.dataset.eval_items
        self.dataset.recording_index = lambda index: items[index][0]
        monitor = self.monitor()
        self.assertEqual(monitor.items, items)
        self.assertEqual(monitor.items[monitor.grid_ids[1]], (15, 0))

    def test_missing_mapping_and_duplicate_or_missing_groups_are_rejected(self):
        del self.dataset.eval_items
        with self.assertRaisesRegex(ValueError, "eval_items or recording_index"):
            self.monitor()
        self.dataset = FakeAudioDataset(self.root / "val")
        self.dataset.eval_items[-1] = self.dataset.eval_items[0]
        with self.assertRaisesRegex(ValueError, "exactly once"):
            self.monitor()
        self.dataset = FakeAudioDataset(self.root / "val")
        self.dataset.eval_items.pop()
        with self.assertRaisesRegex(ValueError, "exactly once"):
            self.monitor()

    def test_unsupported_dataset_protocols_fail_early(self):
        for name, value in (("training", True), ("split", "test"), ("groups_per_take", 1)):
            dataset = FakeAudioDataset(self.root / "val")
            setattr(dataset, name, value)
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.monitor(dataset=dataset)
        self.dataset.npy_paths.reverse()
        with self.assertRaisesRegex(ValueError, "sorted"):
            self.monitor()

    def test_schedule_and_mutable_absolute_final_epoch(self):
        monitor = self.monitor()
        for epoch, active in ((0, True), (10, False), (49, False), (50, True),
                              (1998, False), (1999, True)):
            monitor.begin(epoch, epoch * 30, "fragment")
            self.assertEqual(monitor.active, active)
        self.config["final_epoch"] = 3
        monitor.begin(3, 120, "fragment")
        self.assertTrue(monitor.active)
        monitor.begin(4, 150, "fragment")
        self.assertFalse(monitor.active)
        # None arguments prove inactive calls do not inspect or retain tensors.
        monitor.collect(*([None] * 8))
        self.assertEqual(monitor.finish(None, None), {})
        self.assertEqual(monitor.offset, 0)
        self.assertFalse(monitor.records)
        self.assertFalse((monitor.root / "audio_diagnostic_history.csv").exists())

    def test_disabled_monitor_is_noop(self):
        monitor = self.monitor(config={**self.config, "enabled": False})
        monitor.begin(0, 0, "fragment")
        monitor.collect(*([None] * 8))
        self.assertEqual(monitor.finish(None, None), {})
        self.assertFalse(monitor.root.exists())

    def test_full_probes_balanced_counts_and_rng_preservation(self):
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        monitor = self.monitor()
        monitor.begin(0, 30, "fragment")
        counts = self.collect_all(monitor)
        self.assertEqual(int(counts.sum()), 14400)
        self.assertTrue(bool((monitor.pairs == 100).all()))
        self.assertFalse(any(value.requires_grad for record in monitor.records.values() for value in record))
        with patch.object(monitor, "_write_artifacts"), patch.object(monitor, "_plot_curves"):
            metrics = monitor.finish(counts, "cpu")
        self.assertAlmostEqual(metrics["one_to_one_accuracy"], 1)
        self.assertAlmostEqual(metrics["usage_perplexity"], 12)
        self.assertAlmostEqual(metrics["zs_style_raw_accuracy"], 1)
        self.assertAlmostEqual(metrics["zc_pre_vq_content_raw_accuracy"], 1)
        self.assertAlmostEqual(metrics["ec_vq_style_accuracy"], 1 / 12)
        self.assertAlmostEqual(metrics["recon_loss"], 0.0625)
        expected_mpd = sum(start * min(37, 1200 - start) for start in range(0, 1200, 37)) / 1200
        self.assertAlmostEqual(metrics["content_fragment_mpd"], expected_mpd)
        self.assertIsNone(metrics["style_fragment_mpd"])
        with (monitor.root / "audio_diagnostic_history.csv").open() as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(int(row["fit_fragments"]), 7200)
        self.assertEqual(int(row["score_fragments"]), 7200)
        self.assertFalse(monitor.records)
        self.assertFalse(monitor.grid_records)
        self.assertFalse(monitor.active)
        self.assertEqual(random.getstate(), python_rng)
        self.assert_numpy_rng_equal(numpy_rng)
        torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)
        if cuda_rng is not None:
            for before, after in zip(cuda_rng, torch.cuda.get_rng_state_all()):
                torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_partial_or_disagreeing_full_counts_are_rejected(self):
        monitor = self.monitor()
        monitor.begin(0, 0, "fragment")
        monitor.collect(*synthetic_batch(self.dataset, 0, 12), {})
        with self.assertRaisesRegex(RuntimeError, "complete validation"):
            monitor.finish(monitor.counts, "cpu")
        self.assertFalse(monitor.records)
        monitor.begin(0, 0, "fragment")
        counts = self.collect_all(monitor)
        counts[0, 0] += 1
        with self.assertRaisesRegex(ValueError, "disagree"):
            monitor.finish(counts, "cpu")
        self.assertFalse((monitor.root / "audio_diagnostic_history.csv").exists())

    def test_shuffled_style_duplicate_pitch_and_bad_codes_are_rejected(self):
        monitor = self.monitor()
        for kind in ("style", "pitch", "code", "shape", "nonfinite"):
            monitor.begin(0, 0, "fragment")
            batch = list(synthetic_batch(self.dataset, 0, 12))
            if kind == "style":
                batch[6] = batch[6].roll(1, 0)
            elif kind == "pitch":
                batch[5][0, 0] = 1
            elif kind == "code":
                batch[4][0, 0] = 12
            elif kind == "shape":
                batch[1] = batch[1].unsqueeze(2)
            else:
                batch[2][0, 0, 0] = float("nan")
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                monitor.collect(*batch, {})

    def test_capture_is_detached_and_does_not_alias_forward_tensors(self):
        monitor = self.monitor()
        monitor.begin(0, 0, "fragment")
        # Include first score recording for each style, where fixed grids live.
        batch = list(synthetic_batch(self.dataset, 0, 612))
        batch[2].requires_grad_(True)
        batch[3].requires_grad_(True)
        monitor.collect(*batch, {})
        original_record = monitor.records[0][0].clone()
        grid_id = monitor.grid_ids[0]
        original_grid = monitor.grid_records[grid_id][0].clone()
        with torch.no_grad():
            batch[0].zero_()
            batch[2].zero_()
            batch[3].zero_()
        torch.testing.assert_close(monitor.records[0][0], original_record)
        torch.testing.assert_close(monitor.grid_records[grid_id][0], original_grid)
        self.assertFalse(monitor.records[0][0].requires_grad)
        self.assertIsNone(batch[2].grad)

    def test_artifacts_keep_all_notes_and_share_the_hungarian_mapping(self):
        monitor = self.monitor()
        monitor.begin(1999, 60000, "fragment")
        counts = self.collect_all(monitor, batch_size=32)
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
        monitor.finish(counts, "cpu")
        directory = monitor.root / "audio_diagnostics" / "epoch01999"
        payload = json.loads((directory / "metrics.json").read_text())
        self.assertEqual(payload["code_to_label"], {str((p + 1) % 12): p for p in range(12)})
        self.assertEqual(payload["row_code_order"], list(range(1, 12)) + [0])
        np.testing.assert_allclose(np.asarray(payload["probabilities"]).sum(0), 1)
        self.assertEqual(payload["final_epoch"], 1999)
        with np.load(directory / "reconstructions.npz") as data:
            self.assertEqual(data["originals"].shape, (12, 12, 2, 3))
            self.assertGreater(data["outputs"].max(), 1)
            np.testing.assert_allclose(data["outputs"] - data["originals"], 0.25)
            np.testing.assert_array_equal(data["dataset_indices"], monitor.grid_ids)
        paths = [directory / "confusion.png", monitor.root / "audio_diagnostics.png"]
        paths += [directory / f"reconstruction_style{s:02d}.png" for s in range(12)]
        from PIL import Image
        for path in paths:
            with Image.open(path) as image:
                pixels = np.asarray(image.convert("RGB"))
                self.assertGreater(min(pixels.shape[:2]), 100)
                self.assertGreater(float(pixels.std()), 1)
        from matplotlib import pyplot as plt
        self.assertEqual(plt.get_fignums(), [])
        self.assertEqual(random.getstate(), before[0])
        self.assert_numpy_rng_equal(before[1])
        torch.testing.assert_close(torch.get_rng_state(), before[2], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
