"""CPU-only fixtures; run through SLURM, never directly on a login node."""

import ast
import hashlib
import json
from pathlib import Path
import pickle
import random
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np
import torch

from dataloader import insnotes_mel_dataloader as mel


class InsNotesMelDataTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(random.setstate, random.getstate())
        self.addCleanup(np.random.set_state, np.random.get_state())
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for split in ("train", "val", "test"):
            (self.root / split).mkdir()
        self.manifest = {
            "dataset": "InsNotesMelV1",
            "status": "complete",
            "audio": {
                "sample_rate": 16000, "n_fft": 1024, "hop_length": 512,
                "n_mels": 128, "note_frames": 32, "note_stride_frames": 33,
                "notes_per_take": 120,
            },
            "content_midi_pitches": list(range(60, 72)),
            "source_programs": [64, 19, 21, 41, 56, 59, 68, 71, 72, 75, 22, 52],
        }
        self._write_manifest(self.manifest)
        self.array = np.arange(128 * 3960, dtype=np.float32).reshape(128, 3960)
        self.array -= np.float32(250000)
        self.array[:, 32::33] = -1234567
        self.array[0, 0] = 0
        self.path = self._write_take("val", "ins03_080.npy")

    def _write_manifest(self, manifest):
        (self.root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def _write_take(self, split, name, array=None):
        path = self.root / split / name
        np.save(path, self.array if array is None else array, allow_pickle=False)
        return path

    def _dataset(self, split="val", **kwargs):
        return mel.InsNotesMelDataset(self.root / split, **kwargs)

    def _loader(self, split="val", **kwargs):
        kwargs.setdefault("shuffle", False)
        kwargs.setdefault("pin_memory", False)
        return mel.get_dataloader(self.root / split, batch_size=2, **kwargs)

    def _expected_fragments(self, notes):
        return torch.from_numpy(np.stack([self.array[:, n * 33:n * 33 + 32]
                                          for n in notes]))

    def _rng_states(self):
        return random.getstate(), np.random.get_state(), torch.get_rng_state()

    def _assert_rng_states(self, expected):
        actual = self._rng_states()
        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1][0], expected[1][0])
        np.testing.assert_array_equal(actual[1][1], expected[1][1])
        self.assertEqual(actual[1][2:], expected[1][2:])
        self.assertTrue(torch.equal(actual[2], expected[2]))

    def test_labels_match_legacy_without_importing_torchaudio(self):
        path = Path(mel.__file__).with_name("insnotes_dataloader.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        labels = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ("S_LIST", "C_LIST"):
                        labels[target.id] = ast.literal_eval(node.value)
        self.assertEqual(mel.S_LIST, labels["S_LIST"])
        self.assertEqual(mel.C_LIST, labels["C_LIST"])

    def test_metadata_and_recording_groups(self):
        dataset = self._dataset(training=False)
        self.assertEqual(dataset.npy_paths, [str(self.path)])
        self.assertEqual(dataset.groups_per_take, 10)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(dataset.eval_items, [(0, group) for group in range(10)])
        self.assertEqual([dataset.recording_index(i) for i in range(10)], [0] * 10)
        self.assertEqual(dataset.source_groups, [str(self.path)] * 10)
        self.assertEqual(dataset.s_list, mel.S_LIST)
        self.assertEqual(dataset.c_list, mel.C_LIST)
        self.assertEqual(dataset.source_programs, self.manifest["source_programs"])
        self.assertEqual(dataset.content_midi_pitches, list(range(60, 72)))
        self.assertEqual(dataset.n_mels, 128)
        self.assertEqual(dataset.expected_shape, (128, 3960))
        self.assertEqual(dataset.manifest, self.manifest)
        manifest_path = self.root / "manifest.json"
        self.assertEqual(dataset.manifest_path, str(manifest_path))
        self.assertEqual(dataset.manifest_sha256,
                         hashlib.sha256(manifest_path.read_bytes()).hexdigest())

    def test_val_and_test_cover_every_note_once_without_augmentation(self):
        self._write_take("test", "ins11_090.npy")
        for split, style in (("val", 3), ("test", 11)):
            with self.subTest(split=split):
                dataset = self._loader(split).dataset
                groups = []
                with mock.patch.object(mel.random, "randrange", side_effect=AssertionError):
                    for group in range(10):
                        fragments, content, styles = dataset[group]
                        notes = list(range(group * 12, (group + 1) * 12))
                        self.assertTrue(torch.equal(fragments, self._expected_fragments(notes)))
                        self.assertEqual(content.tolist(), list(range(12)))
                        self.assertEqual(styles.tolist(), [style] * 12)
                        self.assertEqual(tuple(fragments.shape), (12, 128, 32))
                        self.assertEqual(fragments.dtype, torch.float32)
                        self.assertEqual(content.dtype, torch.long)
                        self.assertEqual(styles.dtype, torch.long)
                        self.assertEqual(fragments.device.type, "cpu")
                        groups.append(fragments)
                expected = self._expected_fragments(range(120))
                self.assertTrue(torch.equal(torch.cat(groups), expected))
                self.assertEqual(groups[0][0, 0, 0].item(), 0)
                self.assertFalse(torch.any(torch.cat(groups) == -1234567).item())

    def test_training_random_cyclic_starts_are_balanced(self):
        self._write_take("train", "ins00_000.npy")
        dataset = self._loader("train", shuffle=True).dataset
        self.assertTrue(dataset.training)
        self.assertEqual(dataset.groups_per_take, 1)
        self.assertEqual(len(dataset), 1)
        self.assertIsNone(dataset.eval_items)
        self.assertEqual(dataset.recording_index(0), 0)
        with mock.patch.object(mel.random, "randrange", side_effect=range(120)) as choose:
            for start in range(120):
                fragments, content, styles = dataset[0]
                notes = [(start + offset) % 120 for offset in range(12)]
                self.assertTrue(torch.equal(fragments, self._expected_fragments(notes)))
                self.assertEqual(content.tolist(), [note % 12 for note in notes])
                self.assertEqual(sorted(content.tolist()), list(range(12)))
                self.assertEqual(styles.tolist(), [0] * 12)
            self.assertEqual(choose.call_count, 120)
            choose.assert_called_with(120)

    def test_training_crop_changes_reproducibly_with_random_seed(self):
        dataset = self._dataset(training=True)
        random.seed(7)
        first = dataset[0]
        second = dataset[0]
        random.seed(7)
        repeated = dataset[0]
        self.assertFalse(torch.equal(first[0], second[0]))
        for expected, actual in zip(first, repeated):
            self.assertTrue(torch.equal(expected, actual))

    def test_boundary_crop_wraps_and_retains_last_mel_bin(self):
        dataset = self._dataset(training=True)
        with mock.patch.object(mel.random, "randrange", return_value=119):
            fragments, content, _ = dataset[0]
        self.assertTrue(torch.equal(fragments[0], torch.from_numpy(self.array[:, 3927:3959])))
        self.assertTrue(torch.equal(fragments[1], torch.from_numpy(self.array[:, :32])))
        self.assertTrue(torch.equal(fragments[0, -1], torch.from_numpy(self.array[-1, 3927:3959])))
        self.assertEqual(content.tolist(), [11] + list(range(11)))

    def test_loader_batch_shape_defaults_and_explicit_modes(self):
        loader = mel.get_dataloader(self.root / "val", 2, shuffle=False)
        self.assertTrue(loader.pin_memory)
        self.assertFalse(loader.persistent_workers)
        self.assertIs(loader.worker_init_fn, mel._seed_worker)
        self.assertIsNotNone(loader.generator)
        x, c, s = next(iter(self._loader()))
        self.assertEqual(tuple(x.shape), (2, 12, 128, 32))
        self.assertEqual(tuple(c.shape), (2, 12))
        self.assertEqual(tuple(s.shape), (2, 12))
        self.assertEqual(len(self._loader(shuffle=True).dataset), 1)
        self.assertEqual(len(self._loader(shuffle=False, training=True).dataset), 1)
        self.assertEqual(len(self._loader(shuffle=True, training=False).dataset), 10)
        self.assertIsNone(self._loader(training=True).generator)

    def test_eval_is_repeatable_and_does_not_advance_any_global_rng(self):
        random.seed(123)
        np.random.seed(234)
        torch.manual_seed(345)
        state = self._rng_states()
        loader = self._loader(portion=0.001)
        self._assert_rng_states(state)
        first = list(loader)
        self._assert_rng_states(state)
        second = list(loader)
        self._assert_rng_states(state)
        for batch_a, batch_b in zip(first, second):
            for a, b in zip(batch_a, batch_b):
                self.assertTrue(torch.equal(a, b))

    def test_explicit_shuffled_eval_uses_its_own_seed(self):
        state = self._rng_states()
        first = list(self._loader(shuffle=True, training=False, eval_seed=17))
        self._assert_rng_states(state)
        second = list(self._loader(shuffle=True, training=False, eval_seed=17))
        third = list(self._loader(shuffle=True, training=False, eval_seed=18))
        first_x = torch.cat([batch[0] for batch in first])
        self.assertTrue(torch.equal(first_x, torch.cat([batch[0] for batch in second])))
        self.assertFalse(torch.equal(first_x, torch.cat([batch[0] for batch in third])))
        self._assert_rng_states(state)

    def test_sorted_portion_selects_takes_before_group_expansion_without_rng(self):
        self._write_take("val", "ins11_099.npy")
        self._write_take("val", "ins00_081.npy")
        self._write_take("val", "ins00_079.npy")
        expected = sorted(str(path) for path in (self.root / "val").glob("*.npy"))
        state = self._rng_states()
        dataset = self._dataset(training=False, portion=0.5)
        again = self._dataset(training=False, portion=0.5)
        smoke = self._dataset(training=False, portion=0.001)
        train = self._dataset(training=True, portion=0.5)
        self._assert_rng_states(state)
        self.assertEqual(dataset.npy_paths, expected[:2])
        self.assertEqual(dataset.npy_paths, again.npy_paths)
        self.assertEqual(train.npy_paths, expected[:2])
        self.assertEqual(smoke.npy_paths, expected[:1])
        self.assertEqual(len(dataset), 20)
        self.assertEqual(len(smoke), 10)
        self.assertEqual(dataset.source_groups, expected[:2] * 10)

    def test_interleaved_items_map_to_recordings_and_note_groups(self):
        other = self._write_take("val", "ins11_081.npy", self.array + 1)
        dataset = self._dataset(training=False)
        self.assertEqual(dataset.npy_paths, [str(self.path), str(other)])
        self.assertEqual(dataset.eval_items,
                         [(file_idx, group) for group in range(10) for file_idx in (0, 1)])
        self.assertEqual(dataset[18][2].tolist(), [3] * 12)
        self.assertTrue(torch.equal(dataset[18][0], self._expected_fragments(range(108, 120))))
        fragments, content, styles = dataset[1]
        self.assertTrue(torch.equal(fragments, self._expected_fragments(range(12)) + 1))
        self.assertEqual(content.tolist(), list(range(12)))
        self.assertEqual(styles.tolist(), [11] * 12)
        self.assertEqual([dataset.recording_index(i) for i in range(20)], [0, 1] * 10)
        self.assertEqual(dataset.source_groups, [str(self.path), str(other)] * 10)

    def test_first_twelve_eval_rows_cover_twelve_styles(self):
        for style in reversed(range(12)):
            if style != 3:
                self._write_take("val", f"ins{style:02d}_080.npy")
        state = self._rng_states()
        loader = mel.get_dataloader(self.root / "val", batch_size=12, shuffle=False,
                                    pin_memory=False)
        dataset = loader.dataset
        first, content, styles = next(iter(loader))
        self._assert_rng_states(state)
        self.assertEqual(styles[:, 0].tolist(), list(range(12)))
        self.assertEqual(content.tolist(), [list(range(12))] * 12)
        self.assertEqual(tuple(first.shape), (12, 12, 128, 32))
        self.assertEqual(dataset.eval_items,
                         [(file_idx, group) for group in range(10) for file_idx in range(12)])
        self.assertEqual(len(set(dataset.eval_items)), 120)
        for index, (file_idx, group_idx) in enumerate(dataset.eval_items):
            self.assertEqual(dataset.recording_index(index), file_idx)
            self.assertEqual(dataset.source_groups[index], dataset.npy_paths[file_idx])
        again = self._dataset(training=False)
        self.assertEqual(dataset.eval_items, again.eval_items)

    def test_round_robin_supports_uneven_style_counts_and_local_takes(self):
        self._write_take("val", "ins00_081.npy")
        self._write_take("val", "ins00_079.npy")
        self._write_take("val", "ins11_082.npy")
        dataset = self._dataset(training=False)
        # Sorted files are style 0/take 79, style 0/take 81, style 3, style 11.
        self.assertEqual(dataset.eval_items,
                         [(file_idx, group) for group in range(10) for file_idx in (0, 2, 3, 1)])
        self.assertEqual(len(set(dataset.eval_items)), 40)
        self.assertEqual([dataset[i][2][0].item() for i in range(4)], [0, 3, 11, 0])
        for file_idx in range(4):
            self.assertEqual([group for file, group in dataset.eval_items if file == file_idx],
                             list(range(10)))
        train = self._dataset(training=True)
        self.assertEqual([train.recording_index(i) for i in range(4)], list(range(4)))
        self.assertEqual(train.source_groups, train.npy_paths)

    def test_lazy_readonly_single_take_cache_and_writable_tensor(self):
        self._write_take("val", "ins04_081.npy")
        with mock.patch.object(mel.np, "load", wraps=np.load) as load, \
                mock.patch.object(mel.np, "isfinite", wraps=np.isfinite) as finite:
            dataset = self._dataset(training=False)
            load.assert_not_called()
            finite.assert_not_called()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fragments = dataset[0][0]
                fragments.fill_(42)
                for index in range(10):
                    actual = dataset[index * 2][0]
                    self.assertTrue(torch.equal(actual, self._expected_fragments(
                        range(index * 12, (index + 1) * 12))))
            self.assertFalse(any("writ" in str(w.message).lower() for w in caught))
            load.assert_called_once_with(str(self.path), mmap_mode="r", allow_pickle=False)
            self.assertIsInstance(dataset._cached_array, np.memmap)
            self.assertFalse(dataset._cached_array.flags.writeable)
            dataset[1]
            dataset[0]
            self.assertEqual(load.call_count, 3)
            self.assertEqual(finite.call_count, 2)

    def test_spawn_serialization_drops_cached_mapping(self):
        dataset = self._dataset(training=False)
        expected = dataset[0]
        restored = pickle.loads(pickle.dumps(dataset))
        self.assertIsNone(restored._cached_array)
        self.assertIsNone(restored._cached_path)
        self.assertEqual(restored._validated_paths, set())
        self.assertEqual(restored.npy_paths, dataset.npy_paths)
        for actual, original in zip(restored[0], expected):
            self.assertTrue(torch.equal(actual, original))

    def test_worker_seeds_follow_torch_without_cuda(self):
        seed = 2 ** 32 + 123
        expected_python = random.Random(123).random()
        expected_numpy = np.random.RandomState(123).random_sample()
        with mock.patch.object(mel.torch, "initial_seed", return_value=seed):
            with mock.patch.object(mel.torch.cuda, "is_available", side_effect=AssertionError):
                mel._seed_worker(0)
        self.assertEqual(random.random(), expected_python)
        self.assertEqual(np.random.random(), expected_numpy)

    def test_worker_loader_configuration_does_not_initialize_cuda(self):
        with mock.patch.object(mel.torch.cuda, "is_available", side_effect=AssertionError):
            loader = self._loader(num_workers=2)
        self.assertTrue(loader.persistent_workers)
        self.assertEqual(loader.num_workers, 2)
        self.assertIs(loader.worker_init_fn, mel._seed_worker)

    def test_persistent_spawn_workers_repeat_eval_without_advancing_parent_rng(self):
        loader = self._loader(num_workers=2)
        loader.multiprocessing_context = "spawn"
        loader.timeout = 60
        state = self._rng_states()
        try:
            first = list(loader)
            second = list(loader)
            self._assert_rng_states(state)
            self.assertEqual(len(first), 5)
            self.assertEqual(len(second), 5)
            for batch_a, batch_b in zip(first, second):
                for a, b in zip(batch_a, batch_b):
                    self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.equal(torch.cat([batch[0] for batch in first]).flatten(0, 1),
                                        self._expected_fragments(range(120))))
        finally:
            # Persistent workers must exit before the temporary recordings go away.
            if loader._iterator is not None:
                loader._iterator._shutdown_workers()

    def test_invalid_manifest_metadata(self):
        changes = [
            ("dataset", "InsNotes"), ("status", "building"),
            ("audio", None), ("content_midi_pitches", list(range(61, 73))),
            ("source_programs", list(reversed(self.manifest["source_programs"]))),
        ]
        for key, value in changes:
            with self.subTest(key=key):
                manifest = dict(self.manifest, **{key: value})
                self._write_manifest(manifest)
                with self.assertRaisesRegex(ValueError, "manifest"):
                    self._dataset()
        for key, value in self.manifest["audio"].items():
            for replacement in (None, value + 1):
                with self.subTest(audio=key, value=replacement):
                    manifest = dict(self.manifest, audio=dict(self.manifest["audio"]))
                    if replacement is None:
                        del manifest["audio"][key]
                    else:
                        manifest["audio"][key] = replacement
                    self._write_manifest(manifest)
                    with self.assertRaisesRegex(ValueError, "manifest"):
                        self._dataset()
        for key in self.manifest:
            with self.subTest(missing=key):
                manifest = dict(self.manifest)
                del manifest[key]
                self._write_manifest(manifest)
                with self.assertRaisesRegex(ValueError, "manifest"):
                    self._dataset()

    def test_missing_malformed_and_nonobject_manifest(self):
        path = self.root / "manifest.json"
        for contents in ("{", "[]", "null"):
            with self.subTest(contents=contents):
                path.write_text(contents, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "manifest"):
                    self._dataset()
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self._dataset()

    def test_extra_manifest_provenance_is_preserved(self):
        manifest = dict(self.manifest, generation_seed=12, generator_sha256="abc")
        self._write_manifest(manifest)
        self.assertEqual(self._dataset().manifest, manifest)

    def test_invalid_shape_dtype_and_nonfinite_takes_are_rejected_lazily(self):
        arrays = [
            np.zeros((127, 3960), dtype=np.float32),
            np.zeros((128, 3959), dtype=np.float32),
            np.zeros((128, 3961), dtype=np.float32),
            np.zeros((1, 128, 3960), dtype=np.float32),
            self.array.astype(np.float64), self.array.astype(np.int32),
        ]
        for array in arrays:
            with self.subTest(shape=array.shape, dtype=array.dtype):
                np.save(self.path, array, allow_pickle=False)
                dataset = self._dataset(training=False)
                with self.assertRaisesRegex(ValueError, "float32.*shape"):
                    dataset[0]
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(nonfinite=value):
                array = self.array.copy()
                array[-1, -1] = value  # Outside group zero, in the discarded gap.
                np.save(self.path, array, allow_pickle=False)
                dataset = self._dataset(training=False)
                with self.assertRaisesRegex(ValueError, "Non-finite"):
                    dataset[0]

    def test_every_newly_loaded_take_is_validated(self):
        self._write_take("val", "ins04_081.npy", np.zeros((128, 33), dtype=np.float32))
        dataset = self._dataset(training=False)
        dataset[0]
        with self.assertRaisesRegex(ValueError, "shape"):
            dataset[1]

    def test_truncated_missing_and_object_arrays_fail_safely(self):
        dataset = self._dataset(training=False)
        self.path.write_bytes(self.path.read_bytes()[:256])
        with self.assertRaisesRegex(ValueError, "Cannot load"):
            dataset[0]
        self.path.unlink()
        with self.assertRaisesRegex(ValueError, "Cannot load"):
            dataset[0]
        np.save(self.path, np.array([object()], dtype=object))
        with self.assertRaisesRegex(ValueError, "Cannot load"):
            dataset[0]

    def test_invalid_filenames_and_style_indices(self):
        for name in ("ins12_000.npy", "ins99_000.npy", "ins003_000.npy",
                     "ins3_000.npy", "ins03_00.npy", "other.npy"):
            with self.subTest(name=name):
                path = self.root / "val" / name
                path.touch()
                try:
                    with self.assertRaisesRegex(ValueError, "filename|style index"):
                        self._dataset(portion=0.001)
                finally:
                    path.unlink()

    def test_empty_split_invalid_portion_and_fragment_configuration(self):
        with self.assertRaisesRegex(ValueError, "No InsNotesMelV1 takes"):
            self._dataset("train")
        with self.assertRaisesRegex(ValueError, "train/val/test"):
            mel.InsNotesMelDataset(self.root)
        (self.root / "test").rmdir()
        with self.assertRaises(FileNotFoundError):
            self._dataset("test")
        for portion in (0, -0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(portion=portion):
                with self.assertRaisesRegex(ValueError, "portion"):
                    self._dataset(portion=portion)
        for kwargs in ({"n_fragments": 11}, {"n_fragments": 16}, {"fragment_len": 31}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "n_fragments=12.*fragment_len=32"):
                    self._dataset(**kwargs)
        for kwargs in ({"n_fragments": 12.0}, {"fragment_len": 32.0}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "must be integers"):
                    self._dataset(**kwargs)
        with self.assertRaisesRegex(ValueError, "training must"):
            self._dataset(training="eval")

    def test_index_bounds_and_negative_indices(self):
        dataset = self._dataset(training=False)
        for index in (10, -11):
            with self.assertRaises(IndexError):
                dataset[index]
            with self.assertRaises(IndexError):
                dataset.recording_index(index)
        self.assertEqual(dataset.recording_index(-1), dataset.recording_index(9))
        for a, b in zip(dataset[-1], dataset[9]):
            self.assertTrue(torch.equal(a, b))
        with self.assertRaises(TypeError):
            dataset[0.5]


if __name__ == "__main__":
    unittest.main()
