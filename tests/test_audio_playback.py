import unittest
from unittest.mock import patch
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from utils.audio_playback import AudioPlayback, invert_log_mel, preserve_audio_rng


class AudioPlaybackTests(unittest.TestCase):
    def test_magnitude_log_contract(self):
        values = np.full((128, 32), np.log(2.0 + 1e-5), dtype=np.float32)
        with patch("utils.audio_playback.librosa.feature.inverse.mel_to_stft") as mel, \
                patch("utils.audio_playback.librosa.griffinlim") as gl:
            mel.return_value = np.ones((513, 32), np.float32)
            gl.return_value = np.zeros(15872, np.float32)
            out = invert_log_mel(values, seed=11)
            np.testing.assert_allclose(mel.call_args.args[0], 2.0, rtol=1e-6)
            self.assertEqual(mel.call_args.kwargs["power"], 1.0)
            self.assertEqual(gl.call_args.kwargs["random_state"], 11)
            self.assertEqual(out.shape, (15872,))

    def test_invalid_inputs(self):
        for values in (np.zeros((127, 32)), np.full((128, 32), np.nan)):
            with self.assertRaises(ValueError):
                invert_log_mel(values)

    def test_real_inversion_seeded_finite(self):
        values = np.full((128, 4), -6., dtype=np.float32)
        first = invert_log_mel(values, iterations=2, seed=7)
        second = invert_log_mel(values, iterations=2, seed=7)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.isfinite(first).all())
        self.assertGreater(float(np.std(first)), 0)

    def test_rng_restores_even_on_failure(self):
        py, np_state, th = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        with self.assertRaises(RuntimeError), preserve_audio_rng():
            random.random()
            np.random.random()
            torch.rand(3)
            raise RuntimeError("probe")
        self.assertEqual(py, random.getstate())
        np.testing.assert_array_equal(np_state[1], np.random.get_state()[1])
        self.assertTrue(torch.equal(th, torch.get_rng_state()))

    def test_schedule_output_and_shared_gain(self):
        dataset = SimpleNamespace(s_list=[f"style{i}" for i in range(12)],
                                  c_list=[str(i) for i in range(12)], manifest_sha256="fixture")
        with tempfile.TemporaryDirectory() as root:
            writer = AudioPlayback(root, dataset, {}, {"style_ids": [0], "pitch_ids": [0]})
            self.assertTrue(writer.due(0, 1999))
            self.assertTrue(writer.due(200, 1999))
            self.assertTrue(writer.due(1999, 1999))
            self.assertFalse(writer.due(50, 1999))
            record = dict(path="val/ins00_005.npy", sha256="fixture")
            writer.references[0] = ([np.full(15872, .4)], [np.full(15872, .5)], record)
            originals = np.zeros((12, 12, 128, 32), np.float32)
            codes = np.zeros((12, 12), int)
            with patch("utils.audio_playback.invert_log_mel", return_value=np.full(15872, 2.)):
                target = writer.write("epoch00000", originals, originals, codes, epoch=0, mode="page_mean")
            manifest = json.loads((target / "manifest.json").read_text())
            self.assertEqual(manifest["examples"][0]["common_playback_gain"], .475)
            self.assertEqual(manifest["decoder_style_mode"], "page_mean")
            self.assertEqual(len(list(target.glob("*.wav"))), 3)
            self.assertIn("Model reconstruction", (target / "index.html").read_text())
            with self.assertRaises(ValueError):
                writer.write("../escape", originals, originals, codes, epoch=0, mode="fragment")

    def test_matched_repo_configs(self):
        root = Path(__file__).resolve().parents[1]
        a = yaml.safe_load((root / "configs/insnotes/cfg_vvi_insnotes_mel_repo_k12_seed0.yaml").read_text())
        b = yaml.safe_load((root / "configs/insnotes/cfg_vvi_insnotes_mel_repo_mean_k12_seed0.yaml").read_text())
        repo = yaml.safe_load((root / "cfg_v3_insnotes.yaml").read_text())
        self.assertEqual(a["model_config"]["decoder_style_mode"], "fragment")
        self.assertEqual(b["model_config"]["decoder_style_mode"], "page_mean")
        b["name"] = a["name"]
        b["model_config"]["decoder_style_mode"] = "fragment"
        self.assertEqual(a, b)
        self.assertEqual(a["optimizer_config"], repo["optimizer_config"])
        self.assertEqual(a["loss_config"]["weights"], repo["loss_config"]["weights"])
        for key in ("vq_ema_decay", "threshold_ema_dead_code"):
            self.assertEqual(a["model_config"][key], repo["model_config"][key])
        self.assertEqual(a["epochs"], 2000)
        self.assertTrue(a["save_best_validation_loss"])


if __name__ == "__main__":
    unittest.main()
