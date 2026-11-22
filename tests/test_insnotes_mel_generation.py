"""CPU tests; invoke on a SLURM compute node."""
import unittest

import numpy as np

from dataset.insnotes.generate_mel_dataset import AUDIO, build_envelope, log_mel, split_takes


class MelGenerationTests(unittest.TestCase):
    def test_stratified_disjoint_split(self):
        split = split_takes(100, 17, 2)
        self.assertEqual(split, split_takes(100, 17, 2))
        self.assertNotEqual(split, split_takes(100, 17, 3))
        self.assertEqual({s: list(split.values()).count(s) for s in set(split.values())},
                         {"train": 80, "val": 10, "test": 10})
        self.assertEqual(set(split), set(range(100)))

    def test_envelope_bounds_and_alignment(self):
        stride = 33 * 512
        length = 120 * stride
        actual = build_envelope(length, np.random.default_rng(7))
        self.assertTrue(np.isfinite(actual).all())
        self.assertGreaterEqual(float(actual.min()), 0.8 - 1e-6)
        self.assertLessEqual(float(actual.max()), 1.2 + 1e-6)
        np.testing.assert_array_equal(actual, build_envelope(length, np.random.default_rng(7)))
        # Gaps use a constant held endpoint, not the next note's ramp.
        for i in (0, 1, 119):
            gap = actual[i * stride + 16000:(i + 1) * stride]
            self.assertEqual(float(gap.min()), float(gap.max()))

    def test_log_mel_is_finite_exact_shape_and_not_linear_stft(self):
        samples = 120 * 33 * 512
        time = np.arange(samples, dtype=np.float32) / AUDIO["sample_rate"]
        wave = 0.25 * np.sin(2 * np.pi * 440 * time)
        result = log_mel(wave)
        self.assertEqual(result.shape, (128, 3960))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.isfinite(result).all())
        self.assertGreater(float(np.std(result)), 0.1)


if __name__ == "__main__":
    unittest.main()
