import unittest

import torch

from scripts.accept_insnotes_repo_pair import spectral_errors


class SpectralErrorTests(unittest.TestCase):
    def test_identity(self):
        x = torch.linspace(-5, 1, 128 * 32).reshape(1, 1, 128, 32)
        mse, shape = spectral_errors(x, x)
        self.assertEqual(float(mse), 0)
        self.assertLess(float(shape), 1e-12)

    def test_gain_invariant_shape_but_not_mse(self):
        magnitude = torch.linspace(.1, 2, 128 * 32).reshape(1, 1, 128, 32).double()
        x, y = (magnitude + 1e-5).log(), (3 * magnitude + 1e-5).log()
        mse, shape = spectral_errors(x, y)
        self.assertGreater(float(mse), 1)
        self.assertLess(float(shape), 1e-12)

    def test_frequency_change_and_nonfinite(self):
        magnitude = torch.linspace(.1, 3, 128).reshape(1, 1, 128, 1).double()
        x, y = (magnitude + 1e-5).log(), (magnitude.flip(-2) + 1e-5).log()
        mse, shape = spectral_errors(x, y)
        self.assertGreater(float(mse), 0)
        self.assertGreater(float(shape), .1)
        with self.assertRaises(ValueError):
            spectral_errors(x, torch.full_like(y, float("nan")))


if __name__ == "__main__":
    unittest.main()
