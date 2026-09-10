import unittest

import numpy as np

from run_reconstruction_grid import compute_hungarian_mapping


class ReconstructionGridTest(unittest.TestCase):
    def test_hungarian_mapping_is_bijective_and_maximizes_counts(self):
        counts = np.array(
            [
                [1, 9, 0],
                [8, 0, 1],
                [0, 2, 7],
            ]
        )

        code_to_label, label_to_code, accuracy = compute_hungarian_mapping(counts)

        self.assertEqual(code_to_label, {0: 1, 1: 0, 2: 2})
        self.assertEqual(label_to_code, {1: 0, 0: 1, 2: 2})
        self.assertAlmostEqual(accuracy, 24 / 28)


if __name__ == "__main__":
    unittest.main()
