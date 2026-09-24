import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from utils.foreground_color import fragment_color_metrics, ForegroundColorAccumulator
from utils.confusion_diagnostics import (column_probabilities, column_conserving_hundredths,
    validate_balanced_counts, content_style_counts, save_confusion_diagnostics, text_color)


class ForegroundTests(unittest.TestCase):
    def inputs(self):
        x = torch.ones(2, 3, 20, 20)
        x[:, :, 7:13, 7:13] = -1
        return x

    def test_identity_and_empty_mask(self):
        x = self.inputs()
        acc = ForegroundColorAccumulator(['a', 'b'])
        acc.update(x, x, torch.tensor([0, 1]))
        result = acc.result()
        self.assertEqual(result['foreground_rgb_mae'], 0)
        self.assertEqual(result['foreground_chroma_rmse'], 0)
        self.assertEqual(result['foreground_valid_count'], 2)
        blank = torch.ones_like(x)
        acc = ForegroundColorAccumulator(['a', 'b'])
        acc.update(blank, blank, torch.tensor([0, 1]))
        self.assertTrue(math.isnan(acc.result()['foreground_rgb_mae']))
        self.assertEqual(acc.result()['foreground_invalid_count'], 2)

    def test_color_background_and_unclipped_prediction(self):
        x = self.inputs()
        y = x.clone()
        y[:, 0, 7:13, 7:13] = 1
        metrics = fragment_color_metrics(x, y)
        torch.testing.assert_close(metrics['foreground_rgb_mae'], torch.full((2,), 1/3))
        self.assertGreater(float(metrics['foreground_chroma_rmse'][0]), .4)
        self.assertGreater(float(metrics['foreground_rgb_mae'][0]), ((y-x)/2).abs().mean().item()*10)
        y[:, 0, 7:13, 7:13] = 9
        large = fragment_color_metrics(x, y)
        self.assertGreater(float(large['foreground_rgb_mae'][0]), 1)
        self.assertGreater(float(large['prediction_out_of_range_fraction'][0]), 0)
        torch.testing.assert_close(metrics['mask'], large['mask'])

    def test_equal_fragment_weight_and_per_style(self):
        x = self.inputs()
        x[1, :, 5:15, 5:15] = -1
        y = x.clone()
        y[0] += 1
        acc = ForegroundColorAccumulator(['a', 'b'])
        acc.update(x, y, torch.tensor([0, 1]))
        result = acc.result()
        self.assertAlmostEqual(result['foreground_rgb_mae'], .25)
        self.assertAlmostEqual(result['per_style']['a']['foreground_rgb_mae'], .5)
        self.assertEqual(result['per_style']['b']['foreground_rgb_mae'], 0)


class ConfusionTests(unittest.TestCase):
    def test_rounding_and_balance(self):
        counts = np.array([[1, 2], [1, 1], [1, 0]])
        p = column_probabilities(counts)
        np.testing.assert_allclose(p.sum(0), 1)
        units = column_conserving_hundredths(p)
        np.testing.assert_array_equal(units.sum(0), 100)
        self.assertTrue(np.all(np.abs(units/100-p) <= .01))
        with self.assertRaises(ValueError):
            validate_balanced_counts([[3, 2], [3, 3]])
        with self.assertRaises(ValueError):
            validate_balanced_counts([[3], [3]], counts*2)
        with self.assertRaises(ValueError):
            column_probabilities([[0, 1], [0, 2]])
        pairs = content_style_counts(torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1]), 2, 2)
        np.testing.assert_array_equal(pairs, np.ones((2, 2)))

    def test_hungarian_inactive_rectangular_and_rng(self):
        counts = np.array([[0, 10], [10, 0], [0, 0]])
        with tempfile.TemporaryDirectory() as temp:
            before = torch.random.get_rng_state().clone()
            np_before = np.random.get_state()
            prefix = Path(temp) / 'matrix'
            result = save_confusion_diagnostics(prefix, counts, np.full((2, 2), 5),
                                               ['A', 'B'], ['x', 'y'], 'test')
            payload = json.loads(prefix.with_suffix('.json').read_text())
            self.assertEqual(payload['one_to_one_accuracy'], 1)
            self.assertEqual(payload['row_code_order'], [1, 0, 2])
            for suffix in ['.png', '.svg', '.json', '.csv']:
                self.assertTrue(prefix.with_suffix(suffix).exists())
            torch.testing.assert_close(before, torch.random.get_rng_state())
            after = np.random.get_state()
            np.testing.assert_array_equal(np_before[1], after[1])
            self.assertEqual(np_before[2:], after[2:])
        self.assertEqual(text_color([1, 1, 1]), 'black')
        self.assertEqual(text_color([0, 0, 0]), 'white')


if __name__ == '__main__':
    unittest.main()
