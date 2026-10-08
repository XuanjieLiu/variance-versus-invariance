import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from plot_hex_codebook_geometry import mapping_from_counts, get_mapping


class GeometryMappingTests(unittest.TestCase):
    def test_permuted_mapping_for_supported_sizes(self):
        for size in (10, 16, 26):
            counts = np.roll(np.eye(size) * 12, 3, axis=0)
            mapping, accuracy = mapping_from_counts(counts)
            self.assertEqual(accuracy, 1.)
            for code, label in mapping.items(): self.assertEqual(counts[code, label], 12)

    def test_inactive_and_invalid_counts(self):
        mapping, accuracy = mapping_from_counts([[4, 2], [0, 0]])
        self.assertEqual(len(mapping), 2)
        self.assertAlmostEqual(accuracy, 4/6)
        for counts in ([[0, 0]], [[float('nan')]], [[-1]], [1, 2]):
            with self.assertRaises(ValueError): mapping_from_counts(counts)

    def test_reuses_hash_verified_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp); directory = run / 'acceptance_example'; directory.mkdir()
            checkpoint = run / 'cp_best.pt'
            report = {'checkpoint_sha256': 'test-digest', 'split': 'test', 'pages': 10,
                      'content_labels': ['0', '1'], 'raw_counts': [[0, 10], [10, 0]]}
            path = directory / 'codebook_confusion_matrix__cp_best__test-full.json'
            path.write_text(json.dumps(report))
            result = get_mapping(run, checkpoint, 'test-digest', {}, {}, 2, ['0', '1'])
            self.assertEqual(result[0], {0: 1, 1: 0})
            self.assertEqual(result[1], 1.)
            self.assertFalse(result[-1])
            with self.assertRaises(ValueError):
                get_mapping(run, checkpoint, 'test-digest', {}, {}, 2, ['1', '0'])


if __name__ == '__main__': unittest.main()
