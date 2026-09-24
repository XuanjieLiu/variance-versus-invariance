import copy
import json
import math
from collections import Counter
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from dataset.hex_digits.generate_v2 import (build_specs, generate_dataset, unrank_hex,
                                           sample_unique_ranks, sha256, validate_entries)
from dataloader.hex_digits_dataloader import C_LIST, S_LIST, HexDigitsDataset, get_dataloader


class HexDigitsDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name) / 'serial'
        cls.manifest = json.loads(generate_dataset(cls.root, workers=1,
                                  counts=dict(train=2, val=1, test=1)).read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_full_specs_and_bounded_rank_sampling(self):
        specs = build_specs()
        self.assertEqual(len(specs), 100000)
        self.assertEqual(len({Path(e['path']).name for e in specs}), 100000)
        self.assertTrue(all(sorted(e['text']) == C_LIST for e in specs))
        self.assertEqual(Counter((e['split'], e['style']) for e in specs),
                         {(p, s): n for p, n in dict(train=10000, val=1250, test=1250).items() for s in S_LIST})
        self.assertEqual(unrank_hex(0), ''.join(C_LIST))
        self.assertEqual(unrank_hex(math.factorial(16)-1), ''.join(reversed(C_LIST)))
        ranks = sample_unique_ranks(np.random.default_rng(0), 20, 20)
        self.assertEqual(sorted(ranks), list(range(20)))
        for rank in (-1, math.factorial(16), 1.5):
            with self.assertRaises(ValueError): unrank_hex(rank)
        with self.assertRaises(ValueError): sample_unique_ranks(np.random.default_rng(0), 21, 20)

    def test_seeds_and_split_independence(self):
        kw = dict(counts=dict(train=3, val=2, test=2))
        a, b = build_specs(**kw), build_specs(split_seed=1, **kw)
        self.assertEqual(a, build_specs(**kw))
        self.assertNotEqual(a, b)
        keyed = lambda entries: {Path(e['path']).name: e['page_seed'] for e in entries}
        self.assertEqual(keyed(a), keyed(b))
        self.assertNotEqual(keyed(a), keyed(build_specs(generation_seed=1, **kw)))

    def test_parallel_bytes_and_provenance(self):
        root = Path(self.temp.name) / 'parallel'
        manifest = json.loads(generate_dataset(root, workers=2,
                              counts=dict(train=2, val=1, test=1)).read_text())
        self.assertEqual(manifest['entries'], self.manifest['entries'])
        self.assertEqual(manifest['label_values'], {c: i for i, c in enumerate(C_LIST)})
        self.assertEqual(len(manifest['font_sha256']), 64)
        self.assertEqual(len(manifest['generator_sha256']), 64)
        self.assertGreaterEqual(len(manifest['source_sha256']), 5)
        for entry in manifest['entries']:
            self.assertEqual(sha256(root / entry['path']), entry['sha256'])
            with Image.open(root / entry['path']) as image:
                self.assertEqual(image.mode, 'RGB')
                self.assertEqual(image.size, (512, 48))
                self.assertGreater(np.array(image)[:, -2:, :].sum(), 0)
        with self.assertRaises(FileExistsError): generate_dataset(root)

    def test_loader_exact_labels_rgb_and_order(self):
        loader = get_dataloader(self.root / 'val', batch_size=8, shuffle=False)
        x, c, s = next(iter(loader))
        self.assertEqual(tuple(x.shape), (8, 16, 3, 48, 32))
        self.assertTrue(all(sorted(row) == list(range(16)) for row in c.tolist()))
        self.assertEqual(sorted(s[:, 0].tolist()), list(range(8)))
        self.assertTrue(all((row == row[0]).all() for row in s))
        expected = [str(self.root / e['path']) for e in self.manifest['entries'] if e['split'] == 'val']
        self.assertEqual(loader.dataset.png_paths, expected)
        with Image.open(expected[0]) as im:
            pixels = np.array(im)
        joined = np.concatenate([f.permute(1, 2, 0).numpy() for f in x[0]], axis=1)
        np.testing.assert_array_equal(np.rint((joined+1)*127.5).astype(np.uint8), pixels)
        with self.assertRaises(ValueError): HexDigitsDataset(self.root / 'val', n_fragments=15)
        with self.assertRaises(ValueError): HexDigitsDataset(self.root / 'val', fragment_len=31)

    def test_manifest_validation_rejects_collisions_and_imbalance(self):
        entries = copy.deepcopy(self.manifest['entries'])
        counts = self.manifest['pages_per_style']
        with self.assertRaises(ValueError): validate_entries(entries[:-1], counts)
        entries[1] = entries[0]
        with self.assertRaises(ValueError): validate_entries(entries, counts)
        entries = copy.deepcopy(self.manifest['entries'])
        entries[0]['text'] = '0' * 16
        with self.assertRaises(ValueError): validate_entries(entries, counts)

    def test_loader_rejects_grayscale_and_missing_image(self):
        loader = HexDigitsDataset(self.root / 'test')
        path = Path(loader.png_paths[0]); original = path.read_bytes()
        try:
            with Image.open(path) as image: gray = image.convert('L')
            gray.save(path)
            with self.assertRaises(ValueError): loader[0]
            path.unlink()
            with self.assertRaises(ValueError): HexDigitsDataset(self.root / 'test')
        finally:
            path.write_bytes(original)


if __name__ == '__main__': unittest.main()
