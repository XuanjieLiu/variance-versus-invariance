import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from dataset.phonenums.typography import add_gaussian_noise, random_gradual_blur
from dataset.uppercase_letters.generate_v2 import build_specs, generate_dataset, sha256
from dataloader.uppercase_letters_dataloader import get_dataloader


class UppercaseV2Test(unittest.TestCase):
    def test_signed_noise_boundaries_and_legacy(self):
        x = np.array([245, 8, 0, 255], dtype=np.uint8)
        noise = np.array([20., -20., -1., 1.])
        np.testing.assert_array_equal(add_gaussian_noise(x, noise, 2), [255, 0, 0, 255])
        np.testing.assert_array_equal(add_gaussian_noise(x, noise, 1), [9, 244, 255, 0])
        np.testing.assert_array_equal(x, [245, 8, 0, 255])

    def test_blur_remainder(self):
        x = np.full((48, 832, 3), 120, dtype=np.uint8)
        old = random_gradual_blur(x, rng=np.random.RandomState(0))
        new = random_gradual_blur(x, rng=np.random.RandomState(0), fix_remainder=True)
        self.assertTrue((old[:, -2:] == 0).all())
        np.testing.assert_array_equal(new, x)
        divisible = x[:, :830]
        np.testing.assert_array_equal(
            random_gradual_blur(divisible, rng=np.random.RandomState(0)),
            random_gradual_blur(divisible, rng=np.random.RandomState(0), fix_remainder=True))

    def test_seeds_and_balanced_disjoint_specs(self):
        counts = dict(train=4, val=2, test=2)
        first = build_specs(0, 0, counts)
        self.assertEqual(first, build_specs(0, 0, counts))
        second = build_specs(0, 1, counts)
        def images(specs):
            return {Path(e["path"]).name: e["page_seed"] for e in specs}
        self.assertEqual(images(first), images(second))
        self.assertNotEqual([e["path"] for e in first], [e["path"] for e in second])
        self.assertNotEqual(images(first), images(build_specs(1, 0, counts)))

    def test_publication_manifest_loader_and_worker_reproducibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = [Path(tmp) / "serial", Path(tmp) / "parallel"]
            for root, workers in zip(roots, (1, 2)):
                manifest_path = generate_dataset(root, workers=workers, counts=dict(train=2, val=1, test=1))
                manifest = json.loads(manifest_path.read_text())
                self.assertEqual(len(manifest["entries"]), 32)
                self.assertEqual(len(manifest["generator_sha256"]), 64)
                for e in manifest["entries"]:
                    self.assertEqual(sha256(root / e["path"]), e["sha256"])
                loader = get_dataloader(str(root / "val"), batch_size=8, shuffle=False)
                x, c, s = next(iter(loader))
                self.assertEqual(tuple(x.shape), (8, 26, 3, 48, 32))
                self.assertEqual(sorted(s[:, 0].tolist()), list(range(8)))
                self.assertTrue(all(sorted(row) == list(range(26)) for row in c.tolist()))
                entry = next(e for e in manifest["entries"] if e["split"] == "val")
                with Image.open(root / entry["path"]) as im:
                    raw = np.array(im)
                recovered = np.concatenate([f.permute(1, 2, 0).numpy() for f in x[0]], axis=1)
                np.testing.assert_array_equal(np.rint((recovered + 1) * 127.5).astype(np.uint8), raw)
                with self.assertRaises(FileExistsError):
                    generate_dataset(root, counts=dict(train=2, val=1, test=1))
            entries = [json.loads((root / "manifest.json").read_text())["entries"] for root in roots]
            self.assertEqual(entries[0], entries[1])
            (roots[0] / entries[0][0]["path"]).unlink()
            with self.assertRaises(ValueError):
                get_dataloader(str(roots[0] / entries[0][0]["split"]), batch_size=8)


if __name__ == "__main__":
    unittest.main()
