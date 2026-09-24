import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from dataset.phonenums.generate_v2 import build_specs, generate_dataset, unrank_digits, sha256
from dataloader.phonenums_dataloader import get_dataloader


class PhoneNumsV2Test(unittest.TestCase):
    def test_full_specs_are_unique_balanced_and_reproducible(self):
        specs = build_specs()
        self.assertEqual(len(specs), 100000)
        self.assertEqual(len({Path(e['path']).name for e in specs}), 100000)
        self.assertTrue(all(sorted(e['text']) == list('0123456789') for e in specs))
        self.assertEqual(unrank_digits(0), '0123456789')
        self.assertEqual(unrank_digits(3628799), '9876543210')
        tiny = dict(train=4,val=2,test=2)
        a = build_specs(counts=tiny)
        self.assertEqual(a, build_specs(counts=tiny))
        b = build_specs(split_seed=1, counts=tiny)
        self.assertNotEqual([e['path'] for e in a], [e['path'] for e in b])
        self.assertEqual({Path(e['path']).name: e['page_seed'] for e in a},
                         {Path(e['path']).name: e['page_seed'] for e in b})
        self.assertNotEqual(a, build_specs(generation_seed=1, counts=tiny))

    def test_parallel_publication_and_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = [Path(directory)/'serial', Path(directory)/'parallel']
            manifests = []
            for root, workers in zip(roots, (1,2)):
                manifest = json.loads(generate_dataset(root, workers=workers, counts=dict(train=2,val=1,test=1)).read_text())
                manifests.append(manifest)
                for e in manifest['entries']:
                    self.assertEqual(sha256(root/e['path']), e['sha256'])
                    with Image.open(root/e['path']) as im:
                        self.assertEqual(im.mode, 'RGB')
                        self.assertEqual(im.size, (320,48))
                x,c,s = next(iter(get_dataloader(root/'val', batch_size=8,n_fragments=10,shuffle=False)))
                self.assertEqual(tuple(x.shape), (8,10,3,48,32))
                self.assertEqual(sorted(s[:,0].tolist()), list(range(8)))
                self.assertTrue(all(sorted(row)==list(range(10)) for row in c.tolist()))
                entry = next(e for e in manifest['entries'] if e['split']=='val')
                with Image.open(root/entry['path']) as im:
                    pixels=np.array(im)
                reconstructed=np.concatenate([f.permute(1,2,0).numpy() for f in x[0]],axis=1)
                np.testing.assert_array_equal(np.rint((reconstructed+1)*127.5).astype(np.uint8),pixels)
                with self.assertRaises(FileExistsError):
                    generate_dataset(root)
            self.assertEqual(manifests[0]['entries'], manifests[1]['entries'])
            with self.assertRaises(ValueError):
                get_dataloader(roots[0]/'val',batch_size=8,n_fragments=9)
            (roots[0]/manifests[0]['entries'][0]['path']).unlink()
            with self.assertRaises(ValueError):
                get_dataloader(roots[0]/manifests[0]['entries'][0]['split'],batch_size=8,n_fragments=10)
