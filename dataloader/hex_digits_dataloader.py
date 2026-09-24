"""Immutable, balanced hexadecimal pages; labels are integer values 0..15."""
import json
from collections import Counter
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
from dataloader.phonenums_dataloader import S_LIST

C_LIST = list('0123456789abcdef')
CONTENT_GROUPS = {'digits_0_9': list(range(10)), 'letters_a_f': list(range(10, 16))}


class HexDigitsDataset(Dataset):
    def __init__(self, data_dir, n_fragments=16, fragment_len=32, portion=1):
        if n_fragments != 16 or fragment_len != 32:
            raise ValueError('HexDigitsV2 requires all 16 complete 32px fragments')
        if not 0 < portion <= 1:
            raise ValueError('portion must be in (0, 1]')
        self.data_dir = str(data_dir)
        self.n_fragments, self.fragment_len = n_fragments, fragment_len
        self.c_list, self.s_list = C_LIST, S_LIST
        split_dir = Path(data_dir).resolve()
        manifest_path = split_dir.parent / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get('dataset') != 'HexDigitsV2'
                or manifest.get('alphabet') != ''.join(C_LIST)
                or manifest.get('label_values') != {c: i for i, c in enumerate(C_LIST)}
                or manifest.get('styles') != S_LIST
                or manifest.get('page_size') != [512, 48]
                or manifest.get('fragment_size') != [32, 48]):
            raise ValueError(f'Invalid HexDigitsV2 manifest: {manifest_path}')
        entries = [e for e in manifest['entries'] if e['split'] == split_dir.name]
        paths = []
        for e in entries:
            if (len(e['text']) != 16 or set(e['text']) != set(C_LIST)
                    or e['style'] not in S_LIST
                    or Path(e['path']).parts != (split_dir.name, f"{e['text']}_{e['style']}.png")):
                raise ValueError(f'Invalid HexDigitsV2 entry: {e}')
            paths.append(str(manifest_path.parent / e['path']))
        actual = {str(p) for p in split_dir.glob('*.png')}
        if not paths or len(set(paths)) != len(paths) or set(paths) != actual:
            raise ValueError(f'HexDigitsV2 manifest/files mismatch: {manifest_path}')
        expected = manifest['pages_per_style'][split_dir.name]
        if Counter(e['style'] for e in entries) != {s: expected for s in S_LIST}:
            raise ValueError('Unbalanced HexDigitsV2 manifest')
        if any(manifest['content_style_counts'][split_dir.name][c][s] != expected
               for c in C_LIST for s in S_LIST):
            raise ValueError('Incorrect content/style count declaration')
        self.png_paths, self.manifest_path = paths, str(manifest_path)
        if portion != 1:
            random.shuffle(self.png_paths)
            self.png_paths = sorted(self.png_paths[:int(len(paths) * portion)])
        self.transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((.5, .5, .5), (.5, .5, .5))])

    def __len__(self):
        return len(self.png_paths)

    def __getitem__(self, index):
        path = Path(self.png_paths[index])
        text, style = path.stem.rsplit('_', 1)
        with Image.open(path) as image:
            if image.mode != 'RGB' or image.size != (512, 48):
                raise ValueError(f'Expected RGB 512x48, got {image.mode}/{image.size}: {path}')
            pixels = np.array(image)
        tensor = self.transform(pixels)
        fragments = tensor.reshape(3, 48, 16, 32).permute(2, 0, 1, 3).contiguous()
        return (fragments, torch.tensor([int(c, 16) for c in text]),
                torch.full((16,), S_LIST.index(style), dtype=torch.long))


def get_dataloader(data_dir, batch_size, n_fragments=16, fragment_len=32,
                   num_workers=0, portion=1, shuffle=True, distributed=False):
    dataset = HexDigitsDataset(data_dir, n_fragments, fragment_len, portion)
    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle if sampler is None else False,
                      sampler=sampler, num_workers=num_workers)
