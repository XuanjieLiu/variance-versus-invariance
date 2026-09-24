"""GPU-node publication audit: every PNG/hash plus an original-only glyph grid."""
from concurrent.futures import ProcessPoolExecutor
import datetime
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch
from dataset.hex_digits.generate_v2 import validate_entries, sha256
from dataloader.hex_digits_dataloader import C_LIST, S_LIST, get_dataloader


def verify_file(task):
    root, entry = task
    path = Path(root) / entry['path']
    if sha256(path) != entry['sha256']:
        raise ValueError(f'PNG checksum mismatch: {path}')
    with Image.open(path) as im:
        if im.mode != 'RGB' or im.size != (512, 48):
            raise ValueError(f'Invalid PNG: {path}')
        im.load()
    return 1


def main():
    assert socket.gethostname().startswith('ws-') and torch.cuda.is_available()
    os.chdir(ROOT); torch.set_num_threads(4)
    root = (ROOT / '../data/HexDigitsV2').resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    validate_entries(manifest['entries'], manifest['pages_per_style'])
    assert manifest['page_count'] == len(manifest['entries']) == 100000
    for relative, digest in manifest['source_sha256'].items():
        assert sha256(ROOT / relative) == digest, relative
    assert sha256(ROOT / 'dataset/phonenums/fonts/ITCKRIST.TTF') == manifest['font_sha256']
    with ProcessPoolExecutor(max_workers=4) as pool:
        total = sum(pool.map(verify_file, ((str(root), e) for e in manifest['entries']), chunksize=64))
    assert total == 100000
    for split, size in [('train', 80000), ('val', 10000), ('test', 10000)]:
        loader = get_dataloader(root / split, batch_size=32, shuffle=False)
        assert len(loader.dataset) == size
        x, c, s = next(iter(loader))
        assert x.shape == (32, 16, 3, 48, 32)
        assert all(sorted(row) == list(range(16)) for row in c.tolist())
        assert ((x >= -1) & (x <= 1)).all()
        if split == 'val': validation = loader.dataset
    selected = [next(i for i, p in enumerate(validation.png_paths)
                     if Path(p).stem.rsplit('_', 1)[-1] == style) for style in S_LIST]
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(16, 8, figsize=(12, 20))
    for col, index in enumerate(selected):
        x, labels, _ = validation[index]
        for row, char in enumerate(C_LIST):
            patch = x[int((labels == row).nonzero()[0])]
            ax = axes[row, col]
            ax.imshow((patch.permute(1, 2, 0).numpy() + 1) / 2)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0: ax.set_ylabel(char, rotation=0, labelpad=14, fontsize=15)
            if row == 0: ax.set_title(S_LIST[col])
    fig.suptitle('HexDigitsV2 — generated inputs only (no model/reconstruction)')
    fig.tight_layout(rect=(0, 0, 1, .975))
    output = ROOT / 'logs/diagnostics/20260915_hex_dataset'
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / 'hex_inputs_16x8.png', dpi=150); plt.close(fig)
    report = {'validated_at': datetime.datetime.now().astimezone().isoformat(),
              'node': socket.gethostname(), 'manifest_sha256': sha256(root / 'manifest.json'),
              'verified_png_count': total, 'all_png_hashes_and_rgb_dimensions_checked': True,
              'source_and_font_hashes_checked': True, 'split_counts': {'train': 80000, 'val': 10000, 'test': 10000},
              'content_style_counts': manifest['content_style_counts'],
              'preview_pages': [validation.png_paths[i] for i in selected]}
    (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print('HEX_DATASET_AUDIT_OK', report['manifest_sha256'], total, flush=True)


if __name__ == '__main__': main()
