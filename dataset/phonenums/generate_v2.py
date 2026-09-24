"""Correct-color, independently seeded PhoneNums; never overwrite old datasets."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import socket
import subprocess
import tempfile

import numpy as np
from PIL import Image
from dataloader.phonenums_dataloader import S_LIST
from dataset.phonenums.number_office import NumberOffice

REPO = Path(__file__).resolve().parents[2]
COUNTS = {"train": 10000, "val": 1250, "test": 1250}
UPSTREAM = "bbbc7c9c26bda7170612814a407b76633cd22875"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def unrank_digits(rank):
    remaining, result = list('0123456789'), []
    if not 0 <= rank < math.factorial(10):
        raise ValueError('Invalid permutation rank')
    for n in range(9, -1, -1):
        index, rank = divmod(int(rank), math.factorial(n))
        result.append(remaining.pop(index))
    return ''.join(result)


def build_specs(generation_seed=0, split_seed=0, counts=None):
    counts = COUNTS if counts is None else counts
    if set(counts) != set(COUNTS) or any(int(v) != v or v <= 0 for v in counts.values()):
        raise ValueError('Positive integer train/val/test counts required')
    if min(generation_seed, split_seed) < 0 or sum(counts.values()) > math.factorial(10):
        raise ValueError('Invalid seed or more pages than unique digit permutations')
    total = sum(counts.values())
    split_rng = np.random.default_rng(split_seed)
    specs = []
    for s, style in enumerate(S_LIST):
        rank_rng = np.random.default_rng(np.random.SeedSequence([generation_seed, s, 0]))
        ranks = rank_rng.choice(math.factorial(10), total, replace=False)
        assignment, offset = {}, 0
        order = split_rng.permutation(total)
        for split in COUNTS:
            for index in order[offset:offset+counts[split]]:
                assignment[int(index)] = split
            offset += counts[split]
        for index, rank in enumerate(ranks):
            digits = unrank_digits(int(rank))
            split = assignment[index]
            specs.append({'path': f'{split}/{digits}_{style}.png', 'split': split,
                          'style': style, 'text': digits, 'page_index': index,
                          'permutation_rank': int(rank),
                          'page_seed': np.random.SeedSequence([generation_seed, s, 1, index]).generate_state(4).tolist()})
    return sorted(specs, key=lambda e: e['path'])


def render_page(task):
    root, spec = task
    import cv2
    from matplotlib import colors
    cv2.setNumThreads(1)
    path = Path(root) / spec['path']
    if path.exists():
        raise FileExistsError(path)
    office = NumberOffice(pagesize='10x1', patchsize='32x48',
                          font=str(REPO/'dataset/phonenums/fonts/ITCKRIST.TTF'))
    color = tuple(int(v*240+8) for v in colors.to_rgb(spec['style']))
    office.typography.printer(spec['text'], font=office.font, fg=color,
                              output_path=str(path), renderer_version=2,
                              rng=np.random.RandomState(spec['page_seed']))
    with Image.open(path) as im:
        if im.mode != 'RGB' or im.size != (320, 48):
            raise ValueError(f'Invalid image: {path}')
        im.load()
    return {**spec, 'sha256': sha256(path)}


def generate_dataset(output_root='../data/PhoneNumsV2', generation_seed=0, split_seed=0,
                     workers=4, counts=None):
    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError(f'Refusing to overwrite {root}')
    if workers < 1:
        raise ValueError('workers must be positive')
    counts = COUNTS if counts is None else counts
    specs = build_specs(generation_seed, split_seed, counts)
    if len({e['path'] for e in specs}) != len(specs):
        raise ValueError('Duplicate output filenames')
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{root.name}.building-', dir=root.parent))
    print('Staging:', staging, flush=True)
    for split in COUNTS:
        (staging/split).mkdir()
    tasks = [(str(staging), spec) for spec in specs]
    def collect(results):
        entries = []
        for i, e in enumerate(results, 1):
            entries.append(e)
            if i % 5000 == 0:
                print(f'Rendered {i}/{len(specs)}', flush=True)
        return entries
    if workers == 1:
        entries = collect(map(render_page, tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            entries = collect(pool.map(render_page, tasks, chunksize=16))
    expected = {(split, style): count for split, count in counts.items() for style in S_LIST}
    assert dict(Counter((e['split'], e['style']) for e in entries)) == expected
    assert len({Path(e['path']).name for e in entries}) == len(entries)
    sources = {name: sha256(REPO/name) for name in (
        'dataset/phonenums/generate_v2.py', 'dataset/phonenums/typography.py',
        'dataset/phonenums/number_office.py', 'dataloader/phonenums_dataloader.py')}
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=REPO, text=True).strip()
    manifest = {'dataset': 'PhoneNumsV2', 'version': 2, 'renderer_version': 2,
        'upstream_reference': UPSTREAM, 'generation_seed': generation_seed, 'split_seed': split_seed,
        'styles': S_LIST, 'alphabet': '0123456789', 'page_size': [320, 48], 'fragment_size': [32, 48],
        'pages_per_style': counts, 'page_count': len(entries),
        'content_style_counts': {split: {c: {s: count for s in S_LIST} for c in '0123456789'}
                                for split, count in counts.items()},
        'rng': 'unique rank choice per style; independent SeedSequence(gen,style,1,page) renderer; independent split seed',
        'renderer': {'fixes': ['signed_float_noise_clip_then_uint8', 'complete_blur_strips'],
                     'font_size': 'NumberOffice default', 'fit_to_bbox': False, 'background_rgb': [245,245,245],
                     'distortions': 'unchanged NumberOffice defaults; no clean targets'},
        'font_sha256': sha256(REPO/'dataset/phonenums/fonts/ITCKRIST.TTF'),
        'source_sha256': sources, 'generator_sha256': hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest(),
        'git_head': git('rev-parse','HEAD'), 'git_status': git('status','--short'),
        'dependencies': {n: importlib.metadata.version(n) for n in ('numpy','Pillow','opencv-python','matplotlib')},
        'entries': entries}
    (staging/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    if root.exists():
        raise FileExistsError(f'Target appeared; staging retained at {staging}')
    staging.rename(root)
    print('PUBLISHED', root, 'manifest_sha256', sha256(root/'manifest.json'), flush=True)
    return root/'manifest.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_root', default='../data/PhoneNumsV2')
    parser.add_argument('--generation_seed', type=int, default=0)
    parser.add_argument('--split_seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not socket.gethostname().startswith('ws-'):
        raise RuntimeError('Generate only on a ws-* compute node')
    generate_dataset(**vars(args))
