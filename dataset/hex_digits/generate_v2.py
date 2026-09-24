"""Deterministic hexadecimal extension of the corrected PhoneNums renderer."""
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
from dataloader.hex_digits_dataloader import C_LIST, S_LIST
from dataset.phonenums.number_office import NumberOffice
from dataset.phonenums.generate_v2 import sha256

REPO = Path(__file__).resolve().parents[2]
COUNTS = {'train': 10000, 'val': 1250, 'test': 1250}
ALPHABET = ''.join(C_LIST)


def unrank_hex(rank):
    if not isinstance(rank, (int, np.integer)) or not 0 <= rank < math.factorial(16):
        raise ValueError('Invalid hexadecimal permutation rank')
    remaining, result = C_LIST.copy(), []
    for n in range(15, -1, -1):
        index, rank = divmod(int(rank), math.factorial(n))
        result.append(remaining.pop(index))
    return ''.join(result)


def sample_unique_ranks(rng, count, population=math.factorial(16)):
    """Floyd sampling: O(count) memory/time, never allocate the 16! population."""
    if not 0 < count <= population:
        raise ValueError('Invalid permutation count')
    selected, ranks = set(), []
    for upper in range(population - count, population):
        candidate = int(rng.integers(0, upper + 1))
        value = upper if candidate in selected else candidate
        selected.add(value)
        ranks.append(value)
    rng.shuffle(ranks)
    return ranks


def build_specs(generation_seed=0, split_seed=0, counts=None):
    counts = COUNTS if counts is None else counts
    if (set(counts) != set(COUNTS) or any(not isinstance(v, int) or v <= 0 for v in counts.values())
            or min(generation_seed, split_seed) < 0):
        raise ValueError('Nonnegative seeds and positive integer split counts required')
    total = sum(counts.values())
    split_rng = np.random.default_rng(split_seed)
    specs = []
    for style_id, style in enumerate(S_LIST):
        rank_rng = np.random.default_rng(np.random.SeedSequence([generation_seed, style_id, 0]))
        ranks = sample_unique_ranks(rank_rng, total)
        order, assignment, offset = split_rng.permutation(total), {}, 0
        for split in COUNTS:
            for index in order[offset:offset + counts[split]]:
                assignment[int(index)] = split
            offset += counts[split]
        for index, rank in enumerate(ranks):
            text, split = unrank_hex(rank), assignment[index]
            specs.append({'path': f'{split}/{text}_{style}.png', 'split': split,
                          'style': style, 'text': text, 'page_index': index,
                          'permutation_rank': rank,
                          'page_seed': np.random.SeedSequence([generation_seed, style_id, 1, index]).generate_state(4).tolist()})
    return sorted(specs, key=lambda e: e['path'])


def render_page(task):
    root, spec = task
    import cv2
    from matplotlib import colors
    cv2.setNumThreads(1)
    path = Path(root) / spec['path']
    if path.exists():
        raise FileExistsError(path)
    office = NumberOffice(pagesize='16x1', patchsize='32x48',
                          font=str(REPO / 'dataset/phonenums/fonts/ITCKRIST.TTF'))
    rgb = tuple(int(v * 240 + 8) for v in colors.to_rgb(spec['style']))
    office.typography.printer(spec['text'], font=office.font, fg=rgb,
                              output_path=str(path), renderer_version=2,
                              rng=np.random.RandomState(spec['page_seed']))
    with Image.open(path) as image:
        if image.mode != 'RGB' or image.size != (512, 48):
            raise ValueError(f'Invalid generated image: {path}')
        image.load()
    return {**spec, 'sha256': sha256(path)}


def validate_entries(entries, counts):
    if Counter((e['split'], e['style']) for e in entries) != {
            (split, s): count for split, count in counts.items() for s in S_LIST}:
        raise ValueError('Incorrect style/split counts')
    if len({Path(e['path']).name for e in entries}) != len(entries):
        raise ValueError('Repeated filenames/permutations across splits')
    for e in entries:
        if (len(e['text']) != 16 or set(e['text']) != set(ALPHABET)
                or Path(e['path']).parts != (e['split'], f"{e['text']}_{e['style']}.png")
                or len(e['page_seed']) != 4 or len(e['sha256']) != 64):
            raise ValueError(f'Invalid entry: {e}')


def generate_dataset(output_root='../data/HexDigitsV2', generation_seed=0, split_seed=0,
                     workers=4, counts=None):
    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError(f'Refusing to overwrite {root}')
    if workers < 1:
        raise ValueError('workers must be positive')
    counts = COUNTS if counts is None else counts
    specs = build_specs(generation_seed, split_seed, counts)
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{root.name}.building-', dir=root.parent))
    print('STAGING', staging, flush=True)
    for split in COUNTS:
        (staging / split).mkdir()
    tasks = ((str(staging), spec) for spec in specs)
    def collect(results):
        entries = []
        for index, entry in enumerate(results, 1):
            entries.append(entry)
            if index % 5000 == 0:
                print(f'Rendered {index}/{len(specs)}', flush=True)
        return entries
    if workers == 1:
        entries = collect(map(render_page, tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            entries = collect(pool.map(render_page, tasks, chunksize=16))
    validate_entries(entries, counts)
    sources = {name: sha256(REPO / name) for name in (
        'dataset/hex_digits/generate_v2.py', 'dataloader/hex_digits_dataloader.py',
        'dataset/phonenums/generate_v2.py', 'dataset/phonenums/number_office.py',
        'dataset/phonenums/typography.py')}
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=REPO, text=True).strip()
    manifest = {'dataset': 'HexDigitsV2', 'version': 2, 'renderer_version': 2,
        'generation_seed': generation_seed, 'split_seed': split_seed,
        'styles': S_LIST, 'alphabet': ALPHABET, 'label_values': {c: i for i, c in enumerate(C_LIST)},
        'page_size': [512, 48], 'fragment_size': [32, 48], 'pages_per_style': counts,
        'page_count': len(entries),
        'content_style_counts': {split: {c: {s: count for s in S_LIST} for c in C_LIST}
                                for split, count in counts.items()},
        'rng': 'Floyd rank sampling; independent SeedSequence(gen,style,1,page) renderer; independent split seed',
        'renderer': {'font_size': 48, 'fit_to_bbox': False, 'background_rgb': [245, 245, 245],
                     'fixes': ['signed_float_noise_clip_then_uint8', 'complete_blur_strips'],
                     'distortions': 'unchanged PhoneNumsV2 defaults; no clean targets; no glyph offsets'},
        'font_sha256': sha256(REPO / 'dataset/phonenums/fonts/ITCKRIST.TTF'),
        'source_sha256': sources,
        'generator_sha256': hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
        'git_head': git('rev-parse', 'HEAD'), 'git_status': git('status', '--short'),
        'dependencies': {name: importlib.metadata.version(name) for name in ('numpy', 'Pillow', 'opencv-python', 'matplotlib')},
        'entries': entries}
    (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    if root.exists():
        raise FileExistsError(f'Target appeared; staging retained at {staging}')
    staging.rename(root)
    print('PUBLISHED', root, 'manifest_sha256', sha256(root / 'manifest.json'), flush=True)
    return root / 'manifest.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_root', default='../data/HexDigitsV2')
    parser.add_argument('--generation_seed', type=int, default=0)
    parser.add_argument('--split_seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not socket.gethostname().startswith('ws-'):
        raise RuntimeError('Generate only on a ws-* compute node')
    generate_dataset(**vars(args))
