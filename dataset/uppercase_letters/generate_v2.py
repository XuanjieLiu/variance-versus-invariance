"""Deterministic, balanced V2 dataset; run only inside a compute allocation."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
from pathlib import Path
import socket
import string
import subprocess
import tempfile

import numpy as np
from PIL import Image

from dataloader.lowercase_letters_dataloader import S_LIST
from dataset.lowercase_letters.letter_office import LetterOffice

DEFAULT_COUNTS = {"train": 2600, "val": 325, "test": 325}
REPO = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_specs(generation_seed=0, split_seed=0, counts=None):
    counts = DEFAULT_COUNTS if counts is None else counts
    if set(counts) != set(DEFAULT_COUNTS) or any(int(v) != v or v <= 0 for v in counts.values()):
        raise ValueError("counts must contain positive integer train/val/test counts per style")
    if generation_seed < 0 or split_seed < 0:
        raise ValueError("Seeds must be nonnegative")
    count = sum(counts.values())
    split_rng = np.random.default_rng(split_seed)
    specs, filenames = [], set()
    for style_index, style in enumerate(S_LIST):
        order = split_rng.permutation(count)
        splits = {}
        start = 0
        for split in DEFAULT_COUNTS:
            for page_index in order[start:start + counts[split]]:
                splits[int(page_index)] = split
            start += counts[split]
        for page_index in range(count):
            text_seed, render_seed = np.random.SeedSequence(
                [int(generation_seed), style_index, page_index]
            ).spawn(2)
            text = "".join(np.random.default_rng(text_seed).permutation(list(string.ascii_uppercase)))
            filename = f"{text}_{style}.png"
            if filename in filenames:
                raise ValueError(f"Duplicate generated filename: {filename}")
            filenames.add(filename)
            split = splits[page_index]
            specs.append({"path": f"{split}/{filename}", "split": split,
                          "style": style, "text": text, "page_index": page_index,
                          "page_seed": render_seed.generate_state(4).tolist()})
    return sorted(specs, key=lambda spec: spec["path"])


def render_page(task):
    root, spec = task
    import cv2
    from matplotlib import colors
    cv2.setNumThreads(1)
    path = Path(root) / spec["path"]
    office = LetterOffice(font=str(REPO / "dataset/phonenums/fonts/ITCKRIST.TTF"),
                          alphabet=string.ascii_uppercase)
    fg = tuple(int(v * 240 + 8) for v in colors.to_rgb(spec["style"]))
    office.typography.printer(spec["text"], font=office.font, fg=fg,
                              output_path=str(path), font_size=office.font_size,
                              fit_to_bbox=True, margin=office.margin,
                              rng=np.random.RandomState(spec["page_seed"]), renderer_version=2)
    with Image.open(path) as im:
        if im.mode != "RGB" or im.size != (832, 48):
            raise ValueError(f"Invalid generated image {path}: {im.mode} {im.size}")
        im.load()
    return {**spec, "sha256": sha256(path)}


def validate_entries(entries, counts):
    expected = {(split, style): count for split, count in counts.items() for style in S_LIST}
    observed = Counter((e["split"], e["style"]) for e in entries)
    if dict(observed) != expected:
        raise ValueError(f"Unbalanced page counts: {observed}")
    if len({Path(e["path"]).name for e in entries}) != len(entries):
        raise ValueError("Repeated page filenames across splits")
    for e in entries:
        if len(e["text"]) != 26 or set(e["text"]) != set(string.ascii_uppercase):
            raise ValueError(f"Invalid alphabet: {e}")


def generate_dataset(output_root, generation_seed=0, split_seed=0, workers=4, counts=None):
    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {root}")
    counts = DEFAULT_COUNTS.copy() if counts is None else dict(counts)
    specs = build_specs(generation_seed, split_seed, counts)
    root.parent.mkdir(parents=True, exist_ok=True)
    # Build in a uniquely named sibling. On failure preserve it for diagnosis;
    # only a verified, complete dataset is ever published under output_root.
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.building-", dir=root.parent))
    print(f"Staging dataset: {staging}", flush=True)
    for split in DEFAULT_COUNTS:
        (staging / split).mkdir()
    tasks = [(str(staging), spec) for spec in specs]
    entries = []
    if int(workers) <= 1:
        results = map(render_page, tasks)
        for i, entry in enumerate(results, 1):
            entries.append(entry)
            if i % 1000 == 0:
                print(f"Rendered {i}/{len(specs)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            for i, entry in enumerate(pool.map(render_page, tasks, chunksize=8), 1):
                entries.append(entry)
                if i % 1000 == 0:
                    print(f"Rendered {i}/{len(specs)}", flush=True)
    validate_entries(entries, counts)
    source_names = ("dataset/uppercase_letters/generate_v2.py", "dataset/phonenums/typography.py",
                    "dataset/lowercase_letters/letter_office.py", "dataloader/lowercase_letters_dataloader.py")
    sources = {name: sha256(REPO / name) for name in source_names}
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()
    manifest = {
        "dataset": "UppercaseLettersV2", "version": 2, "renderer_version": 2,
        "fixes": ["signed_float_noise_clip_then_uint8", "blur_last_strip_complete"],
        "generation_seed": generation_seed, "split_seed": split_seed,
        "rng": "SeedSequence(gen,style,page).spawn(2): text=PCG64; image=RandomState(four uint32 words)",
        "styles": S_LIST, "alphabet": string.ascii_uppercase,
        "page_size": [832, 48], "fragment_size": [32, 48],
        "pages_per_style": counts, "page_count": len(entries),
        "content_style_counts": {split: {c: {s: count for s in S_LIST}
                                          for c in string.ascii_uppercase} for split, count in counts.items()},
        "distortion": "color, gaussian, salt, blur, translate (legacy salt token is a no-op)",
        "gaussian_sigma": "uniform [0,10), independently per page",
        "font": "dataset/phonenums/fonts/ITCKRIST.TTF",
        "font_sha256": sha256(REPO / "dataset/phonenums/fonts/ITCKRIST.TTF"),
        "source_sha256": sources,
        "generator_sha256": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
        "git_head": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
        "dependencies": {name: importlib.metadata.version(name)
                         for name in ("numpy", "Pillow", "opencv-python", "matplotlib")},
        "entries": entries,
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if root.exists():
        raise FileExistsError(f"Output appeared during generation; retained staging at {staging}")
    staging.rename(root)
    print(f"Published {root}; manifest_sha256={sha256(root / 'manifest.json')}", flush=True)
    return root / "manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", default="../data/UppercaseLettersV2")
    parser.add_argument("--generation_seed", type=int, default=0)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not socket.gethostname().startswith("ws-"):
        raise RuntimeError("Run dataset generation via srun/sbatch on a ws-* compute node")
    generate_dataset(**vars(args))


if __name__ == "__main__":
    main()
