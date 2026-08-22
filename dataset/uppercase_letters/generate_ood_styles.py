from __future__ import annotations

import argparse
import hashlib
import json
import random
import string
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dataset.lowercase_letters.letter_office import LetterOffice


SEEN_STYLE_NAMES = (
    "black",
    "blue",
    "green",
    "red",
    "teal",
    "purple",
    "orange",
    "brown",
)
OOD_STYLE_NAMES = (
    "pink",
    "salmon",
    "gold",
    "lime",
    "cyan",
    "magenta",
    "gray",
    "peru",
)
SPLIT_SEEDS = {"adapt": 4100, "val": 4200, "test": 4300}
DEFAULT_PAGES_PER_STYLE = {"adapt": 256, "val": 64, "test": 256}
GENERATOR_VERSION = 1


@dataclass(frozen=True)
class PageSpec:
    split: str
    style: str
    text: str
    image_seed: int
    output_path: str


def build_page_specs(
    output_root: str | Path,
    pages_per_style: dict[str, int],
    split_seeds: dict[str, int] = SPLIT_SEEDS,
) -> list[PageSpec]:
    output_root = Path(output_root)
    specs: list[PageSpec] = []
    names: set[str] = set()
    for split in ("adapt", "val", "test"):
        split_seed = int(split_seeds[split])
        count = int(pages_per_style[split])
        for style_idx, style in enumerate(OOD_STYLE_NAMES):
            for page_idx in range(count):
                image_seed = split_seed * 100_000 + style_idx * count + page_idx
                if not 0 <= image_seed < 2**32:
                    raise ValueError(f"Generated image seed is outside uint32: {image_seed}")
                rng = random.Random(image_seed)
                letters = list(string.ascii_uppercase)
                rng.shuffle(letters)
                text = "".join(letters)
                filename = f"{text}_{style}.png"
                relative_name = f"{split}/{filename}"
                if relative_name in names:
                    raise RuntimeError(f"Duplicate generated page name: {relative_name}")
                names.add(relative_name)
                specs.append(
                    PageSpec(
                        split=split,
                        style=style,
                        text=text,
                        image_seed=image_seed,
                        output_path=str(output_root / split / filename),
                    )
                )
    return specs


def _render_page(spec: PageSpec) -> str:
    from matplotlib import colors

    np.random.seed(spec.image_seed)
    office = LetterOffice(
        pagesize="26x1",
        patchsize="32x48",
        alphabet=string.ascii_uppercase,
        dataset_name="uppercase letter OOD color",
    )
    color = tuple(int(channel * 240 + 8) for channel in colors.to_rgb(spec.style))
    office.typography.printer(
        spec.text,
        font=office.font,
        fg=color,
        output_path=spec.output_path,
        font_size=office.font_size,
        fit_to_bbox=True,
        margin=office.margin,
    )
    return spec.output_path


def _manifest_digest(specs: list[PageSpec]) -> str:
    digest = hashlib.sha256()
    for spec in specs:
        digest.update(f"{spec.split}\0{spec.style}\0{spec.text}\0{spec.image_seed}\n".encode())
    return digest.hexdigest()


def generate_dataset(
    output_root: str | Path,
    pages_per_style: dict[str, int],
    workers: int,
) -> Path:
    output_root = Path(output_root).resolve()
    if output_root.exists() and any(output_root.rglob("*.png")):
        raise FileExistsError(
            f"Refusing to mix data with existing PNG files under {output_root}"
        )
    for split in ("adapt", "val", "test"):
        (output_root / split).mkdir(parents=True, exist_ok=True)

    specs = build_page_specs(output_root, pages_per_style)
    if int(workers) <= 1:
        for spec in specs:
            _render_page(spec)
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            list(executor.map(_render_page, specs, chunksize=8))

    manifest = {
        "generator_version": GENERATOR_VERSION,
        "seen_style_names": list(SEEN_STYLE_NAMES),
        "ood_style_names": list(OOD_STYLE_NAMES),
        "style_sets_disjoint": set(SEEN_STYLE_NAMES).isdisjoint(OOD_STYLE_NAMES),
        "split_seeds": SPLIT_SEEDS,
        "pages_per_style": pages_per_style,
        "page_count": len(specs),
        "page_size": [832, 48],
        "fragment_size": [32, 48],
        "alphabet": string.ascii_uppercase,
        "font": "dataset/phonenums/fonts/ITCKRIST.TTF",
        "distortion": "color, gaussian, salt, blur, translate",
        "spec_sha256": _manifest_digest(specs),
        "splits": {
            split: {
                style: sum(1 for spec in specs if spec.split == split and spec.style == style)
                for style in OOD_STYLE_NAMES
            }
            for split in ("adapt", "val", "test")
        },
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_root",
        "--output_dir",
        dest="output_root",
        default="../data/UppercaseLettersOODColors",
    )
    parser.add_argument("--adapt_pages_per_style", type=int, default=256)
    parser.add_argument("--val_pages_per_style", type=int, default=64)
    parser.add_argument("--test_pages_per_style", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    pages_per_style = {
        "adapt": args.adapt_pages_per_style,
        "val": args.val_pages_per_style,
        "test": args.test_pages_per_style,
    }
    manifest_path = generate_dataset(args.output_root, pages_per_style, args.workers)
    print(f"Generated OOD uppercase dataset: {manifest_path}")


if __name__ == "__main__":
    main()
