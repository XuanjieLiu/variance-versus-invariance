import os
import random
import string
import json
from pathlib import Path
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms


S_LIST = [
    "black",
    "blue",
    "green",
    "red",
    "teal",
    "purple",
    "orange",
    "brown",
]

C_LIST = list(string.ascii_lowercase)


class LettersDataset(Dataset):
    def __init__(
        self,
        data_dir,
        n_fragments,
        fragment_len,
        portion=1,
        c_list=C_LIST,
        s_list=S_LIST,
    ):
        """
        data_dir: directory directly containing .png files
        n_fragments: number of fragments to cut from each sample
        fragment_len: the width of every fragment in pixels
        portion: portion of the dataset to use
        c_list: the list of possible contents
        s_list: the list of possible styles
        """
        self.data_dir = data_dir
        self.n_fragments = n_fragments
        self.fragment_len = fragment_len
        self.portion = portion
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        self.c_list = c_list
        self.s_list = s_list

        self.png_paths = glob(os.path.join(data_dir, "*.png"))
        self.manifest_path = None
        manifest_path = Path(data_dir).resolve().parent / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("dataset") == "UppercaseLettersV2":
                split = Path(data_dir).name
                entries = [e for e in manifest["entries"] if e["split"] == split]
                declared = [str(manifest_path.parent / e["path"]) for e in entries]
                if (not declared or len(set(declared)) != len(declared)
                        or set(declared) != {str(Path(p).resolve()) for p in self.png_paths}):
                    raise ValueError(f"V2 manifest/files mismatch: {Path(data_dir).resolve()}")
                for e in entries:
                    if (Path(e["path"]).parts != (split, f"{e['text']}_{e['style']}.png")
                            or len(e["text"]) != 26 or set(e["text"]) != set(c_list)
                            or e["style"] not in s_list):
                        raise ValueError(f"Invalid V2 manifest entry: {e}")
                expected = manifest["pages_per_style"][split]
                if any(sum(e["style"] == s for e in entries) != expected for s in s_list):
                    raise ValueError(f"Unbalanced V2 manifest: {manifest_path}")
                self.png_paths = declared
                self.manifest_path = str(manifest_path)
        if portion != 1:
            random.shuffle(self.png_paths)
            self.png_paths = self.png_paths[: int(len(self.png_paths) * portion)]
            self.png_paths.sort()

    def __len__(self):
        return len(self.png_paths)

    def __getitem__(self, idx):
        png_path = self.png_paths[idx]
        png_name = os.path.basename(png_path).split(".")[0]

        letters, style = png_name.split("_")
        c_labels = [self.c_list.index(letter) for letter in letters]
        s_label = self.s_list.index(style)

        img = Image.open(png_path)
        mtx = np.array(img)  # has to keep the uint8 type
        mtx = self.transform(mtx)

        fragments = []
        starting_fragment_idx = random.choice(
            range(len(c_labels) - self.n_fragments + 1)
        )
        for i in range(self.n_fragments):
            fragment_idx = starting_fragment_idx + i
            start_idx = self.fragment_len * fragment_idx
            fragment = mtx[:, :, start_idx : start_idx + self.fragment_len]
            fragments.append(fragment)

        fragments = torch.stack(fragments, dim=0)
        c_labels = c_labels[
            starting_fragment_idx : starting_fragment_idx + self.n_fragments
        ]
        s_labels = [s_label] * self.n_fragments

        c_labels = torch.tensor(c_labels)
        s_labels = torch.tensor(s_labels)

        return fragments, c_labels, s_labels


LowercaseLettersDataset = LettersDataset


def get_letters_dataloader(
    data_dir,
    batch_size,
    n_fragments=26,
    fragment_len=32,
    num_workers=0,
    portion=1,
    shuffle=True,
    distributed=False,
    c_list=C_LIST,
    s_list=S_LIST,
):
    dataset = LettersDataset(
        data_dir=data_dir,
        n_fragments=n_fragments,
        fragment_len=fragment_len,
        portion=portion,
        c_list=c_list,
        s_list=s_list,
    )
    if not distributed:
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )

    sampler = DistributedSampler(dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
    )


def get_dataloader(
    data_dir,
    batch_size,
    n_fragments=26,
    fragment_len=32,
    num_workers=0,
    portion=1,
    shuffle=True,
    distributed=False,
):
    return get_letters_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        n_fragments=n_fragments,
        fragment_len=fragment_len,
        num_workers=num_workers,
        portion=portion,
        shuffle=shuffle,
        distributed=distributed,
        c_list=C_LIST,
        s_list=S_LIST,
    )
