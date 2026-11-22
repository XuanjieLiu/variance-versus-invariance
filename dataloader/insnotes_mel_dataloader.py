"""InsNotesMelV1: float32 natural-log mel takes, with no augmentation.

Each take contains 120 notes at a 33-frame stride. Training samples a random
cyclic run of 12 notes; evaluation visits ten nonoverlapping groups per take,
interleaving styles within each group so evaluation batches mix sources.
Only the first 32 frames of each note are used. Mel bins and stored values are
unchanged: there is no frequency masking, normalization, or zero replacement.
"""

import hashlib
import json
import operator
from pathlib import Path
import random
import re

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# Keep the legacy label order without importing its torchaudio transforms.
S_LIST = [
    "Soprano Sax", "Pipe Organ", "Accordion", "Viola", "Trumpet",
    "Muted Trumpet", "Oboe", "Clarinet", "Piccolo", "Pan Flute",
    "Harmonica", "Choir Aahs",
]
C_LIST = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SOURCE_PROGRAMS = [64, 19, 21, 41, 56, 59, 68, 71, 72, 75, 22, 52]
CONTENT_MIDI_PITCHES = list(range(60, 72))
_AUDIO = {
    "sample_rate": 16000,
    "n_fft": 1024,
    "hop_length": 512,
    "n_mels": 128,
    "note_frames": 32,
    "note_stride_frames": 33,
    "notes_per_take": 120,
}
_TAKE_NAME = re.compile(r"ins([0-9]{2})_([0-9]{3})\.npy")


class InsNotesMelDataset(Dataset):
    """Read a train/val/test directory beside the dataset's manifest.json.

    ``npy_paths`` lists selected recordings, not expanded evaluation items.
    ``eval_items[i]`` is a ``(file_index, group_index)`` pair, ordered by group,
    then local take within a style, then style. ``recording_index(i)`` maps a
    train/eval item to ``npy_paths``; do not use ``i // groups_per_take``.
    ``source_groups`` exposes recording paths per item for diagnostics.
    ``portion`` keeps the first max(1, floor(file_count * portion)) sorted takes.
    Production split counts belong to the generator, so small fixtures/subsets
    do not need additional manifest fields or all twelve styles.

    Arrays are checked lazily and a single read-only mapping is cached per
    process. Shape/dtype are checked on every load; every frame's finiteness is
    checked once per recording, since a completed dataset is immutable.
    """

    def __init__(self, data_dir, n_fragments=12, fragment_len=32, portion=1,
                 training=True):
        try:
            n_fragments = operator.index(n_fragments)
            fragment_len = operator.index(fragment_len)
        except TypeError as exc:
            raise ValueError("n_fragments and fragment_len must be integers") from exc
        if n_fragments != 12 or fragment_len != 32:
            raise ValueError("InsNotesMelV1 requires n_fragments=12 and fragment_len=32")
        if not 0 < portion <= 1:
            raise ValueError("portion must be in (0, 1]")
        if not isinstance(training, bool):
            raise ValueError("training must be a bool")

        split_dir = Path(data_dir).resolve()
        if split_dir.name not in ("train", "val", "test"):
            raise ValueError("data_dir must be an InsNotesMelV1 train/val/test directory")
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing InsNotesMelV1 split: {split_dir}")
        manifest_path = split_dir.parent / "manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        try:
            manifest = json.loads(manifest_bytes)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"Invalid InsNotesMelV1 manifest: {manifest_path}") from exc
        if not isinstance(manifest, dict):
            raise ValueError(f"Invalid InsNotesMelV1 manifest: {manifest_path}")
        audio = manifest.get("audio")
        if (manifest.get("dataset") != "InsNotesMelV1"
                or manifest.get("status") != "complete"
                or not isinstance(audio, dict)
                or any(audio.get(key) != value for key, value in _AUDIO.items())
                or manifest.get("content_midi_pitches") != CONTENT_MIDI_PITCHES
                or manifest.get("source_programs") != SOURCE_PROGRAMS):
            raise ValueError(f"Invalid InsNotesMelV1 manifest metadata: {manifest_path}")

        paths = sorted(split_dir.glob("*.npy"))
        if not paths:
            raise ValueError(f"No InsNotesMelV1 takes found: {split_dir}")
        style_indices = []
        for path in paths:
            match = _TAKE_NAME.fullmatch(path.name)
            if not path.is_file() or match is None:
                raise ValueError(f"Invalid InsNotesMelV1 filename: {path}")
            style_idx = int(match.group(1))
            if not 0 <= style_idx < len(S_LIST):
                raise ValueError(f"InsNotesMelV1 style index out of range: {path}")
            style_indices.append(style_idx)
        count = max(1, int(len(paths) * portion))

        self.data_dir = str(split_dir)
        self.split = split_dir.name
        self.n_fragments = n_fragments
        self.fragment_len = fragment_len
        self.portion = portion
        self.training = training
        self.s_list = list(S_LIST)
        self.c_list = list(C_LIST)
        self.manifest_path = str(manifest_path)
        self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        self.manifest = manifest
        self.audio = dict(audio)
        self.source_programs = list(manifest["source_programs"])
        self.content_midi_pitches = list(manifest["content_midi_pitches"])
        self.n_mels = _AUDIO["n_mels"]
        self.notes_per_take = _AUDIO["notes_per_take"]
        self.note_stride_frames = _AUDIO["note_stride_frames"]
        self.expected_shape = (self.n_mels, self.notes_per_take * self.note_stride_frames)
        self.groups_per_take = 1 if training else self.notes_per_take // n_fragments
        self.npy_paths = [str(path) for path in paths[:count]]
        self.style_indices = style_indices[:count]
        self.eval_items = None
        if not training:
            by_style = [[] for _ in self.s_list]
            for file_idx, style_idx in enumerate(self.style_indices):
                by_style[style_idx].append(file_idx)
            recording_order = [
                indices[local_take]
                for local_take in range(max(len(indices) for indices in by_style))
                for indices in by_style if local_take < len(indices)
            ]
            self.eval_items = [(file_idx, group_idx)
                               for group_idx in range(self.groups_per_take)
                               for file_idx in recording_order]
        self.source_groups = [self.npy_paths[self.recording_index(index)]
                              for index in range(len(self))]
        self._cached_path = None
        self._cached_array = None
        self._validated_paths = set()

    def __len__(self):
        return len(self.npy_paths) * self.groups_per_take

    def __getstate__(self):
        # Spawned workers should open their own mapping, not pickle array data.
        state = self.__dict__.copy()
        state["_cached_path"] = None
        state["_cached_array"] = None
        state["_validated_paths"] = set()
        return state

    def _load_take(self, path):
        if self._cached_path != path:
            self._cached_path = None
            self._cached_array = None
            try:
                array = np.load(path, mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError, EOFError) as exc:
                raise ValueError(f"Cannot load InsNotesMelV1 take: {path}") from exc
            if (not isinstance(array, np.ndarray)
                    or array.shape != self.expected_shape
                    or array.dtype != np.dtype("float32")):
                raise ValueError(
                    f"Expected float32 mel array with shape {self.expected_shape}: {path}"
                )
            if path not in self._validated_paths:
                if not np.isfinite(array).all():
                    raise ValueError(f"Non-finite InsNotesMelV1 values: {path}")
                self._validated_paths.add(path)
            self._cached_path = path
            self._cached_array = array
        return self._cached_array

    def _item_indices(self, index):
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("InsNotesMelV1 index out of range")
        return (index, 0) if self.training else self.eval_items[index]

    def recording_index(self, index):
        """Return the index in npy_paths for a train/eval item, not its group."""
        return self._item_indices(index)[0]

    def __getitem__(self, index):
        take_idx, group_idx = self._item_indices(index)
        array = self._load_take(self.npy_paths[take_idx])
        start = (random.randrange(self.notes_per_take) if self.training
                 else group_idx * self.n_fragments)
        note_indices = [(start + offset) % self.notes_per_take
                        for offset in range(self.n_fragments)]
        # Stack copies read-only slices into writable memory before torch sees it.
        fragments = np.stack([
            array[:, note * self.note_stride_frames:
                  note * self.note_stride_frames + self.fragment_len]
            for note in note_indices
        ])
        return (
            torch.from_numpy(fragments),
            torch.tensor(note_indices, dtype=torch.long) % len(self.c_list),
            torch.full((self.n_fragments,), self.style_indices[take_idx], dtype=torch.long),
        )


def _seed_worker(worker_id):
    """DataLoader already seeds torch; match Python/NumPy to that worker seed."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_dataloader(data_dir, batch_size, n_fragments=12, fragment_len=32,
                   num_workers=0, portion=1, shuffle=True, *, training=None,
                   eval_seed=0, pin_memory=True):
    """Trainer-compatible loader; ``training`` defaults to bool(shuffle).

    Evaluation has its own seeded CPU generator, including worker base seeds,
    so constructing/iterating it cannot consume the global training RNG. No
    CUDA calls occur in the dataset or worker initializer. Pinning is optional.
    """
    if training is None:
        training = bool(shuffle)
    dataset = InsNotesMelDataset(data_dir, n_fragments, fragment_len, portion,
                                 training=training)
    generator = None if training else torch.Generator().manual_seed(eval_seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=num_workers > 0,
        worker_init_fn=_seed_worker, generator=generator,
    )
