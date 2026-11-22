"""Diagnostic inversion of InsNotesMelV1; not a learned waveform decoder."""

from contextlib import contextmanager
import html
import json
import operator
from pathlib import Path
import random
import tempfile

import librosa
import numpy as np
import soundfile as sf
import torch


@contextmanager
def preserve_audio_rng():
    """Including first-use library imports and temporary checkpoint construction."""
    py_state, np_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def invert_log_mel(features, *, iterations=64, seed=0):
    """Invert natural-log MAGNITUDE mel with seeded Griffin-Lim.

    Deliberately do not normalize individual spectrograms or use power=2.
    Phase and within-band detail are unavailable, even for original inputs.
    """
    features = np.asarray(features, dtype=np.float32)
    if (features.ndim != 2 or features.shape[0] != 128
            or features.shape[1] < 2 or not np.isfinite(features).all()):
        raise ValueError("Expected finite [128, frames] natural-log mel")
    if not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    with np.errstate(over="raise", invalid="raise"):
        mel = np.maximum(np.exp(features) - 1e-5, 0)
    magnitude = librosa.feature.inverse.mel_to_stft(
        mel, sr=16000, n_fft=1024, power=1.0, norm="slaney",
        htk=False, fmin=0, fmax=8000)
    result = librosa.griffinlim(
        magnitude, n_iter=iterations, n_fft=1024, hop_length=512,
        win_length=1024, window="hann", center=True, pad_mode="constant",
        length=(features.shape[1] - 1) * 512, random_state=seed)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite waveform inversion")
    return result.astype(np.float32)


class AudioPlayback:
    """Consume fixed detached validation grids; never forward or update a model."""

    def __init__(self, root, dataset, protocol, config):
        self.root = Path(root) / "audio_examples"
        self.dataset, self.protocol, self.config = dataset, protocol, config
        self.every = operator.index(config.get("every_n_epochs", 200))
        self.iterations = operator.index(config.get("griffin_lim_iterations", 64))
        self.seed = operator.index(config.get("seed", 20261114))
        self.style_ids = list(config.get("style_ids", [0, 1, 4]))
        self.pitch_ids = list(config.get("pitch_ids", [0, 4, 7]))
        if self.every <= 0 or self.iterations <= 0:
            raise ValueError("Audio playback interval and iterations must be positive")
        for values in (self.style_ids, self.pitch_ids):
            if (not values or len(set(values)) != len(values)
                    or any(not isinstance(i, int) or not 0 <= i < 12 for i in values)):
                raise ValueError("Audio example IDs must be distinct integers in [0, 12)")
        self.references = {}

    def due(self, epoch, final_epoch):
        return epoch >= 0 and (epoch == 0 or epoch % self.every == 0 or epoch == final_epoch)

    def _reference(self, style, originals):
        if style in self.references:
            return self.references[style]
        from dataset.insnotes.generate_mel_dataset import render_take

        manifest = self.dataset.manifest
        path = Path(self.protocol["grid_items"][style]["recording"])
        record = next(r for r in manifest["records"] if r["path"] == f"val/{path.name}")
        with tempfile.TemporaryDirectory(prefix="insnotes-preview-") as temp:
            (Path(temp) / "val").mkdir()
            recreated = render_take((temp,
                str(Path.home() / "miniconda3/envs/v3_audio_tools/bin/fluidsynth"),
                manifest["soundfont"], manifest["generation_seed"], style,
                record["midi_program"], record["take"], "val"), save_preview=True)
            if recreated["sha256"] != record["sha256"]:
                raise RuntimeError("Audio preview rendering differs from the immutable dataset")
            generated = np.load(Path(temp) / record["path"])
            expected = np.stack([generated[:, i * 33:i * 33 + 32] for i in range(12)])
            np.testing.assert_array_equal(originals, expected)
            wave, sr = sf.read(Path(temp) / "examples" / f"{path.stem}.wav", dtype="float32")
            if sr != 16000:
                raise ValueError("Unexpected preview sample rate")
        notes, roundtrips = [], []
        for pitch in self.pitch_ids:
            notes.append(wave[pitch * 33 * 512:pitch * 33 * 512 + 31 * 512])
            roundtrips.append(invert_log_mel(originals[pitch], iterations=self.iterations,
                                           seed=self.seed + pitch))
        self.references[style] = (notes, roundtrips, record)
        return self.references[style]

    def write(self, tag, originals, outputs, codes, *, epoch, mode, checkpoint=None):
        if Path(tag).name != tag or tag in ("", ".", ".."):
            raise ValueError("Audio output tag must be a single directory name")
        if originals.shape != (12, 12, 128, 32) or outputs.shape != originals.shape:
            raise ValueError("Audio previews require the complete fixed 12-style/12-pitch grid")
        if codes.shape != (12, 12) or not np.isfinite(outputs).all():
            raise ValueError("Invalid audio reconstruction grid")
        directory = self.root / tag
        directory.mkdir(parents=True, exist_ok=True)
        entries, rows = [], []
        with preserve_audio_rng():
            for style in self.style_ids:
                original, roundtrip, record = self._reference(style, originals[style])
                reconstruction = [invert_log_mel(outputs[style, pitch], iterations=self.iterations,
                    seed=self.seed + pitch) for pitch in self.pitch_ids]
                tracks = dict(original=original, mel_roundtrip=roundtrip, reconstruction=reconstruction)
                peak = max(float(np.abs(note).max()) for notes in tracks.values() for note in notes)
                gain = min(1., .95 / max(peak, 1e-10))
                paths = {}
                for name, notes in tracks.items():
                    joined = np.concatenate([v for note in notes for v in (note, np.zeros(2400, np.float32))])
                    path = directory / f"ins{style:02d}_{name}.wav"
                    sf.write(path, gain * joined, 16000, subtype="PCM_16")
                    paths[name] = path.name
                entries.append(dict(style=self.dataset.s_list[style], recording=record["path"],
                    source_sha256=record["sha256"], rerender_sha256_verified=True,
                    pitches=[self.dataset.c_list[p] for p in self.pitch_ids],
                    codes=[int(codes[style, p]) for p in self.pitch_ids],
                    common_playback_gain=gain, paths=paths))
                rows.append("<tr><th>" + html.escape(self.dataset.s_list[style]) + "</th>" + "".join(
                    f'<td><audio controls preload="none" src="{p}"></audio></td>' for p in paths.values()) + "</tr>")
        payload = dict(epoch=int(epoch), completed_epochs=int(epoch) + 1, decoder_style_mode=mode,
            checkpoint=checkpoint, dataset_manifest_sha256=self.dataset.manifest_sha256,
            split="validation", selection="fixed grid: sixth sorted val recording per instrument, group 0",
            inversion=dict(power=1, n_mels=128, n_fft=1024, hop_length=512, sample_rate=16000,
                           griffin_lim_iterations=self.iterations, seed=self.seed),
            gain="shared attenuation for original, roundtrip and model within each instrument; not loudness normalized",
            examples=entries)
        (directory / "manifest.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        np.savez_compressed(directory / "reconstructions.npz", originals=originals, outputs=outputs, codes=codes)
        labels = " / ".join(self.dataset.c_list[p] for p in self.pitch_ids)
        (directory / "index.html").write_text(
            '<!doctype html><html lang="en"><meta charset="utf-8"><title>V3 audio examples</title>'
            '<style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 20px}'
            'th,td{text-align:left;padding:12px;border-bottom:1px solid #ccc}audio{width:260px}</style>'
            f'<h1>V3 mel reconstruction: {html.escape(tag)}</h1><p>Epoch {epoch} (zero-based); '
            f'{html.escape(mode)} style decoding. Pitches: {html.escape(labels)}.</p>'
            '<p>Fixed validation notes. Mel inversion loses phase and within-band detail; compare the '
            'roundtrip control with model reconstruction. No independent loudness normalization.</p>'
            '<table><tr><th>Instrument</th><th>Original</th><th>Input mel roundtrip</th>'
            '<th>Model reconstruction</th></tr>' + "".join(rows) + '</table></html>')
        # A small root index remains usable when the complete audio_examples folder is downloaded.
        links = sorted(p.parent.name for p in self.root.glob("*/index.html"))
        (self.root / "index.html").write_text('<!doctype html><meta charset="utf-8">'
            '<title>V3 listening checkpoints</title><h1>V3 listening checkpoints</h1><ul>' + "".join(
            f'<li><a href="{name}/index.html">{name}</a></li>' for name in links) + '</ul>')
        return directory
