"""Versioned, recording-disjoint InsNotes reproduction with log-mel inputs.

Run on a compute node. Existing datasets are never modified. Notes retain the
paper's 1s duration, 0.056s gap, cyclic pitches and randomized velocity/envelope.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import librosa
import numpy as np
import pretty_midi
import soundfile as sf


SOURCE_PROGRAMS = [64, 19, 21, 41, 56, 59, 68, 71, 72, 75, 22, 52]
OOD_PROGRAMS = [0, 4, 73, 78]
AUDIO = dict(sample_rate=16000, n_fft=1024, hop_length=512, n_mels=128,
             note_frames=32, note_stride_frames=33, notes_per_take=120,
             note_duration_seconds=1.0, note_stride_seconds=1.056,
             mel_power=1.0, mel_norm="slaney", mel_htk=False,
             fmin=0.0, fmax=8000.0, center=True, log_epsilon=1e-5)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_envelope(length, rng):
    """Independent bounded envelopes aligned to actual note onsets, not 1s bins."""
    sr = AUDIO["sample_rate"]
    stride = AUDIO["note_stride_frames"] * AUDIO["hop_length"]
    envelope = np.ones(length, dtype=np.float32)
    for note in range(AUDIO["notes_per_take"]):
        start = note * stride
        stop = min(start + sr, length)
        low, high = rng.uniform(0.8, 1.2, size=2)
        position = np.linspace(0, 1, stop - start, dtype=np.float32)
        if rng.integers(2):
            position = np.sin(position * np.pi / 2)
        envelope[start:stop] = low + (high - low) * position
        envelope[stop:min(start + stride, length)] = high
    return envelope


def log_mel(wave):
    mel = librosa.feature.melspectrogram(
        y=wave, sr=AUDIO["sample_rate"], n_fft=AUDIO["n_fft"],
        hop_length=AUDIO["hop_length"], n_mels=AUDIO["n_mels"],
        power=AUDIO["mel_power"], norm=AUDIO["mel_norm"],
        htk=AUDIO["mel_htk"], fmin=AUDIO["fmin"], fmax=AUDIO["fmax"],
        center=AUDIO["center"], pad_mode="constant")
    frames = AUDIO["notes_per_take"] * AUDIO["note_stride_frames"]
    result = np.log(mel[:, :frames] + AUDIO["log_epsilon"]).astype(np.float32)
    if result.shape != (128, 3960) or not np.isfinite(result).all():
        raise ValueError(f"Invalid mel array: {result.shape}")
    return result


def split_takes(n_takes, seed, style):
    if n_takes < 10 or n_takes % 10:
        raise ValueError("n_takes must be a positive multiple of 10")
    order = np.random.default_rng(np.random.SeedSequence([seed, style, 991])).permutation(n_takes)
    n = n_takes // 10
    return {int(take): ("val" if i < n else "test" if i < 2 * n else "train")
            for i, take in enumerate(order)}


def render_take(task, *, save_preview=False):
    root, fluidsynth, soundfont, seed, style, program, take, split = task
    rng = np.random.default_rng(np.random.SeedSequence([seed, program, take]))
    name = f"ins{style:02d}_{take:03d}"
    output = Path(root) / split / f"{name}.npy"
    with tempfile.TemporaryDirectory(prefix="insnotes_render_") as temporary:
        mid_path, wav_path = Path(temporary) / "take.mid", Path(temporary) / "take.wav"
        midi = pretty_midi.PrettyMIDI(resolution=1000)
        instrument = pretty_midi.Instrument(program=program)
        velocities = []
        for note in range(AUDIO["notes_per_take"]):
            velocity = int(rng.integers(80, 121))
            velocities.append(velocity)
            start = note * AUDIO["note_stride_seconds"]
            instrument.notes.append(pretty_midi.Note(
                velocity=velocity, pitch=60 + note % 12, start=start, end=start + 1))
        midi.instruments.append(instrument)
        midi.write(str(mid_path))
        command = [fluidsynth, "-ni", "-r", "16000", "-g", "0.2", "-F", str(wav_path),
                   "-T", "wav", "-O", "float", "-o", "synth.cpu-cores=1",
                   "-o", "synth.reverb.active=0", "-o", "synth.chorus.active=0",
                   soundfont, str(mid_path)]
        subprocess.run(command, check=True, capture_output=True, timeout=180)
        wave, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        if sr != AUDIO["sample_rate"]:
            raise ValueError("Unexpected renderer sample rate")
        wave = wave.mean(axis=1)
        n_samples = AUDIO["notes_per_take"] * AUDIO["note_stride_frames"] * AUDIO["hop_length"]
        if len(wave) < n_samples:
            wave = np.pad(wave, (0, n_samples - len(wave)))
        wave = wave[:n_samples] * build_envelope(n_samples, rng)
        peak = float(np.max(np.abs(wave)))
        if not np.isfinite(wave).all() or peak < 1e-7:
            raise ValueError(f"Silent/nonfinite synthesis: {name}")
        wave *= 0.9 / peak
        features = log_mel(wave)
        np.save(output, features, allow_pickle=False)
        # One source-only example per instrument makes synthesis auditable.
        if take == 0 or save_preview:
            example_dir = Path(root) / "examples"
            example_dir.mkdir(exist_ok=True)
            sf.write(example_dir / f"{name}.wav", wave[:12 * 33 * 512], sr, subtype="PCM_16")
    return dict(path=f"{split}/{name}.npy", style_index=style, midi_program=program,
                take=take, split=split, seed_components=[seed, program, take],
                note_count=120, velocities=velocities, sha256=sha256(output),
                feature_min=float(features.min()), feature_max=float(features.max()))


def generate(destination, soundfont, fluidsynth, seed=20261106, n_takes=100, workers=4):
    destination = Path(destination).resolve()
    soundfont = Path(soundfont).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite dataset: {destination}")
    executable = shutil.which(fluidsynth)
    if not executable or not soundfont.is_file():
        raise FileNotFoundError("FluidSynth executable and soundfont are required")
    destination.mkdir(parents=True)
    for split in ("train", "val", "test"):
        (destination / split).mkdir()
    manifest = dict(dataset="InsNotesMelV1", status="building", generation_seed=seed,
        audio=AUDIO, content_midi_pitches=list(range(60, 72)), source_programs=SOURCE_PROGRAMS,
        reserved_ood_programs=OOD_PROGRAMS, takes_per_style=n_takes,
        split_policy="stratified 80/10/10 by complete rendered MIDI take",
        soundfont=str(soundfont), soundfont_sha256=sha256(soundfont),
        generator_sha256=sha256(__file__),
        renderer_version=subprocess.check_output([executable, "--version"], text=True).strip(),
        renderer_effects=dict(reverb=False, chorus=False, gain=0.2),
        versions={name: importlib.metadata.version(name) for name in ("numpy", "librosa", "soundfile", "pretty_midi")},
        preprocessing="natural log of Slaney mel magnitude + 1e-5; no dataset normalization",
        paper_deviations=["128 mel bands instead of 512 linear STFT bins",
            "bounded per-note envelope aligned to 1.056s onsets",
            "dry synthesis: reverb and chorus disabled to avoid note-tail mixing",
            "stratified deterministic whole-recording split"],
        records=[])
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    tasks = []
    for style, program in enumerate(SOURCE_PROGRAMS):
        splits = split_takes(n_takes, seed, style)
        for take in range(n_takes):
            tasks.append((str(destination), executable, str(soundfont), seed, style, program, take, splits[take]))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, record in enumerate(pool.map(render_take, tasks), 1):
            manifest["records"].append(record)
            if i % 12 == 0 or i == len(tasks):
                print(f"rendered {i}/{len(tasks)}", flush=True)
    hashes = [row["sha256"] for row in manifest["records"]]
    if len(set(hashes)) != len(hashes):
        raise RuntimeError("Duplicate rendered features found; dataset remains incomplete")
    manifest["status"] = "complete"
    manifest["take_counts"] = {split: sum(r["split"] == split for r in manifest["records"])
                               for split in ("train", "val", "test")}
    manifest["fragments_per_style_content"] = {
        split: n_takes * (8 if split == "train" else 1) for split in ("train", "val", "test")}
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(manifest_path)
    print(json.dumps({"manifest": str(manifest_path), "sha256": sha256(manifest_path),
                      "take_counts": manifest["take_counts"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--takes-per-style", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--soundfont", default=str(Path(__file__).parent / "soundfonts/GeneralUser GS v1.471.sf2"))
    parser.add_argument("--fluidsynth", default=str(Path.home() / "miniconda3/envs/v3_audio_tools/bin/fluidsynth"))
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Generate audio on a SLURM compute node, not the login node")
    generate(args.output, args.soundfont, args.fluidsynth, args.seed, args.takes_per_style, args.workers)


if __name__ == "__main__":
    main()
