"""Read-only source acceptance, reversible BN probes, and matched audio previews.

Run on a SLURM GPU node. Uses full validation only: the source failed its gate,
so the sealed source test set is not consumed for another hyperparameter choice.
"""

import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import socket
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader
import yaml

from dataloader.insnotes_mel_dataloader import InsNotesMelDataset, C_LIST, S_LIST
from dataset.insnotes.generate_mel_dataset import render_take
from model.factory import get_model
from utils.audio_playback import invert_log_mel
from utils.codebook_metrics import batch_confusion_counts, compute_assignment_metrics, file_sha256
from utils.training_diagnostics import preserve_model_and_rng, BN_MODES


def fingerprint(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def history_summary(run):
    def rows(name):
        with (run / name).open() as handle:
            return list(csv.DictReader(handle))
    codes = rows("codebook_epoch_history.csv")
    losses = rows("loss_epoch_history.csv")
    result = {"final_train_loss": next(r for r in reversed(losses) if r["partition"] == "train"),
              "final_val_loss": next(r for r in reversed(losses) if r["partition"] == "val")}
    for key in ("one_to_one_accuracy", "codebook_purity", "macro_atom_purity"):
        best = max(codes, key=lambda r: float(r[key]))
        result["historical_peak_" + key] = {k: best[k] for k in
            ("epoch", "global_step", key, "active_codes", "usage_perplexity", "dominant_label_coverage")}
    result["retention_gate_eligible_evaluations"] = sum(
        float(r["active_codes"]) >= 12 and float(r["usage_perplexity"]) >= 10
        and float(r["dominant_label_coverage"]) >= 10 for r in codes)
    result["saved_checkpoints"] = [p.name for p in sorted(run.glob("cp_*.pt"))]
    return result


@torch.inference_mode()
def evaluate(model, loader, *, bn_mode="eval", mean_style=False):
    counts = torch.zeros(12, 12, dtype=torch.int64)
    balance = torch.zeros_like(counts)
    all_codes = []
    sse, elements = 0., 0
    with preserve_model_and_rng(model):
        model.eval()
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                if (bn_mode == "both_batch_stats"
                        or (bn_mode == "encoder_batch_stats" and name.startswith("encoder."))
                        or (bn_mode == "decoder_batch_stats" and name.startswith("decoder."))):
                    module.train()
        for x, c, s in loader:
            x = x.cuda(non_blocking=True)
            ec, zs = model.encode(x)
            eq, codes, _ = model.quantize(ec, freeze_codebook=True)
            if mean_style:
                zs = zs.mean(1, keepdim=True).expand_as(zs)
            y = model.decode(eq, zs)
            sse += float((y - x).double().square().sum())
            elements += x.numel()
            all_codes.append(codes.cpu())
            counts += batch_confusion_counts(codes.cpu(), c, 12, 12)
            balance += batch_confusion_counts(s, c, 12, 12)
    assert torch.equal(balance, torch.full_like(balance, 100)), balance
    return ({"bn_mode": bn_mode, "mean_style": mean_style,
             "notes": int(balance.sum()), "each_pitch_instrument_count": 100,
             "recon_loss": sse / elements, **compute_assignment_metrics(counts.numpy()),
             "confusion_code_by_pitch": counts.tolist()}, torch.cat(all_codes))


@torch.inference_mode()
def audio_examples(model, dataset, out, checkpoint_epoch):
    """First validation recording for styles 0/1/4; notes C/E/G, not cherry-picked."""
    manifest = dataset.manifest
    selected_styles, pitches = (0, 1, 4), (0, 4, 7)
    audio_dir = out / "audio"
    audio_dir.mkdir()
    entries = []
    for style in selected_styles:
        record = min((r for r in manifest["records"] if r["split"] == "val"
                      and r["style_index"] == style), key=lambda r: r["path"])
        with tempfile.TemporaryDirectory(prefix="insnotes-listening-") as temp:
            (Path(temp) / "val").mkdir()
            replay = render_take((temp, str(Path.home() / "miniconda3/envs/v3_audio_tools/bin/fluidsynth"),
                manifest["soundfont"], manifest["generation_seed"], style,
                record["midi_program"], record["take"], "val"), save_preview=True)
            assert replay["sha256"] == record["sha256"], "Rerender differs from immutable data"
            original, sr = sf.read(Path(temp) / "examples" / (Path(record["path"]).stem + ".wav"),
                                   dtype="float32")
        features = np.load(Path(dataset.data_dir).parent / record["path"])
        group = np.stack([features[:, p * 33:p * 33 + 32] for p in range(12)])
        reconstructed = model(torch.from_numpy(group[None]).cuda(), freeze_codebook=True)[0][0].cpu().numpy()
        tracks = {key: [] for key in ("original", "mel_roundtrip", "model_reconstruction")}
        for pitch in pitches:
            n = 31 * 512
            tracks["original"].append(original[pitch * 33 * 512:pitch * 33 * 512 + n])
            tracks["mel_roundtrip"].append(invert_log_mel(group[pitch], seed=20261114 + pitch))
            tracks["model_reconstruction"].append(invert_log_mel(reconstructed[pitch], seed=20261114 + pitch))
        peak = max(float(np.abs(wave).max()) for notes in tracks.values() for wave in notes)
        gain = min(1., .95 / max(peak, 1e-10))
        paths = {}
        for name, notes in tracks.items():
            joined = np.concatenate([part for note in notes for part in (note, np.zeros(2400, np.float32))])
            path = audio_dir / f"ins{style:02d}_{name}.wav"
            sf.write(path, gain * joined, sr, subtype="PCM_16")
            paths[name] = str(path.relative_to(out))
        entries.append({"style": S_LIST[style], "recording": record["path"], "split": "val",
                        "pitches": [C_LIST[p] for p in pitches], "rerender_sha256_verified": True,
                        "common_playback_gain": gain, "paths": paths})
    write_json(out / "audio_manifest.json", {
        "checkpoint_epoch": checkpoint_epoch, "selection": "first val recording, styles 0/1/4, C/E/G",
        "inversion": "128-band magnitude mel; exp(logmel)-1e-5; NNLS; 64-step seeded Griffin-Lim",
        "warning": "Mel phase/detail loss affects roundtrip and model clips. Originals are matched dry FluidSynth audio.",
        "gain": "One shared attenuation per instrument across all three tracks, no loudness matching",
        "examples": entries})
    cells = []
    for entry in entries:
        cells.append("<tr><th>" + html.escape(entry["style"]) + "<br>C / E / G</th>" + "".join(
            f'<td><audio controls preload="none" src="{path}"></audio></td>'
            for path in entry["paths"].values()) + "</tr>")
    (out / "audio.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>InsNotes listening</title>'
        '<style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 20px}'
        'th,td{text-align:left;padding:12px;border-bottom:1px solid #ccc}audio{width:260px}</style>'
        f'<h1>InsNotes mel: source pretrain acceptance</h1><p>Current checkpoint, epoch {checkpoint_epoch}. '
        'Matched validation notes, no training. Each clip plays C, E, G. '
        'Compare mel roundtrip with the model to separate inversion artifacts from reconstruction error.</p>'
        '<table><tr><th>Instrument</th><th>Original</th><th>Input mel roundtrip</th><th>V3 reconstruction</th></tr>'
        + "".join(cells) + '</table><p>64-step Griffin-Lim, no learned vocoder. '
        'Shared attenuation within each instrument; no independent loudness normalization.</p></html>')
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("A SLURM GPU allocation is required")
    os.chdir(ROOT)
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load((args.run / "config.yaml").read_text())
    checkpoint = json.loads((args.run / "current_checkpoint.json").read_text())["checkpoint"]
    state = torch.load(args.run / checkpoint, weights_only=False, map_location="cpu")
    model = get_model(config["dataloader"], config["model_config"]).cuda().eval()
    model.load_state_dict(state["model"], strict=True)
    model.requires_grad_(False)
    before = fingerprint(model)
    data = InsNotesMelDataset(ROOT / config["data_dir"] / "val", training=False)
    assert data.manifest_sha256 == config["dataset_manifest_sha256"]
    loader = DataLoader(data, batch_size=32, shuffle=False, num_workers=0,
                        generator=torch.Generator().manual_seed(0))
    result = {"scope": "full validation only; source test remains sealed",
              "slurm_job": os.environ["SLURM_JOB_ID"], "node": socket.gethostname(),
              "checkpoint": checkpoint, "checkpoint_sha256": file_sha256(args.run / checkpoint),
              "manifest_sha256": data.manifest_sha256, "history": history_summary(args.run)}
    probes = []
    for mode in BN_MODES:
        metrics, codes = evaluate(model, loader, bn_mode=mode)
        if mode == "eval":
            baseline_codes = codes
        metrics["code_changed_fraction_vs_eval"] = float((codes != baseline_codes).double().mean())
        if mode == "decoder_batch_stats":
            assert torch.equal(codes, baseline_codes)
        assert fingerprint(model) == before
        probes.append(metrics)
        print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_code_by_pitch"}), flush=True)
    result["reversible_bn_diagnostics"] = probes
    result["mean_style_decode_probe"], codes = evaluate(model, loader, mean_style=True)
    assert torch.equal(codes, baseline_codes)
    repeated, repeated_codes = evaluate(model, loader)
    assert torch.equal(repeated_codes, baseline_codes)
    assert repeated["recon_loss"] == probes[0]["recon_loss"]
    result["examples"] = audio_examples(model, data, args.output, int(state["epoch"]))
    assert fingerprint(model) == before
    assert all(p.grad is None for p in model.parameters())
    result["model_all_parameters_buffers_unchanged"] = True
    result["repeated_eval_identical"] = True
    result["diagnostic_warning"] = "BN batch-stat probes and mean-style probe are interventions, NOT accepted checkpoints. No test evaluation."
    write_json(args.output / "acceptance.json", result)
    print("ACCEPTANCE_DONE", args.output, flush=True)


if __name__ == "__main__":
    main()
