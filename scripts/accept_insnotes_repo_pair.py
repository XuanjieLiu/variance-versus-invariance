"""Read-only, validation-only acceptance of the matched mel REPO source pair.

Run on a SLURM GPU node. Spectral-shape error is a reconstruction proxy, not
an instrument classifier or proof of disentanglement. Existing checkpoints and
training artifacts are never overwritten.
"""

import argparse
import csv
import html
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader
import yaml
from scipy.optimize import linear_sum_assignment

from scripts.accept_insnotes_mel import evaluate, fingerprint, history_summary, write_json
from dataloader.insnotes_mel_dataloader import InsNotesMelDataset, C_LIST, S_LIST
from model.factory import get_model
from utils.audio_diagnostics import AudioDiagnostics
from utils.codebook_metrics import batch_confusion_counts, compute_assignment_metrics, file_sha256


def spectral_errors(original, reconstruction):
    """Per-note log-mel MSE and gain-invariant mean-magnitude cosine distance."""
    x, y = original.double(), reconstruction.double()
    mse = (x - y).square().mean((-1, -2))
    x, y = [(value.exp() - 1e-5).clamp_min(0).mean(-1) for value in (x, y)]
    distance = (1 - torch.nn.functional.cosine_similarity(x, y, dim=-1)).clamp(0, 2)
    if not bool(torch.isfinite(mse).all() and torch.isfinite(distance).all()):
        raise ValueError("Nonfinite reconstruction metrics")
    return mse, distance


@torch.inference_mode()
def assess(model, loader, output, config, epoch, detailed):
    counts = torch.zeros(12, 12, dtype=torch.int64)
    by_style = torch.zeros(12, 12, 12, dtype=torch.int64)
    balance = torch.zeros_like(counts)
    sums = torch.zeros(2, 12, 12, dtype=torch.float64)
    diag = None
    if detailed:
        protocol = dict(config["audio_diagnostics"], every_n_epochs=1, final_epoch=epoch,
                        playback={"enabled": False})
        diag = AudioDiagnostics(output, loader.dataset, protocol, 12, C_LIST)
        diag.begin(epoch, (epoch + 1) * 30, config["model_config"]["decoder_style_mode"])
    before = fingerprint(model)
    all_codes = []
    for x, c, s in loader:
        x = x.cuda()
        y, ec, eq, codes, commit, zs = model(x, freeze_codebook=True)
        assert not y.requires_grad and not ec.requires_grad
        all_codes.append(codes.cpu())
        counts += batch_confusion_counts(codes.cpu(), c, 12, 12)
        by_style += torch.bincount(
            (s * 144 + codes.cpu() * 12 + c).reshape(-1), minlength=12 ** 3
        ).reshape(12, 12, 12)
        balance += batch_confusion_counts(s, c, 12, 12)
        flat = (s * 12 + c).reshape(-1)
        for index, values in enumerate(spectral_errors(x, y)):
            sums[index].view(-1).scatter_add_(0, flat, values.cpu().reshape(-1))
        if diag is not None:
            diag.collect(x, y, ec, zs, codes, c, s, None)
    assert torch.equal(balance, torch.full_like(balance, 100))
    means = sums / balance
    metrics = compute_assignment_metrics(counts.numpy())
    q_ids, labels = linear_sum_assignment(-counts.numpy())
    assert torch.equal(by_style.sum(0), counts)
    metrics["per_style_mapping"] = [{
        "style": S_LIST[i],
        "global_hungarian_accuracy": float(table[q_ids, labels].sum()) / int(table.sum()),
        "within_style_hungarian_diagnostic": compute_assignment_metrics(table.numpy())["one_to_one_accuracy"],
        "counts_code_by_pitch": table.tolist(),
    } for i, table in enumerate(by_style)]
    metrics.update({"notes": int(balance.sum()), "each_pitch_instrument_count": 100,
                    "recon_loss": float(means[0].mean()),
                    "spectral_shape_cosine_distance": float(means[1].mean()),
                    "high_pitch_G_to_B_recon_loss": float(means[0, :, 7:].mean()),
                    "high_pitch_G_to_B_spectral_shape_distance": float(means[1, :, 7:].mean()),
                    "per_style_pitch_logmel_mse": means[0].tolist(),
                    "per_style_pitch_spectral_shape_distance": means[1].tolist(),
                    "confusion_code_by_pitch": counts.tolist()})
    if diag is not None:
        metrics["probes"] = diag.finish(counts, "cuda")
    assert fingerprint(model) == before
    assert all(p.grad is None for p in model.parameters())
    metrics["parameters_buffers_unchanged"] = True
    return metrics, torch.cat(all_codes)


def listening_page(runs, output):
    folders = [run / "audio_examples/best_validation" for run in runs]
    manifests = [json.loads((p / "manifest.json").read_text()) for p in folders]
    audio = output / "audio"
    audio.mkdir()
    rows, records = [], []
    for index, (a, b) in enumerate(zip(*(m["examples"] for m in manifests))):
        for key in ("style", "recording", "source_sha256", "pitches"):
            assert a[key] == b[key], key
        common_gain = min(a["common_playback_gain"], b["common_playback_gain"])
        paths = []
        for label, arm, entry, kind in (
                ("Original", 0, a, "original"), ("Mel roundtrip", 0, a, "mel_roundtrip"),
                ("Fragment best", 0, a, "reconstruction"), ("Mean best", 1, b, "reconstruction")):
            wave, sr = sf.read(folders[arm] / entry["paths"][kind], dtype="float32")
            wave *= common_gain / entry["common_playback_gain"]
            assert sr == 16000 and np.isfinite(wave).all() and np.std(wave) > 0
            path = audio / f"style{index}_{label.lower().replace(' ', '_')}.wav"
            sf.write(path, wave, sr, subtype="PCM_16")
            paths.append(f'<td><audio controls preload="none" src="{path.relative_to(output)}"></audio></td>')
        rows.append(f'<tr><th>{html.escape(a["style"])}</th>{"".join(paths)}</tr>')
        records.append({"style": a["style"], "recording": a["recording"],
                        "pitches": a["pitches"], "common_gain": common_gain})
    (output / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Mel source comparison</title>'
        '<style>body{font:16px system-ui;max-width:1280px;margin:40px auto;padding:0 16px}'
        'th,td{text-align:left;padding:10px;border-bottom:1px solid #ddd}audio{width:240px}'
        '.table{overflow:auto}</style><h1>Fragment vs mean style: best validation checkpoints</h1>'
        '<p>Each clip plays C4, E4, G4, B4 from the same held-out recording. '
        'Compare input-mel roundtrip with reconstruction to separate inversion artifacts. '
        '64-step Griffin-Lim, fixed seed, one shared attenuation per instrument across all four tracks.</p>'
        f'<p>Fragment epoch {manifests[0]["epoch"]}; mean epoch {manifests[1]["epoch"]}. '
        'Both selected by validation total loss, not mapping or listening quality.</p>'
        '<div class="table"><table><tr><th>Instrument</th><th>Original</th><th>Input mel roundtrip</th>'
        '<th>Fragment best</th><th>Mean best</th></tr>' + ''.join(rows) + '</table></div></html>')
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fragment", type=Path, required=True)
    parser.add_argument("--mean", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("A SLURM GPU allocation is required")
    os.chdir(ROOT)
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"scope": "full validation only; test sealed; no training",
              "job": os.environ["SLURM_JOB_ID"], "node": socket.gethostname(), "arms": {},
              "spectral_proxy": "1-cos(mean_time(exp(logmel)-1e-5)); not instrument accuracy"}
    summary = []
    for arm, run in (("fragment", args.fragment), ("mean", args.mean)):
        config = yaml.safe_load((run / "config.yaml").read_text())
        dataset = InsNotesMelDataset(ROOT / config["data_dir"] / "val", training=False)
        assert dataset.manifest_sha256 == config["dataset_manifest_sha256"]
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0,
                            generator=torch.Generator().manual_seed(0))
        with (run / "codebook_epoch_history.csv").open() as f:
            history = {int(r["epoch"]): r for r in csv.DictReader(f)}
        with (run / "loss_epoch_history.csv").open() as f:
            losses = {int(r["epoch"]): r for r in csv.DictReader(f) if r["partition"] == "val"}
        arm_result = {"run": str(run), "history": history_summary(run), "checkpoints": {}}
        for role, metadata_file in (("best_validation", "best_validation_loss.json"),
                                    ("best_macro", "best_macro_atom_purity.json"),
                                    ("final", "current_checkpoint.json")):
            metadata = json.loads((run / metadata_file).read_text())
            checkpoint = run / metadata["checkpoint"]
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            epoch = int(state["epoch"])
            model = get_model(config["dataloader"], config["model_config"]).cuda().eval()
            model.load_state_dict(state["model"], strict=True)
            model.requires_grad_(False)
            del state
            directory = args.output / arm / role
            directory.mkdir(parents=True)
            metrics, codes = assess(model, loader, directory, config, epoch,
                                    detailed=role == "best_validation")
            for key in ("one_to_one_accuracy", "codebook_purity", "active_codes", "usage_perplexity"):
                assert abs(metrics[key] - float(history[epoch][key])) < 1e-8, key
            assert abs(metrics["recon_loss"] - float(losses[epoch]["recon_loss"])) < 1e-5
            metrics.update({"epoch": epoch, "checkpoint": str(checkpoint),
                            "sha256": file_sha256(checkpoint), "logged_validation_loss": losses[epoch]})
            if role == "best_validation":
                before = fingerprint(model)
                repeated, repeated_codes = evaluate(model, loader)
                assert torch.equal(codes, repeated_codes)
                assert abs(repeated["recon_loss"] - metrics["recon_loss"]) < 1e-8
                metrics["repeated_eval_identical"] = True
                probes = []
                for mode in ("decoder_batch_stats", "both_batch_stats"):
                    probe, probe_codes = evaluate(model, loader, bn_mode=mode)
                    probe["code_changed_fraction"] = float((probe_codes != codes).double().mean())
                    probes.append(probe)
                assert fingerprint(model) == before
                metrics["bn_diagnostics_only"] = probes
                audio = json.loads((run / "audio_examples/best_validation/manifest.json").read_text())
                assert audio["checkpoint"]["sha256"] == metrics["sha256"]
            metrics["source_semantic_gate_passed"] = (
                metrics["one_to_one_accuracy"] >= .95 and metrics["codebook_purity"] >= .95
                and metrics["active_codes"] == 12 and metrics["usage_perplexity"] >= 10)
            write_json(directory / "acceptance.json", metrics)
            arm_result["checkpoints"][role] = metrics
            row = {"arm": arm, "role": role, "epoch": epoch,
                   **{key: metrics[key] for key in ("one_to_one_accuracy", "codebook_purity",
                       "active_codes", "usage_perplexity", "recon_loss",
                       "spectral_shape_cosine_distance", "high_pitch_G_to_B_recon_loss",
                       "source_semantic_gate_passed")}}
            summary.append(row)
            print(json.dumps(row), flush=True)
            del model
            torch.cuda.empty_cache()
        result["arms"][arm] = arm_result
    result["listening"] = listening_page([args.fragment, args.mean], args.output)
    write_json(args.output / "acceptance.json", result)
    with (args.output / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print("ACCEPTANCE_DONE", args.output, flush=True)


if __name__ == "__main__":
    main()
