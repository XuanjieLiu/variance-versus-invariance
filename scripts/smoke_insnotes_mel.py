"""Full-data CUDA preflight; temporary training artifacts are always removed."""
import copy
import argparse
import csv
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import socket
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch
import yaml

from dataset.insnotes.generate_mel_dataset import render_take
from model.factory import get_model
from trainer import Trainer
from utils.codebook_metrics import file_sha256


def fingerprint(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
            [state.clone() for state in torch.cuda.get_rng_state_all()])


def assert_rng_equal(a, b):
    assert a[0] == b[0]
    assert a[1][0] == b[1][0] and a[1][2:] == b[1][2:]
    np.testing.assert_array_equal(a[1][1], b[1][1])
    assert torch.equal(a[2], b[2])
    assert all(torch.equal(x, y) for x, y in zip(a[3], b[3]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Run this preflight on a SLURM GPU node")
    os.chdir(ROOT)
    torch.set_num_threads(4)
    path = args.config or ROOT / "configs/insnotes/cfg_vvi_insnotes_mel_k12_seed0.yaml"
    config = yaml.safe_load(path.read_text())
    data = (ROOT / config["data_dir"]).resolve()
    manifest = json.loads((data / "manifest.json").read_text())
    assert file_sha256(data / "manifest.json") == config["dataset_manifest_sha256"]
    assert len(manifest["records"]) == 1200 and manifest["status"] == "complete"
    source = set(manifest["source_programs"])
    assert not source.intersection(manifest["reserved_ood_programs"])
    counts = {(split, style): 0 for split in ("train", "val", "test") for style in range(12)}
    hashes, paths = set(), set()
    for row in manifest["records"]:
        feature_path = data / row["path"]
        digest = file_sha256(feature_path)
        assert digest == row["sha256"] and digest not in hashes
        assert row["path"] not in paths
        hashes.add(digest); paths.add(row["path"])
        counts[row["split"], row["style_index"]] += 1
    for (split, _), count in counts.items():
        assert count == (80 if split == "train" else 10)
    # A real re-render verifies the recorded seed/renderer/soundfont recipe.
    first = manifest["records"][0]
    with tempfile.TemporaryDirectory(prefix="insnotes_rerender_") as directory:
        (Path(directory) / first["split"]).mkdir()
        replay = render_take((directory, str(Path.home() / "miniconda3/envs/v3_audio_tools/bin/fluidsynth"),
            manifest["soundfont"], manifest["generation_seed"], first["style_index"],
            first["midi_program"], first["take"], first["split"]))
        assert replay["sha256"] == first["sha256"], "Synthesis recipe is not reproducible"

    (ROOT / "logs").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="smoke-insnotes-mel-") as directory:
        config.update(epochs=2, log_dir=directory, name="run")
        trainer = Trainer(config)
        trainer.prepare_data()
        trainer.build_model()
        assert len(trainer.train_loader) == 30
        assert len(trainer.train_loader.dataset) == 960
        assert len(trainer.val_loader.dataset) == 1200
        batch, content, style = next(iter(trainer.val_loader))
        assert batch.shape == (32, 12, 128, 32)
        assert (content == torch.arange(12)[None]).all()
        assert (style == style[:, :1]).all()
        assert len(set(style[:, 0].tolist())) == 12
        initial = fingerprint(trainer.model)
        started = time.monotonic()
        trainer.train()
        training_seconds = time.monotonic() - started
        assert fingerprint(trainer.model) != initial
        grad_norms = {}
        for name in ("encoder", "decoder"):
            gradients = [p.grad.detach().double().square().sum() for p in getattr(trainer.model, name).parameters()
                         if p.grad is not None]
            grad_norms[name] = float(torch.stack(gradients).sum().sqrt())
            assert np.isfinite(grad_norms[name]) and grad_norms[name] > 0
        run = Path(trainer.log_dir)
        with (run / "loss_epoch_history.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        assert {int(row["epoch"]) for row in rows} == {0, 1}
        for row in rows:
            for key in ("total_loss", "recon_loss", "commit_loss", "content_loss", "style_loss", "sample_loss", "fragment_loss"):
                assert np.isfinite(float(row[key])), (key, row)
        assert (run / "cp_current_epoch1.pt").is_file(), "Final off-interval checkpoint missing"
        assert not (run / "cp_current_epoch0.pt").exists()
        assert len(list(run.glob("cp*.pt"))) <= (3 if config.get("save_best_validation_loss") else 2)
        if config.get("save_best_validation_loss"):
            best = json.loads((run / "best_validation_loss.json").read_text())
            val_rows = [row for row in rows if row["partition"] == "val"]
            selected = min(val_rows, key=lambda row: float(row["total_loss"]))
            assert best["epoch"] == int(selected["epoch"])
            assert best["validation_total_loss"] == float(selected["total_loss"])
            best_state = torch.load(run / best["checkpoint"], map_location="cpu", weights_only=False)
            assert best_state["epoch"] == best["epoch"]
            assert len(list(run.glob("cp_best_validation_loss_epoch*.pt"))) == 1
            if config["audio_diagnostics"].get("playback", {}).get("enabled", False):
                import soundfile as sf
                best_audio = run / "audio_examples/best_validation/manifest.json"
                assert json.loads(best_audio.read_text())["epoch"] == best["epoch"]
                assert json.loads(best_audio.read_text())["decoder_style_mode"] == config["model_config"]["decoder_style_mode"]
                for tag in ("epoch00000", "epoch00001", "best_validation"):
                    files = list((run / "audio_examples" / tag).glob("*.wav"))
                    assert len(files) == 9
                    for file in files:
                        wave, sr = sf.read(file)
                        assert sr == 16000 and np.isfinite(wave).all() and np.std(wave) > 0
                        assert np.abs(wave).max() <= 0.951
                state_before_audio, rng_before_audio = fingerprint(trainer.model), rng_state()
                hashes = {p.name: file_sha256(p) for p in (run / "audio_examples/best_validation").glob("*.wav")}
                trainer.audio_diagnostic.export_best_validation_audio(config, trainer.device)
                assert fingerprint(trainer.model) == state_before_audio
                assert_rng_equal(rng_before_audio, rng_state())
                assert hashes == {p.name: file_sha256(p) for p in (run / "audio_examples/best_validation").glob("*.wav")}
            del best_state
        assert (run / "audio_diagnostics/epoch00001/metrics.json").is_file()
        audio_metrics = json.loads((run / "audio_diagnostics/epoch00001/metrics.json").read_text())
        final_recon = float(next(row for row in rows if row["partition"] == "val" and int(row["epoch"]) == 1)["recon_loss"])
        np.testing.assert_allclose(final_recon, audio_metrics["recon_loss"], rtol=1e-6)
        for image_path in run.rglob("*.png"):
            pixels = np.asarray(Image.open(image_path).convert("RGB"))
            assert min(pixels.shape[:2]) > 100 and pixels.std() > 5, image_path
        for filename in ("loss_curves.png", "codebook_metrics.png", "v3_ratios.png", "audio_diagnostics.png"):
            assert (run / filename).is_file(), filename
        # Validation must leave every parameter, BN/VQ buffer, gradient and RNG unchanged.
        state_before, rng_before = fingerprint(trainer.model), rng_state()
        gradients_before = [None if p.grad is None else p.grad.detach().clone() for p in trainer.model.parameters()]
        first_eval = trainer._collect_validation()
        second_eval = trainer._collect_validation()
        # Geometry diagnostics legitimately contain NaN when no aliases exist.
        for group in ("losses", "v3_ratios", "codebook_metrics"):
            assert set(first_eval[group]) == set(second_eval[group])
            for key in first_eval[group]:
                np.testing.assert_allclose(first_eval[group][key], second_eval[group][key],
                                           rtol=0, atol=0, equal_nan=True)
        assert first_eval["fragment_count"] == 14400
        assert fingerprint(trainer.model) == state_before
        assert_rng_equal(rng_before, rng_state())
        for old, parameter in zip(gradients_before, trainer.model.parameters()):
            assert (old is None and parameter.grad is None) or torch.equal(old, parameter.grad)
        state = torch.load(run / "cp_current_epoch1.pt", map_location="cpu", weights_only=False)
        restored = get_model(config["dataloader"], config["model_config"])
        restored.load_state_dict(state["model"], strict=True)
        assert fingerprint(restored) == state_before
        assert state["epoch"] == 1 and "optimizer" in state and "scheduler" in state and "scaler" in state
        resume_config = copy.deepcopy(config)
        resume_config.update(name="resume", load_checkpoint=str(run / "cp_current_epoch1.pt"), epochs=1)
        resumed = Trainer(resume_config)
        resumed.prepare_data()
        resumed.build_model()
        assert resumed.start_epoch == 2
        assert fingerprint(resumed.model) == state_before
        assert resumed.optimizer.param_groups[0]["lr"] == trainer.optimizer.param_groups[0]["lr"]
        report = dict(status="passed", slurm_job=os.environ["SLURM_JOB_ID"], node=socket.gethostname(),
            cuda_device=torch.cuda.get_device_name(0), config_sha256=file_sha256(path),
            manifest_sha256=file_sha256(data / "manifest.json"), all_1200_hashes_verified=True,
            seeded_synthesis_reproduced=True, train_batches=60, validation_fragments=14400,
            train_shape=list(batch.shape), training_and_two_evals_seconds=training_seconds,
            gradient_norms=grad_norms, validation_deterministic=True, eval_state_rng_unchanged=True,
            initial_model_sha256=initial, decoder_style_mode=config["model_config"]["decoder_style_mode"],
            optimizer_type=type(trainer.optimizer).__name__,
            best_validation_enabled=bool(config.get("save_best_validation_loss")),
            audio_playback_enabled=bool(config["audio_diagnostics"].get("playback", {}).get("enabled", False)),
            checkpoint_strict_load=True, resume_optimizer_scheduler_restored=True,
            final_off_interval_artifacts_verified=True, temporary_training_artifacts_deleted=True,
            final_validation=first_eval)
        for active_trainer in (trainer, resumed):
            for loader in (active_trainer.train_loader, active_trainer.val_loader):
                iterator = getattr(loader, "_iterator", None)
                if iterator is not None:
                    iterator._shutdown_workers()
        del active_trainer, loader, iterator, trainer, resumed, restored, state, gradients_before
        for handler in list(logging.getLogger().handlers):
            if isinstance(handler, logging.FileHandler) and handler.baseFilename.startswith(directory + "/"):
                handler.close(); logging.getLogger().removeHandler(handler)
        torch.cuda.empty_cache()
        gc.collect()
    assert not Path(directory).exists()
    suffix = "_" + path.stem if args.config else ""
    target = ROOT / "logs" / f"insnotes_mel_preflight_{os.environ['SLURM_JOB_ID']}{suffix}.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    print("INSNOTES_MEL_PREFLIGHT_PASSED", target, flush=True)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
