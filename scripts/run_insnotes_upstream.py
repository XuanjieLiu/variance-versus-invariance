"""Run and evaluate the pinned upstream InsNotes experiment on Slurm."""

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time


SEGMENT_END = 75
STEPS_PER_EPOCH = 30


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_dataset(data, soundfont_file, revision):
    import numpy as np

    soundfont = soundfont_file.read_text().split()[0]
    manifest = data / "manifest.json"
    if manifest.is_file():
        old = read_json(manifest)
        if (old["upstream_revision"] != revision or old["soundfont_sha256"] != soundfont
                or old["generation_seed_per_stage"] != 0):
            raise RuntimeError("Refusing a dataset with different provenance")
        for entry in old["files"]:
            path = data / entry["path"]
            if not path.is_file() or sha256_file(path) != entry["sha256"]:
                raise RuntimeError(f"Dataset file differs from recorded digest: {path}")
        return

    files = []
    names = set()
    digests = set()
    for split, count in (("train", 960), ("val", 120), ("test", 120)):
        paths = sorted((data / split).glob("*.npy"))
        if len(paths) != count:
            raise RuntimeError(f"{split}: expected {count} recordings, found {len(paths)}")
        for path in paths:
            if path.name in names:
                raise RuntimeError(f"Duplicate recording across splits: {path.name}")
            names.add(path.name)
            digest = sha256_file(path)
            if digest in digests:
                raise RuntimeError(f"Duplicate spectrogram across splits: {path}")
            digests.add(digest)
            raw = np.load(path, mmap_mode="r", allow_pickle=False)
            if raw.ndim != 2 or raw.shape[0] != 513 or raw.shape[1] < 3959 or not np.isfinite(raw).all():
                raise RuntimeError(f"Invalid upstream log-STFT features: {path}, shape={raw.shape}")
            files.append({"path": path.relative_to(data).as_posix(), "sha256": digest, "shape": list(raw.shape)})
    expected = {f"ins{instrument:02}_{take:02}.npy" for instrument in range(12) for take in range(100)}
    if names != expected:
        raise RuntimeError("The generated dataset does not contain the 12 × 100 upstream takes")
    write_json(manifest, {"upstream_revision": revision, "soundfont_sha256": soundfont,
                         "generation_seed_per_stage": 0, "files": files})


def stock_config(run, data):
    import yaml

    with Path("cfg_v3_insnotes.yaml").open() as source:
        cfg = yaml.safe_load(source)
    cfg.update(log_dir=str(run.parent), name=run.name, data_dir=str(data))
    return cfg


def smoke(run, data):
    import torch
    from trainer import Trainer

    cfg = stock_config(run, data)
    cfg["epochs"] = 1
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the stock InsNotes experiment")
    trainer = Trainer(cfg)
    trainer.prepare_data()
    trainer.build_model()
    first = next(iter(trainer.train_loader))
    if tuple(first[0].shape) != (32, 12, 512, 32):
        raise RuntimeError(f"Upstream batch has wrong shape: {first[0].shape}")
    trainer.train_loader = [first]
    trainer.val_loader = [next(iter(trainer.val_loader))]
    trainer.train()
    if not all(math.isfinite(x) for x in trainer.performance_history.values()):
        raise RuntimeError("Upstream validation loss is nonfinite")
    peak_gib = torch.cuda.max_memory_reserved() / 2**30
    print(f"UPSTREAM_SMOKE_PASS peak_reserved_GiB={peak_gib:.2f}", flush=True)


def train(run, data, segment_seconds):
    import numpy as np
    import torch
    from trainer import Trainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    original = stock_config(run, data)
    budget = original["epochs"]
    started = time.monotonic()
    resume_path = run / "slurm_boundary_state.pt"
    progress = run / "validation_selection.csv"

    class SegmentBoundary(Exception):
        pass

    class StockTrainer(Trainer):
        def _save_checkpoint(self, epoch, val_loss):
            if not math.isfinite(val_loss):
                raise RuntimeError(f"Nonfinite upstream validation loss at epoch {epoch}")
            super()._save_checkpoint(epoch, val_loss)
            initial = not progress.exists()
            with progress.open("a", newline="") as stream:
                writer = csv.writer(stream)
                if initial:
                    writer.writerow(("file_epoch", "cumulative_updates", "validation_total_loss"))
                writer.writerow((epoch, (epoch + 1) * STEPS_PER_EPOCH, val_loss))
            if epoch + 1 < budget and time.monotonic() - started >= segment_seconds:
                state = {"epoch": epoch, "performance_history": self.performance_history,
                         "scheduler": self.scheduler.state_dict(), "scaler": self.scaler.state_dict(),
                         "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                         "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}
                temp = resume_path.with_suffix(".tmp")
                torch.save(state, temp)
                temp.replace(resume_path)
                raise SegmentBoundary

    trainer = StockTrainer(copy.deepcopy(original))
    trainer.prepare_data()
    trainer.build_model()
    if len(trainer.train_loader) != STEPS_PER_EPOCH:
        raise RuntimeError("Upstream data loader does not have 30 steps per epoch")
    if resume_path.is_file():
        resume = torch.load(resume_path, map_location="cpu")
        saved = torch.load(run / f"cp_epoch{resume['epoch']}.pt", map_location="cpu")
        trainer.model.load_state_dict(saved["model"])
        trainer.optimizer.load_state_dict(saved["optimizer"])
        trainer.scheduler.load_state_dict(resume["scheduler"])
        trainer.scaler.load_state_dict(resume["scaler"])
        trainer.performance_history = resume["performance_history"]
        trainer.start_epoch = resume["epoch"] + 1
        # Upstream saves before taking its once-per-epoch scheduler step.
        trainer.scheduler.step()
        trainer.config["epochs"] = budget - trainer.start_epoch
        random.setstate(resume["python_rng"])
        np.random.set_state(resume["numpy_rng"])
        torch.set_rng_state(resume["torch_rng"])
        torch.cuda.set_rng_state_all(resume["cuda_rng"])
        print(f"UPSTREAM_SEGMENT_RESUME next_file_epoch={trainer.start_epoch}", flush=True)
    elif progress.exists():
        raise RuntimeError("Existing loss history has no planned Slurm boundary checkpoint")
    try:
        trainer.train()
    except SegmentBoundary:
        print("UPSTREAM_SEGMENT_SAVED", flush=True)
        return SEGMENT_END

    metadata = {"epochs": budget, "updates": budget * STEPS_PER_EPOCH,
                "loss_best_epoch": min(trainer.performance_history, key=trainer.performance_history.get),
                "final_epoch": budget - 1, "retained_validation_loss": trainer.performance_history}
    write_json(run / "training_complete.json", metadata)
    if resume_path.exists():
        resume_path.unlink()
    print("UPSTREAM_TRAINING_COMPLETE", metadata, flush=True)
    return 0


def evaluate(run, data):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from tester import Tester
    from utils.eval_utils import confusion_mtx_acc
    from utils.training_utils import setup_seed

    metadata = read_json(run / "training_complete.json")
    results = {}
    for role in ("loss_best", "final"):
        epoch = metadata[f"{role}_epoch"]
        cfg = stock_config(run, data)
        cfg["active_checkpoint"] = str(run / f"cp_epoch{epoch}.pt")
        setup_seed(0)
        tester = Tester(cfg)
        tester.output_dir = str(run / "evaluation" / role)
        tester.prepare_data()
        tester.build_model()
        tester.test(confusion_mtx=True)
        labels = np.array([sample[1] for sample in tester.ground_truth], dtype=int)
        style_labels = np.array([sample[2] for sample in tester.ground_truth], dtype=int)
        codes = np.array(tester.sample_vq_indices).ravel()
        counts = np.zeros((12, 12), dtype=int)
        np.add.at(counts, (codes, labels), 1)
        if counts.sum() != 1440 or not np.all(counts.sum(axis=0) == 120):
            raise RuntimeError("Unexpected sample count in the upstream InsNotes test")
        code_ids, pitches = linear_sum_assignment(-counts)
        mapping = dict(zip(code_ids.tolist(), pitches.tolist()))
        decoded = np.array([mapping[code] for code in codes])
        rows = counts / (counts.sum(axis=1, keepdims=True) + 1e-7)
        results[role] = {
            "file_epoch": epoch, "split": "test", "recordings": 120, "notes": 1440,
            "evaluation_seed": 0, "upstream_codebook_accuracy": float(confusion_mtx_acc(rows)),
            "hungarian_one_to_one": float(np.mean(decoded == labels)),
            "weighted_purity": float(counts.max(axis=1).sum() / counts.sum()),
            "active_codes": int(np.count_nonzero(counts.sum(axis=1))),
            "count_by_code_and_pitch": counts.tolist(), "global_hungarian_map": mapping,
            "per_instrument_hungarian": {
                name: {"n": int((style_labels == i).sum()),
                       "accuracy": float(np.mean(decoded[style_labels == i] == labels[style_labels == i]))}
                for i, name in enumerate(tester.S_LIST) if (style_labels == i).any()
            },
        }
    write_json(run / "upstream_test_results.json", results)
    print("UPSTREAM_TEST_RESULTS", json.dumps(results), flush=True)


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Run through Slurm, never on the login node")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("smoke", "train", "evaluate"):
        item = sub.add_parser(command)
        item.add_argument("run", type=Path)
        item.add_argument("data", type=Path)
        if command == "train":
            item.add_argument("--segment-seconds", type=int, default=20 * 3600)
    item = sub.add_parser("audit")
    item.add_argument("data", type=Path)
    item.add_argument("soundfont_file", type=Path)
    item.add_argument("revision")
    args = parser.parse_args()
    if args.command == "audit":
        audit_dataset(args.data, args.soundfont_file, args.revision)
    else:
        # Use pinned upstream modules from the isolated working directory.
        sys.path.insert(0, str(Path.cwd()))
        if args.command == "train":
            raise SystemExit(train(args.run, args.data, args.segment_seconds))
        (smoke if args.command == "smoke" else evaluate)(args.run, args.data)


if __name__ == "__main__":
    main()
