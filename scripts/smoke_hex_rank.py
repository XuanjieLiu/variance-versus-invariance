"""Four real training batches + full validation for each v3_rank Hex MEAN arm.

No formal runs; temporary checkpoints/images are removed immediately after
validation. Keep a compact preflight report only. Run on a ws-* GPU node.
"""
import argparse
import copy
import csv
import hashlib
import itertools
import json
import logging
import os
from pathlib import Path
import random
import socket
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import numpy as np
import torch
import yaml
from trainer import Trainer
from tester import Tester
from model.factory import get_model
from model.rank_regularization import RANK_LOG_COLUMNS
from test_training_diagnostics import assert_equal, rng_state
from utils.codebook_metrics import file_sha256


class FirstBatches:
    def __init__(self, loader, batches=4):
        self.loader, self.batches = loader, batches
        self.dataset = loader.dataset

    def __len__(self): return self.batches

    def __iter__(self): return itertools.islice(self.loader, self.batches)


def smoke(rank, report_root):
    path = ROOT / f"configs/hex/v2/cfg_vvi_hex_k16_mean_rank{rank}_seed0.yaml"
    config = yaml.safe_load(path.read_text())
    config.update(epochs=1, snapshot_every_n_epochs=1)
    with tempfile.TemporaryDirectory(prefix=f"smoke-v3_rank-r{rank}-", dir=ROOT / "logs") as directory:
        config.update(log_dir=directory, name="run")
        trainer = Trainer(config)
        trainer.prepare_data()
        trainer.build_model()
        full_loader = trainer.train_loader
        assert len(full_loader.dataset) == 80000 and len(full_loader) == 2500
        assert len(trainer.val_loader.dataset) == 10000
        assert trainer.model.active_decoder_style_mode == "page_mean"
        initial = json.loads((Path(trainer.log_dir) / "initial_model_state.json").read_text())
        before = rng_state()
        order = np.asarray(list(iter(full_loader.sampler)), dtype=np.int64)
        order_hash = hashlib.sha256(order.tobytes()).hexdigest()
        random.setstate(before[0]); np.random.set_state(before[1])
        torch.set_rng_state(before[2]); torch.cuda.set_rng_state_all(before[3])
        trainer.train_loader = FirstBatches(full_loader)
        for monitor in (trainer.bn_diagnostic, trainer.disentanglement_monitor, trainer.per_style_codebook_monitor):
            original = monitor.finish
            def checked(*args, _original=original, **kwargs):
                state, rng = copy.deepcopy(trainer.model.state_dict()), rng_state()
                optimizer = copy.deepcopy(trainer.optimizer.state_dict())
                flags = [module.training for module in trainer.model.modules()]
                result = _original(*args, **kwargs)
                assert_equal(unittest.TestCase(), state, trainer.model.state_dict())
                assert_equal(unittest.TestCase(), rng, rng_state())
                assert_equal(unittest.TestCase(), optimizer, trainer.optimizer.state_dict())
                assert flags == [module.training for module in trainer.model.modules()]
                return result
            monitor.finish = checked
        trainer.train()
        run = Path(trainer.log_dir)
        with (run / "loss_epoch_history.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        assert {row["partition"] for row in rows} == {"train", "val"}
        for row in rows:
            assert all(np.isfinite(float(row[key])) for key in RANK_LOG_COLUMNS)
            assert float(row["rank_target"]) == rank
            assert np.isclose(float(row["weighted_rank_loss"]), .01 * float(row["rank_loss"]))
        for name in ("loss_curves.png", "codebook_metrics.png", "v3_ratios.png",
                     "foreground_color_metrics.png", "disentanglement_probes.png",
                     "training_codebook_usage.png", "bn_diagnostics.png", "optimization_scales.png",
                     "per_style_codebook_metrics.png", "content_group_metrics.png",
                     "cp_current_epoch0.pt", "cp_best_validation_loss_epoch0.pt", "cp_snapshot_epoch0.pt"):
            assert (run / name).is_file(), name
        matrices = list((run / "reconstruction_diagnostics").glob("*column-normalized*.json"))
        assert len(matrices) == 1
        matrix = json.loads(matrices[0].read_text())
        np.testing.assert_array_equal(matrix["content_style_counts"], np.full((16, 8), 1250))
        for ext in (".png", ".svg", ".csv"): assert matrices[0].with_suffix(ext).is_file()
        assert list((run / "reconstruction_diagnostics").glob("reconstruction*page_mean.png"))
        state = torch.load(run / "cp_current_epoch0.pt", map_location="cpu", weights_only=False)
        restored = get_model(config["dataloader"], config["model_config"])
        restored.load_state_dict(state["model"], strict=True)
        assert restored.active_decoder_style_mode == "page_mean"
        assert len(list(run.glob("cp*.pt"))) <= 4  # smoke forces one snapshot
        # Exercise the evaluation model resolver with method: v3_rank as well.
        tester = Tester.__new__(Tester)
        tester.config = {**config, "active_checkpoint": str(run / "cp_current_epoch0.pt")}
        tester.device = torch.device("cuda")
        tester.build_model()
        assert tester.loss.rank_enabled
        report = {"target_rank": rank, "method": "v3_rank", "node": socket.gethostname(),
                  "slurm_job": os.environ.get("SLURM_JOB_ID"), "config_sha256": file_sha256(path),
                  "dataset_manifest_sha256": config["dataset_manifest_sha256"],
                  "initial_model": initial, "first_epoch_sampler_sha256": order_hash,
                  "bn_pages_sha256": json.loads((run / "bn_diagnostic_protocol.json").read_text())["pages_sha256"],
                  "training_batches": 4, "validation_pages": 10000,
                  "validation_fragments": 160000, "loss_rows": rows,
                  "diagnostics_state_rng_neutral": True, "strict_checkpoint_load": True,
                  "evaluation_alias": True, "smoke_artifacts_removed": True}
        trainer.optimization_scale_monitor.close()
        del tester, restored, state, trainer
        for handler in list(logging.getLogger().handlers):
            if isinstance(handler, logging.FileHandler) and str(handler.baseFilename).startswith(directory + "/"):
                handler.close(); logging.getLogger().removeHandler(handler)
        torch.cuda.empty_cache()
    assert not Path(directory).exists()
    (report_root / f"rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"V3_RANK_PREFLIGHT_PASSED rank={rank} artifacts_deleted={directory}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert socket.gethostname().startswith("ws-") and torch.cuda.is_available()
    os.chdir(ROOT); torch.set_num_threads(4)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    a, b = [smoke(rank, output) for rank in (4, 2)]
    for key in ("initial_model", "first_epoch_sampler_sha256", "dataset_manifest_sha256", "bn_pages_sha256"):
        assert a[key] == b[key], key
    (output / "matched.json").write_text(json.dumps({"matched_initialization_order_diagnostics": True,
        "node": socket.gethostname(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "source_sha256": {name: file_sha256(ROOT / name) for name in (
            "model/rank_regularization.py", "model/v3_loss.py", "model/autoencoder.py", "trainer.py")}}, indent=2) + "\n")
    print("V3_RANK_MATCHED_PREFLIGHT_PASSED", flush=True)


if __name__ == "__main__": main()
