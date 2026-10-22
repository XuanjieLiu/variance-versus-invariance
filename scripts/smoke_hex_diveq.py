"""DiVeQ preflight:100 training steps and full10k-page validation per arm.

Temporary smoke runs are deleted immediately, retaining only compact reports.
No formal training or historical-artifact mutation. Run on a ws-* GPU node.
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
import subprocess
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
from model.factory import get_model
from test_training_diagnostics import assert_equal, rng_state
from utils.codebook_metrics import file_sha256


class FirstBatches:
    def __init__(self, loader, batches=100):
        self.loader, self.batches, self.dataset = loader, batches, loader.dataset
    def __len__(self): return self.batches
    def __iter__(self): return itertools.islice(self.loader, self.batches)


def smoke(dim, report_root):
    path = ROOT / f"configs/hex/v2/cfg_vvi_hex_k16_diveq_c{dim}_s{dim}_seed0.yaml"
    config = yaml.safe_load(path.read_text())
    config.update(epochs=1, snapshot_every_n_epochs=1)
    with tempfile.TemporaryDirectory(prefix=f"smoke-diveq-c{dim}-", dir=ROOT / "logs") as directory:
        config.update(log_dir=directory, name="run")
        trainer = Trainer(config)
        trainer.prepare_data(); trainer.build_model()
        run = Path(trainer.log_dir)
        full_loader = trainer.train_loader
        assert len(full_loader.dataset) == 80000 and len(full_loader) == 2500
        assert len(trainer.val_loader.dataset) == 10000
        assert trainer.model.active_decoder_style_mode == "fragment"
        assert trainer.model.vq_code_dim == dim and not trainer.loss.rank_enabled
        atom_parameter = trainer.model.vq._codebook.embed
        assert any(p is atom_parameter for g in trainer.optimizer.param_groups for p in g['params'])
        initial_atoms = atom_parameter.detach().clone()
        initial = json.loads((run / "initial_model_state.json").read_text())
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
        assert trainer.model.vq.iter_counter.item() == 100
        assert not trainer.model.vq.replacement_pending.item()
        assert not torch.equal(atom_parameter, initial_atoms)
        assert torch.isfinite(atom_parameter.grad).all() and atom_parameter.grad.norm() > 0
        with (run / "loss_epoch_history.csv").open() as handle: rows = list(csv.DictReader(handle))
        assert {row["partition"] for row in rows} == {"train", "val"}
        for row in rows:
            assert float(row["commit_loss"]) == 0
            assert np.isfinite(float(row["total_loss"]))
        for name in ("loss_curves.png", "codebook_metrics.png", "v3_ratios.png",
                     "foreground_color_metrics.png", "disentanglement_probes.png",
                     "training_codebook_usage.png", "bn_diagnostics.png", "optimization_scales.png",
                     "per_style_codebook_metrics.png", "content_group_metrics.png", "quantizer_config.json",
                     "cp_current_epoch0.pt", "cp_best_validation_loss_epoch0.pt", "cp_snapshot_epoch0.pt"):
            assert (run / name).is_file(), name
        matrices = list((run / "reconstruction_diagnostics").glob("*column-normalized*.json"))
        assert len(matrices) == 1
        matrix = json.loads(matrices[0].read_text())
        np.testing.assert_array_equal(matrix["content_style_counts"], np.full((16, 8), 1250))
        for ext in (".png", ".svg", ".csv"): assert matrices[0].with_suffix(ext).is_file()
        assert list((run / "reconstruction_diagnostics").glob("reconstruction*fragment.png"))
        state = torch.load(run / "cp_current_epoch0.pt", map_location="cpu", weights_only=False)
        restored = get_model(config["dataloader"], config["model_config"])
        restored.load_state_dict(state["model"], strict=True)
        assert restored.vq.iter_counter.item() == 100
        optimizer = torch.optim.AdamW(restored.parameters(), lr=.001)
        optimizer.load_state_dict(state['optimizer'])
        assert optimizer.state[restored.vq._codebook.embed]['step'].item() == 100
        assert len(list(run.glob("cp*.pt"))) <= 4
        grad_norm = atom_parameter.grad.norm().item()
        trainer.optimization_scale_monitor.close()
        del restored, optimizer, state, initial_atoms, trainer
        torch.cuda.empty_cache()
        subprocess.run([sys.executable, 'run_evaluation.py', '--run', str(run),
                        '--active_checkpoint', 'current', '--confusion_mtx',
                        '--test_subset_size', '512', '--test_subset_seed', '0',
                        '--test_subset_strategy', 'style_stratified'], check=True)
        eval_files = list((run / 'vis').glob('*test-n512*.json'))
        assert len(eval_files) == 1
        report = {"dim": dim, "node": socket.gethostname(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
                  "config_sha256": file_sha256(path), "dataset_manifest_sha256": config["dataset_manifest_sha256"],
                  "initial_model": initial, "first_epoch_sampler_sha256": order_hash,
                  "bn_pages_sha256": json.loads((run / "bn_diagnostic_protocol.json").read_text())["pages_sha256"],
                  "training_batches": 100, "validation_pages": 10000, "validation_fragments": 160000,
                  "loss_rows": rows, "codebook_gradient_norm": grad_norm,
                  "diagnostics_state_rng_neutral": True, "strict_checkpoint_optimizer_load": True,
                  "replacement_hook_exercised": True, "evaluation_alias": True, "smoke_artifacts_removed": True}
        for handler in list(logging.getLogger().handlers):
            if isinstance(handler, logging.FileHandler) and str(handler.baseFilename).startswith(directory + "/"):
                handler.close(); logging.getLogger().removeHandler(handler)
    assert not Path(directory).exists()
    (report_root / f"c{dim}s{dim}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"DIVEQ_PREFLIGHT_PASSED dim={dim} artifacts_deleted={directory}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert socket.gethostname().startswith("ws-") and torch.cuda.is_available()
    os.chdir(ROOT); torch.set_num_threads(4)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    a, b = [smoke(dim, output) for dim in (512, 4)]
    for key in ("dataset_manifest_sha256", "bn_pages_sha256"):
        assert a[key] == b[key], key
    # Different widths necessarily have different initial tensors. Record, do not
    # claim identical model initialization or sampler order across architectures.
    (output / "pair.json").write_text(json.dumps({"node": socket.gethostname(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"), "same_data_and_fixed_diagnostic_pages": True,
        "same_first_epoch_sampler_order": a['first_epoch_sampler_sha256'] == b['first_epoch_sampler_sha256'],
        "source_sha256": {name: file_sha256(ROOT / name) for name in (
            "model/diveq.py", "model/v3_loss.py", "model/autoencoder.py", "trainer.py")}}, indent=2) + "\n")
    print("DIVEQ_PAIR_PREFLIGHT_PASSED", flush=True)


if __name__ == "__main__": main()
