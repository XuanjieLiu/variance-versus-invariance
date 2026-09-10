import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import yaml

from model.autoencoder import CSAE
from model.v3_loss import V3Loss
from trainer import Trainer
from utils.disentanglement_monitor import fixed_page_split, code_style_probe, linear_readouts


class TinyEncoder(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        return x[..., :2], x[..., 2:]


class TinyDecoder(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, c, s):
        return torch.cat((c, s), -1)


def tiny_config(mode=None):
    config = dict(n_channels=1, n_feature=1, fragment_len=1,
                  d_emb_c=2, d_emb_s=2, n_atoms=3)
    if mode is not None:
        config.update(decoder_style_mode=mode, decoder_style_warmup_epochs=100)
    return config


class StyleBypassTest(unittest.TestCase):
    def test_only_decoder_style_is_averaged_and_gradients_are_shared(self):
        model = CSAE(tiny_config("page_mean"), TinyEncoder, TinyDecoder)
        x = torch.randn(2, 4, 4, requires_grad=True)
        model.eval()
        output, ec, q, indices, commit, raw_style = model(x, freeze_codebook=True)
        torch.testing.assert_close(raw_style, x[..., 2:])
        torch.testing.assert_close(output[..., 2:], x[..., 2:].mean(1, keepdim=True).expand(-1, 4, -1))
        output[:, 0, 2:].sum().backward()
        torch.testing.assert_close(x.grad[..., 2:], torch.full((2, 4, 2), 0.25))

    def test_warmup_boundary_and_checkpoint_restore(self):
        model = CSAE(tiny_config("page_mean_warmup"), TinyEncoder, TinyDecoder)
        model.set_decoder_epoch(99)
        state = copy.deepcopy(model.state_dict())
        model.set_decoder_epoch(100)
        self.assertEqual(model.active_decoder_style_mode, "fragment")
        restored = CSAE(tiny_config("page_mean_warmup"), TinyEncoder, TinyDecoder)
        restored.load_state_dict(state, strict=True)
        restored.eval()
        self.assertEqual(restored.active_decoder_style_mode, "page_mean")
        restored.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(restored.active_decoder_style_mode, "fragment")

    def test_legacy_state_unchanged_and_control_initialization_identical(self):
        torch.manual_seed(0)
        legacy = CSAE(tiny_config(), TinyEncoder, TinyDecoder)
        torch.manual_seed(0)
        control = CSAE(tiny_config("fragment"), TinyEncoder, TinyDecoder)
        self.assertNotIn("_decoder_epoch", legacy.state_dict())
        for key, value in legacy.state_dict().items():
            torch.testing.assert_close(value, control.state_dict()[key])
        clone = CSAE(tiny_config(), TinyEncoder, TinyDecoder)
        clone.load_state_dict(legacy.state_dict(), strict=True)

    def test_global_style_collapse_is_not_zero_v3_loss(self):
        loss = V3Loss({"relativity": 15, "weights": {"recon_loss": 1}, "monitor_raw_mpd": True})
        c = torch.randn(4, 5, 3)
        x = torch.randn(4, 5, 3)
        result = loss.compute_loss(x, c, c, torch.tensor(0.), torch.zeros_like(c), x)
        self.assertEqual(float(result["style_loss"]), 1)
        self.assertEqual(float(result["sample_loss"]), 1)
        self.assertGreaterEqual(float(result["v3_loss"]), 2)

    def test_fixed_probe_split_is_page_disjoint_balanced_and_order_independent(self):
        paths = [f"/data/{i:03d}_{style}.png" for style in ("blue", "red") for i in range(10)]
        dataset = SimpleNamespace(s_list=["blue", "red"], png_paths=paths)
        rng_before = np.random.get_state()
        fit, score, grid = fixed_page_split(dataset, 16, 0)
        self.assertFalse(set(fit) & set(score))
        self.assertEqual(len(fit), 8)
        self.assertEqual(len(score), 8)
        for ids in (fit, score):
            self.assertEqual(sum(paths[i].endswith("_blue.png") for i in ids), 4)
        reversed_ds = SimpleNamespace(s_list=dataset.s_list, png_paths=paths[::-1])
        other = fixed_page_split(reversed_ds, 16, 0)
        for ids, reversed_ids in zip((fit, score, grid), other):
            self.assertEqual([paths[i] for i in ids], [reversed_ds.png_paths[i] for i in reversed_ids])
        np.testing.assert_array_equal(rng_before[1], np.random.get_state()[1])

    def test_code_lookup_and_scale_aware_probe(self):
        y = torch.arange(3).repeat(100)
        values = code_style_probe(y, y, y, y, 3, 3)
        self.assertEqual(values["ec_vq_style_accuracy"], 1)
        generator = torch.Generator().manual_seed(7)
        x = torch.cat((1000 * torch.randn(600, 1, generator=generator),
                       torch.nn.functional.one_hot(y.repeat(2), 3).float() * 1e-3), 1)
        result = linear_readouts(x[:300], x[300:], {"label": (y, y, 3)}, 4)
        self.assertGreater(result["label_whitened_accuracy"], .99)

    def test_periodic_snapshots_survive_current_and_best_rotation(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer.__new__(Trainer)
            trainer.log_dir = directory
            trainer.model = nn.Linear(2, 2)
            trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=.1)
            trainer.scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, 1)
            trainer.scaler = torch.amp.GradScaler("cuda", enabled=False)
            trainer.best_macro_atom_purity = trainer.best_macro_val_loss = None
            trainer.best_macro_epoch = trainer.best_macro_stage = None
            trainer._save_best_macro_checkpoint(24, .2, .7)
            trainer._save_current_checkpoint(24, .2, .7)
            trainer._save_periodic_checkpoint(24)
            trainer._save_best_macro_checkpoint(49, .1, .8)
            trainer._save_current_checkpoint(49, .1, .8)
            trainer._save_periodic_checkpoint(49)
            self.assertEqual(len(list(Path(directory).glob("cp*.pt"))), 4)
            self.assertTrue((Path(directory) / "cp_snapshot_epoch24.pt").exists())
            state = torch.load(Path(directory) / "cp_snapshot_epoch24.pt", weights_only=False)
            self.assertEqual(state["epoch"], 24)
            self.assertIn("optimizer", state)
            self.assertIn("scheduler", state)

    def test_configs_match_historical_training_protocol(self):
        root = Path(__file__).resolve().parents[1]
        historical = yaml.safe_load((root / "logs/20260807-1506__VVI-RQ1A-C128-K26-S0/config.yaml").read_text())
        configs = [yaml.safe_load(path.read_text()) for path in sorted((root / "configs/uppercase/rq2").glob("cfg_vvi_rq2_c128_k26_*seed0.yaml"))]
        self.assertEqual(len(configs), 3)
        for config in configs:
            self.assertEqual(config["optimizer_config"], historical["optimizer_config"])
            self.assertEqual(config["loss_config"]["weights"], historical["loss_config"]["weights"])
            self.assertEqual(config["macro_best_health_gate"], historical["macro_best_health_gate"])
            for key, value in historical["model_config"].items():
                self.assertEqual(config["model_config"][key], value)
            self.assertEqual(config["epochs"], 200)
            self.assertIsNone(config["load_checkpoint"])
            self.assertEqual(config["snapshot_every_n_epochs"], 25)


if __name__ == "__main__":
    unittest.main()
