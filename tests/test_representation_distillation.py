"""Objective boundaries and source-only RD-V3 protocol checks (run on SLURM)."""

import copy
from pathlib import Path
import unittest

import torch
from torch import nn
import yaml

from model.distillation_loss import (
    RepresentationDistillationLoss, calibrate_teacher, validate_distillation_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/hex/rd/cfg_rd_v3_hex_rank2_lr1e4_seed0.yaml"


class TinyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 6), nn.BatchNorm1d(6))
        self.register_buffer("atoms", torch.randn(3, 3))

    def encode(self, x):
        values = self.encoder(x.reshape(-1, 4)).reshape(*x.shape[:2], 6)
        return values[..., :3], values[..., 3:]

    def quantize(self, x, freeze_codebook=False):
        assert freeze_codebook
        ids = (x.unsqueeze(-2) - self.atoms).square().sum(-1).argmin(-1)
        return self.atoms[ids], ids, x.new_zeros(())


class DistillationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.config = yaml.safe_load(CONFIG.read_text())
        self.teacher = TinyTeacher().eval()
        self.objective = RepresentationDistillationLoss(
            self.config["loss_config"], {}, self.teacher,
            {"options": {"content_weight": 2, "style_weight": 3},
             "calibration": {"content_variance": 4, "style_variance": 5}})

    def inputs(self):
        x = torch.randn(2, 3, 4)
        return [torch.randn_like(x, requires_grad=True),
                torch.randn(2, 3, 3, requires_grad=True),
                torch.randn(2, 3, 3, requires_grad=True),
                torch.tensor(.2, requires_grad=True),
                torch.randn(2, 3, 3, requires_grad=True), x]

    def test_objective_and_gradient_isolation(self):
        inputs = self.inputs()
        before = {k: v.clone() for k, v in self.teacher.state_dict().items()}
        result = self.objective.compute_loss(*inputs)
        expected = (result["recon_loss"] + .1 * inputs[3]
                    + 2 * result["distill_content_mse"] / 4
                    + 3 * result["distill_style_mse"] / 5)
        torch.testing.assert_close(result["total_loss"], expected)
        result["total_loss"].backward()
        for index in (0, 1, 3, 4):
            self.assertGreater(float(inputs[index].grad.abs().sum()), 0)
        self.assertIsNone(inputs[2].grad)  # No extra gradient from structure diagnostics.
        for name in ("content_loss", "style_loss", "sample_loss", "fragment_loss", "v3_loss"):
            self.assertFalse(result[name].requires_grad)
        for p in self.teacher.parameters():
            self.assertFalse(p.requires_grad)
            self.assertIsNone(p.grad)
        for name, value in self.teacher.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))

    def test_structure_margin_does_not_change_optimizer_objective(self):
        inputs = self.inputs()
        first = self.objective.compute_loss(*inputs)
        self.objective.config["relativity"] = 1e6
        second = self.objective.compute_loss(*inputs)
        torch.testing.assert_close(first["total_loss"], second["total_loss"])
        self.assertNotEqual(float(first["v3_loss"]), float(second["v3_loss"]))

    def test_calibration_ignores_labels_and_is_deterministic(self):
        x = torch.randn(7, 3, 4)
        dataset = torch.utils.data.TensorDataset(x, torch.zeros(7, 3), torch.ones(7, 3))
        options = {"calibration_pages": 7, "calibration_seed": 42}
        result = calibrate_teacher(self.teacher, dataset, 2, options, "cpu")
        other = torch.utils.data.TensorDataset(x, torch.randn(7, 3), torch.randn(7, 3))
        repeated = calibrate_teacher(self.teacher, other, 3, options, "cpu")
        for name, target in zip(("content", "style"), self.teacher.encode(x)):
            exact = float(target.detach().double().reshape(-1, 3).var(0, unbiased=False).mean())
            self.assertAlmostEqual(result[name + "_variance"], exact, places=7)
            self.assertAlmostEqual(result[name + "_variance"], repeated[name + "_variance"], places=7)

    def test_rejects_active_v3_rank_or_schedules(self):
        self.assertTrue(validate_distillation_config(self.config))
        for key in ("content_loss", "style_loss", "sample_loss", "fragment_loss", "v3_loss"):
            config = copy.deepcopy(self.config)
            config["loss_config"]["weights"][key] = .01
            with self.assertRaises(ValueError): validate_distillation_config(config)
        for key, value in (("loss_schedules", {"anything": 1}), ("method", "v3_rank")):
            config = copy.deepcopy(self.config)
            config[key] = value
            with self.assertRaises(ValueError): validate_distillation_config(config)

    def test_pair_only_differs_in_lr_and_name(self):
        other = yaml.safe_load((CONFIG.parent / "cfg_rd_v3_hex_rank2_lr3e4_seed0.yaml").read_text())
        self.assertEqual(other["optimizer_config"].pop("lr"), .0003)
        self.assertEqual(self.config["optimizer_config"].pop("lr"), .0001)
        other.pop("name")
        self.config.pop("name")
        self.assertEqual(other, self.config)

    def test_bad_config_and_variance_rejected(self):
        for key, value in (("content_weight", -1), ("style_weight", float("nan")),
                           ("calibration_pages", 0), ("variance_floor", 0)):
            config = copy.deepcopy(self.config)
            config["representation_distillation"][key] = value
            with self.assertRaises(ValueError): validate_distillation_config(config)
        with self.assertRaises(ValueError):
            RepresentationDistillationLoss(self.config["loss_config"], {}, self.teacher,
                {"options": {}, "calibration": {"content_variance": 0, "style_variance": 1}})


if __name__ == "__main__":
    unittest.main()
