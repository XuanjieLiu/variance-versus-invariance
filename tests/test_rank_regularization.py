import copy
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from model.autoencoder import CSAE
from model.rank_regularization import RANK_LOG_COLUMNS, RankRegularization
from model.v3_loss import V3Loss
from utils.loss_logging import HISTORY_COLUMNS, LossLogger


def rank_config(**overrides):
    return {"enabled": True, "target_rank": 4, "weight": .01, "eps": 1e-12, **overrides}


def loss_config():
    return {"relativity": 15, "weights": {"recon_loss": 1, "commit_loss": .1,
        "content_loss": 1, "style_loss": 1, "sample_loss": 1, "fragment_loss": 1}}


class TinyEncoder(nn.Module):
    def __init__(self, channels, width, height, content, style):
        super().__init__()
        self.content = nn.Linear(6, content)
        self.style = nn.Linear(6, style)

    def forward(self, x):
        return self.content(x), self.style(x)


class TinyDecoder(nn.Module):
    def __init__(self, channels, width, height, content, style):
        super().__init__()
        self.output = nn.Linear(content + style, 6)

    def forward(self, c, s):
        return self.output(torch.cat((c, s), dim=-1))


def tiny_model(dim=4):
    config = {"n_channels": 1, "n_feature": 1, "fragment_len": 1,
              "d_emb_c": 8, "d_emb_s": 8, "n_atoms": 16, "n_fragments": 16,
              "vq_codebook_dim": dim, "vq_ema_decay": .5, "threshold_ema_dead_code": 0}
    return CSAE(config, TinyEncoder, TinyDecoder)


class RankFormulaTests(unittest.TestCase):
    def test_line_and_isotropic_rank_two_four(self):
        for rank in (1, 2, 4):
            axes = torch.eye(rank, dtype=torch.float64)
            points = torch.cat((axes, -axes)).unsqueeze(0)
            # Plenty of fragments/native dimensions even for the line case.
            points = nn.functional.pad(points.repeat(1, 4, 1), (0, 8 - rank))
            result = RankRegularization(rank_config())(points)
            self.assertAlmostEqual(result["rank_effective_rank"].item(), rank, places=8)
            self.assertAlmostEqual(result["rank_loss"].item(), max(0, 1 - rank / 4), places=8)

    def test_gram_equals_explicit_covariance_and_pagewise_hinge(self):
        torch.manual_seed(7)
        for pages, fragments, dim in ((3, 7, 512), (3, 16, 4)):
            q = torch.randn(pages, fragments, dim, dtype=torch.float64)
            centered = q - q.mean(1, keepdim=True)
            cov = centered.transpose(-1, -2) @ centered / (fragments - 1)
            expected = cov.diagonal(dim1=-2, dim2=-1).sum(-1).square() / (cov.square().sum((-2, -1)) + 1e-12)
            result = RankRegularization(rank_config())(q)
            torch.testing.assert_close(result["rank_effective_rank"], expected.mean())
            torch.testing.assert_close(result["rank_loss"], (1 - expected / 4).relu().mean())
        line = torch.arange(8, dtype=torch.float64) - 3.5
        q = torch.zeros(2, 8, 4, dtype=torch.float64)
        q[0, :, 0] = line
        q[1, :, 1] = line
        # Pooled pages would look rank2. Each page is rank1 and must be penalized.
        result = RankRegularization(rank_config())(q)
        self.assertAlmostEqual(result["rank_effective_rank"].item(), 1)
        self.assertAlmostEqual(result["rank_loss"].item(), .75)

    def test_rotation_translation_permutation_and_scale(self):
        torch.manual_seed(9)
        q = torch.randn(2, 12, 8, dtype=torch.float64)
        rotation = torch.linalg.qr(torch.randn(8, 8, dtype=torch.float64)).Q
        regularizer = RankRegularization(rank_config())
        expected = regularizer(q)["rank_effective_rank"]
        for transformed in (q @ rotation, q + 10, q[:, torch.randperm(12)], q * 3):
            torch.testing.assert_close(regularizer(transformed)["rank_effective_rank"], expected)

    def test_zero_variance_is_penalized_without_nan(self):
        q = torch.zeros(2, 16, 8, dtype=torch.float64, requires_grad=True)
        result = RankRegularization(rank_config())(q)
        self.assertEqual(result["rank_effective_rank"].item(), 0)
        self.assertEqual(result["rank_loss"].item(), 1)
        result["rank_loss"].backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        # Documented limitation: the smooth spectral objective cannot create
        # missing directions at an exactly constant initialization by itself.
        self.assertEqual(q.grad.abs().sum().item(), 0)

    def test_gradient_check_and_weak_direction_gradient(self):
        torch.manual_seed(1)
        q = torch.randn(2, 8, 4, dtype=torch.float64)
        q[..., 1:] *= .1
        q.requires_grad_()
        fn = lambda x: RankRegularization(rank_config())(x)["rank_loss"]
        self.assertTrue(torch.autograd.gradcheck(fn, (q,)))
        fn(q).backward()
        self.assertGreater(q.grad[..., 1:].abs().sum().item(), 0)

    def test_invalid_configs_shapes_and_theoretical_rank(self):
        for value in ({"target_rank": .5}, {"weight": -1}, {"eps": 0},
                      {"target_rank": float("inf")}, {"weight": float("nan")},
                      {"enabled": "false"}, {"typo": 1}):
            with self.assertRaises(ValueError): RankRegularization(rank_config(**value))
        regularizer = RankRegularization(rank_config())
        for dimensions in ((16, 2, 16), (3, 512, 16), (16, 512, 3)):
            with self.assertRaisesRegex(ValueError, "rank bound"):
                regularizer.validate_dimensions(*dimensions)
        for q in (None, torch.zeros(3, 4), torch.zeros(0, 8, 4), torch.zeros(2, 1, 4)):
            with self.assertRaises(ValueError): regularizer(q)
        with self.assertRaisesRegex(ValueError, "rank bound"):
            V3Loss({**loss_config(), "rank_regularization": rank_config()},
                   {"n_fragments": 16, "d_emb_c": 512, "vq_codebook_dim": 2, "n_atoms": 16})

    def test_disabled_exact_compatibility_and_addition_once(self):
        torch.manual_seed(3)
        args = (torch.randn(2, 8, 3), torch.randn(2, 8, 8), torch.randn(2, 8, 8),
                torch.tensor(.4), torch.randn(2, 8, 8), torch.randn(2, 8, 3))
        for adapted in (False, True):
            config = loss_config()
            if adapted: config["supersample_content"] = False
            historical = V3Loss(config)
            base = historical._base_compute_loss(*args)
            disabled = V3Loss({**config, "rank_regularization": rank_config(enabled=False)}).compute_loss(*args)
            self.assertEqual(base.keys(), disabled.keys())
            for name in base: self.assertTrue(torch.equal(base[name], disabled[name]), name)
            enabled = V3Loss({**config, "rank_regularization": rank_config(weight=.3)})
            with self.assertRaisesRegex(ValueError, "native_vq"): enabled.compute_loss(*args)
            native = torch.randn(2, 8, 4, requires_grad=True)
            result = enabled.compute_loss(*args, native_vq=native)
            for name in base.keys() - {"total_loss"}: torch.testing.assert_close(base[name], result[name])
            torch.testing.assert_close(result["total_loss"], base["total_loss"] + .3 * result["rank_loss"])
            result["total_loss"].backward()
            self.assertGreater(native.grad.abs().sum().item(), 0)
        with self.assertRaisesRegex(ValueError, "rank weight"):
            V3Loss({"weights": {"rank_loss": 1}, "relativity": 15})


@unittest.skipUnless(torch.cuda.is_available(), "GPU allocation required")
class RankVQTests(unittest.TestCase):
    def test_native_capture_preserves_forward_ema_rng_and_state_dict(self):
        for dim in (4, 8):
            torch.manual_seed(0)
            plain = tiny_model(dim).cuda().train()
            observed = copy.deepcopy(plain)
            x = torch.randn(2, 16, 6, device="cuda")
            before_cpu = torch.get_rng_state()
            before_gpu = torch.cuda.get_rng_state()
            old = plain(x)
            after_cpu = torch.get_rng_state()
            after_gpu = torch.cuda.get_rng_state()
            torch.set_rng_state(before_cpu); torch.cuda.set_rng_state(before_gpu)
            new = observed(x, return_native=True)
            self.assertEqual(len(old), 6); self.assertEqual(len(new), 7)
            # GPU k-means/scatter reductions can differ by a few ULPs between
            # otherwise identical forwards; assignments and RNG must agree.
            for a, b in zip(old, new): torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
            self.assertTrue(torch.equal(after_cpu, torch.get_rng_state()))
            self.assertTrue(torch.equal(after_gpu, torch.cuda.get_rng_state()))
            for key, value in plain.state_dict().items():
                torch.testing.assert_close(value, observed.state_dict()[key], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(observed.lift_vq_codes(new[6]), new[2])
            # K-means can already equal this first batch's assignment means;
            # use another batch to exercise nontrivial EMA motion.
            new = observed(torch.randn_like(x) + .7, return_native=True)
            torch.testing.assert_close(observed.lift_vq_codes(new[6]), new[2])
            # EMA really moved: a post-forward lookup is NOT this forward's code.
            self.assertGreater((new[6] - observed.get_native_vq_codes(new[3])).abs().max().item(), 1e-5)
            restored = tiny_model(dim).cuda()
            restored.load_state_dict(observed.state_dict(), strict=True)
            observed.zero_grad(set_to_none=True)
            RankRegularization(rank_config())(new[6])["weighted_rank_loss"].backward()
            self.assertGreater(observed.encoder.content.weight.grad.abs().sum().item(), 0)
            if dim == 4:
                self.assertGreater(observed.vq.project_in.weight.grad.abs().sum().item(), 0)
                self.assertIsNone(observed.vq.project_out.weight.grad)
            self.assertIsNone(observed.decoder.output.weight.grad)
            self.assertFalse(observed.vq.codebook.requires_grad)
            observed.eval()
            state = copy.deepcopy(observed.state_dict())
            with torch.inference_mode():
                out = observed(x, freeze_codebook=True, return_native=True)
                torch.testing.assert_close(out[6], observed.get_native_vq_codes(out[3]))
            for key, value in state.items(): torch.testing.assert_close(value, observed.state_dict()[key], rtol=0, atol=0)

    def test_temporary_hook_removed_on_failure(self):
        model = tiny_model().cuda()
        with patch.object(model.vq, "forward", side_effect=RuntimeError("injected")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                model(torch.randn(2, 16, 6, device="cuda"), return_native=True)
        self.assertEqual(len(model.vq.project_out._forward_pre_hooks), 0)

    def test_autocast_uses_float32_for_spectral_statistics(self):
        q = torch.randn(2, 16, 512, device="cuda") * 20
        regularizer = RankRegularization(rank_config())
        reference = regularizer(q)
        for dtype in (torch.float16, torch.bfloat16):
            with torch.autocast("cuda", dtype=dtype): result = regularizer(q)
            self.assertEqual(result["rank_loss"].dtype, torch.float32)
            torch.testing.assert_close(result["rank_effective_rank"], reference["rank_effective_rank"], rtol=0, atol=0)


class RankLoggingTests(unittest.TestCase):
    def test_optional_columns_resume_migration_and_plot(self):
        with tempfile.TemporaryDirectory() as directory:
            old = LossLogger(directory)
            old.log_epoch("train", 0, 1, {"total_loss": 1}, .01)
            path = Path(directory) / "loss_epoch_history.csv"
            with path.open() as f: self.assertEqual(next(csv.reader(f)), list(HISTORY_COLUMNS))
            logger = LossLogger(directory, extra_columns=RANK_LOG_COLUMNS)
            values = {"total_loss": .8, "rank_loss": .75, "weighted_rank_loss": .0075,
                      "rank_effective_rank": 1., "rank_target": 4., "rank_trace": 10., "rank_active_fraction": 1.}
            logger.log_epoch("train", 1, 2, values, .01)
            logger.log_epoch("val", 1, 2, values, .01)
            logger.log_step("train", 2, 3, 0, values, .01)
            logger.plot()
            with path.open() as f: rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["rank_loss"], "")
            self.assertEqual(rows[0]["total_loss"], "1.0")
            self.assertAlmostEqual(float(rows[1]["weighted_rank_loss"]), .0075)
            self.assertTrue(all(None not in row for row in rows))
            self.assertTrue((Path(directory) / "loss_curves.png").is_file())
            # Disabling optional columns later must not corrupt an expanded CSV.
            LossLogger(directory).log_epoch("train", 2, 3, {"total_loss": .7}, .01)
            with path.open() as f: self.assertEqual(list(csv.DictReader(f))[-1]["rank_loss"], "")


if __name__ == "__main__":
    unittest.main()
