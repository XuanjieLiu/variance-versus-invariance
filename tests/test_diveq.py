import copy
from pathlib import Path
import unittest

import torch
from torch import nn
import yaml

from model.diveq import DiVeQ, build_diveq
from model.autoencoder import CSAE
from model.v3_loss import V3Loss
from test_rank_regularization import TinyEncoder, TinyDecoder, loss_config

ROOT = Path(__file__).resolve().parents[1]


class DiVeQTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def make(self, dim=4, **kwargs):
        return DiVeQ(dim, 16, **kwargs).to(self.device)

    def test_formula_values_and_gradients_match_directional_reference(self):
        for dim in (4, 512):
            q = self.make(dim).double()
            x = torch.randn(2, 16, dim, device=self.device, dtype=torch.float64, requires_grad=True)
            rng = torch.cuda.get_rng_state() if self.device == "cuda" else torch.get_rng_state()
            y, ids, commit = q(x)
            actual_grad = torch.autograd.grad(y.square().mean(), (x, q._codebook.embed))
            if self.device == "cuda": torch.cuda.set_rng_state(rng)
            else: torch.set_rng_state(rng)
            reference_x = x.detach().clone().requires_grad_()
            reference_atoms = q.codebook.detach().clone().requires_grad_()
            flat = reference_x.reshape(-1, dim)
            near = ((flat[:, None] - reference_atoms[None]) ** 2).sum(-1).argmin(-1)
            hard = reference_atoms[near]
            direction = hard - flat
            vector = direction + .001 * torch.randn_like(direction)
            reference = flat + direction.norm(dim=1, keepdim=True) * (
                vector / vector.norm(dim=1, keepdim=True).clamp_min(1e-12)).detach()
            expected_grad = torch.autograd.grad(reference.square().mean(), (reference_x, reference_atoms))
            torch.testing.assert_close(y, reference.reshape_as(y))
            torch.testing.assert_close(ids.flatten(), near)
            torch.testing.assert_close(actual_grad[0], expected_grad[0])
            torch.testing.assert_close(actual_grad[1][0], expected_grad[1])
            self.assertGreater(actual_grad[0].norm().item(), 0)
            self.assertGreater(actual_grad[1].norm().item(), 0)
            self.assertEqual(commit.item(), 0)

    def test_eval_hard_state_and_rng_neutral(self):
        q = self.make().eval()
        x = torch.randn(3, 16, 4, device=self.device)
        state = copy.deepcopy(q.state_dict())
        cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        with torch.inference_mode():
            y, ids, _ = q(x)
            y2, ids2, _ = q(x)
        self.assertTrue(torch.equal(y, q.codebook.detach()[ids]))
        self.assertTrue(torch.equal(y, y2) and torch.equal(ids, ids2))
        for k, v in state.items(): self.assertTrue(torch.equal(v, q.state_dict()[k]), k)
        self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
        for a, b in zip(gpu_rng, torch.cuda.get_rng_state_all()): self.assertTrue(torch.equal(a, b))

    def test_frozen_codebook_no_updates_but_encoder_gradients(self):
        q = self.make().train()
        x = torch.randn(2, 16, 4, device=self.device, requires_grad=True)
        before = copy.deepcopy(q.state_dict())
        y, _, _ = q(x, freeze_codebook=True)
        y.square().sum().backward()
        self.assertGreater(x.grad.norm().item(), 0)
        self.assertIsNone(q._codebook.embed.grad)
        for k in before: self.assertTrue(torch.equal(before[k], q.state_dict()[k]))
        q.eval()
        x.grad = None
        y, ids, _ = q(x, freeze_codebook=True)
        self.assertTrue(torch.equal(y, q.codebook[ids]))
        y.square().sum().backward()
        self.assertGreater(x.grad.norm().item(), 0)

    def test_no_grad_train_diagnostic_is_hard_and_neutral(self):
        q = self.make().train()
        x = torch.randn(2, 16, 4, device=self.device)
        before = copy.deepcopy(q.state_dict())
        rng = torch.cuda.get_rng_state_all()
        with torch.no_grad(): y, ids, _ = q(x, freeze_codebook=True)
        self.assertTrue(torch.equal(y, q.codebook[ids]))
        for k in before: self.assertTrue(torch.equal(before[k], q.state_dict()[k]))
        for a, b in zip(rng, torch.cuda.get_rng_state_all()): self.assertTrue(torch.equal(a, b))

    def test_batch_presence_and_deferred_replacement(self):
        q = self.make(replacement_iters=2, discard_threshold=.5, perturb_eps=1e-5)
        with torch.no_grad():
            q._codebook.embed.copy_(torch.arange(16, device=self.device)[None, :, None].expand(1, 16, 4))
        old = q.codebook.detach().clone()
        x = torch.full((2, 16, 4), .01, device=self.device, requires_grad=True)
        q(x)
        self.assertEqual(q.codebook_usage[0].item(), 1)  # not32 duplicate assignments
        q.after_optimizer_step()
        self.assertTrue(torch.equal(old, q.codebook))
        y, _, _ = q(x)
        self.assertEqual(q.codebook_usage[0].item(), 2)
        self.assertTrue(torch.equal(old, q.codebook))  # no in-forward mutation
        y.square().mean().backward()
        q.after_optimizer_step()
        self.assertEqual(q.total_replaced.item(), 15)
        self.assertEqual(q.codebook_usage.sum().item(), 0)
        self.assertFalse(q.replacement_pending.item())
        self.assertLess(q.codebook[1:].abs().max().item(), 1e-3)

    def test_strict_resume_optimizer_and_replacement_state(self):
        q = self.make(replacement_iters=3)
        opt = torch.optim.AdamW(q.parameters(), lr=.001)
        x = torch.randn(2, 16, 4, device=self.device)
        def step(module, optimizer):
            optimizer.zero_grad()
            module(x)[0].square().mean().backward()
            optimizer.step(); module.after_optimizer_step()
        step(q, opt)
        saved_model, saved_opt = copy.deepcopy(q.state_dict()), copy.deepcopy(opt.state_dict())
        restored = self.make(replacement_iters=3)
        restored.load_state_dict(saved_model, strict=True)
        restored_opt = torch.optim.AdamW(restored.parameters(), lr=.001)
        restored_opt.load_state_dict(saved_opt)
        torch.manual_seed(71); step(q, opt); step(q, opt)
        torch.manual_seed(71); step(restored, restored_opt); step(restored, restored_opt)
        for k, v in q.state_dict().items(): torch.testing.assert_close(v, restored.state_dict()[k], rtol=0, atol=0)
        self.assertEqual(q.iter_counter.item(), 3)

    def test_native_forward_and_v3_gradients(self):
        for dim in (4, 512):
            cfg = dict(n_channels=1, n_feature=1, fragment_len=1, d_emb_c=dim,
                       d_emb_s=dim, n_atoms=16, n_fragments=16, quantizer="diveq")
            model = CSAE(cfg, TinyEncoder, TinyDecoder).to(self.device)
            x = torch.randn(2, 16, 6, device=self.device)
            out, c, cq, ids, commit, s, native = model(x, return_native=True)
            self.assertEqual(native.shape, (2, 16, dim))
            self.assertTrue(torch.equal(native, cq))
            self.assertEqual(model.vq.iter_counter.item(), 1)
            self.assertEqual(len(model.vq.project_out._forward_pre_hooks), 0)
            lc = loss_config(); lc["weights"]["commit_loss"] = 0
            losses = V3Loss(lc, cfg).compute_loss(out, c, cq, commit, s, x)
            losses["total_loss"].backward()
            for parameters in (model.encoder.parameters(), model.decoder.parameters(), model.vq.parameters()):
                grads = [p.grad for p in parameters if p.grad is not None]
                self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
                self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)
            self.assertEqual(losses["commit_loss"].item(), 0)

    def test_equal_input_atom_is_finite(self):
        q = self.make()
        x = q.codebook[0].detach().clone().reshape(1, 1, 4).requires_grad_()
        y, _, _ = q(x)
        y.sum().backward()
        self.assertTrue(torch.isfinite(y).all() and torch.isfinite(x.grad).all())

    def test_invalid_configs(self):
        for kw in ({"noise_std": 0}, {"noise_std": float("nan")}, {"perturb_eps": -1},
                   {"replacement_iters": 0}, {"discard_threshold": 2}):
            with self.assertRaises(ValueError): self.make(**kw)
        with self.assertRaisesRegex(ValueError, "EMA"):
            build_diveq({"d_emb_c": 4, "n_atoms": 16, "vq_ema_decay": .98})
        with self.assertRaisesRegex(ValueError, "native"):
            build_diveq({"d_emb_c": 512, "n_atoms": 16, "vq_codebook_dim": 4})
        with self.assertRaisesRegex(ValueError, "commit_loss"):
            V3Loss(loss_config(), {"quantizer": "diveq"})

    def test_legacy_ema_default_checkpoint_compatibility(self):
        cfg = dict(n_channels=1, n_feature=1, fragment_len=1, d_emb_c=4, d_emb_s=4, n_atoms=16)
        torch.manual_seed(0); a = CSAE(cfg, TinyEncoder, TinyDecoder)
        torch.manual_seed(0); b = CSAE({**cfg, "quantizer": "ema"}, TinyEncoder, TinyDecoder)
        b.load_state_dict(a.state_dict(), strict=True)
        for key in a.state_dict(): torch.testing.assert_close(a.state_dict()[key], b.state_dict()[key])
        self.assertFalse(a.vq._codebook.embed.requires_grad)

    def test_formal_configs_match_fragment_hex_protocol(self):
        base = yaml.safe_load((ROOT / 'configs/hex/v2/cfg_vvi_hex_k16_repo_seed0.yaml').read_text())
        for dim in (4, 512):
            cfg = yaml.safe_load((ROOT / f'configs/hex/v2/cfg_vvi_hex_k16_diveq_c{dim}_s{dim}_seed0.yaml').read_text())
            self.assertEqual(cfg['model_config']['decoder_style_mode'], 'fragment')
            self.assertNotIn('rank_regularization', cfg['loss_config'])
            self.assertEqual(cfg['model_config']['d_emb_c'], dim)
            self.assertEqual(cfg['model_config']['d_emb_s'], dim)
            self.assertEqual(cfg['loss_config']['weights']['commit_loss'], 0)
            self.assertEqual(cfg['model_config'].pop('quantizer'), 'diveq')
            self.assertEqual(cfg['model_config'].pop('diveq_config'),
                dict(noise_std=.001, replacement_iters=100, discard_threshold=.01, perturb_eps=1e-9))
            cfg.pop('diveq_reference'); cfg['name'] = base['name']
            cfg['model_config'].update(d_emb_c=512, d_emb_s=512, threshold_ema_dead_code=16, vq_ema_decay=.98)
            cfg['loss_config']['weights']['commit_loss'] = .1
            self.assertEqual(cfg, base)


if __name__ == '__main__': unittest.main()
