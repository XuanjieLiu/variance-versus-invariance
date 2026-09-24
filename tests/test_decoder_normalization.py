import copy
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn
import yaml

from model.factory import get_model
from model.normalization import configure_decoder_normalization
from utils.training_diagnostics import evaluate_bn_modes
from utils.training_utils import setup_seed
from test_training_diagnostics import assert_equal, rng_state

ROOT = Path(__file__).resolve().parents[1]


def config(arm):
    return yaml.safe_load((ROOT/'configs/uppercase/rq2/v2/normalization'/
        f'cfg_vvi_rq2_v2_c512_k26_w50_lr1e4_{arm}_seed0.yaml').read_text())


class DecoderNormalizationTests(unittest.TestCase):
    def test_conversion_preserves_affine_modes_rng_and_validates_before_mutation(self):
        decoder = nn.Sequential(nn.BatchNorm2d(16), nn.Sequential(nn.BatchNorm2d(32)))
        decoder[0].eval(); decoder[0].weight.requires_grad_(False)
        decoder.double()
        params = dict(decoder.named_parameters())
        before = rng_state()
        layers = configure_decoder_normalization(decoder, 'group', 8)
        assert_equal(self, before, rng_state())
        self.assertEqual(len(layers), 2)
        self.assertFalse(decoder[0].training)
        for name, value in decoder.named_parameters(): self.assertIs(value, params[name])
        self.assertFalse(list(decoder.buffers()))
        self.assertEqual(decoder[0].num_groups, 8)
        bad = nn.Sequential(nn.BatchNorm2d(16), nn.BatchNorm2d(6))
        with self.assertRaisesRegex(ValueError, 'not divisible'):
            configure_decoder_normalization(bad, 'group', 8)
        self.assertIsInstance(bad[0], nn.BatchNorm2d)
        with self.assertRaises(ValueError): configure_decoder_normalization(bad, 'unknown')
        with self.assertRaises(ValueError): configure_decoder_normalization(bad, 'group', 0)
        with self.assertRaises(ValueError): configure_decoder_normalization(nn.Identity(), 'group')

    def test_real_models_match_parameters_rng_encoder_and_strict_load(self):
        cfg = config('bn')['model_config']
        torch.manual_seed(0)
        bn = get_model('uppercase_letters_dataloader', cfg)
        after_bn = torch.get_rng_state()
        torch.manual_seed(0)
        gn = get_model('uppercase_letters_dataloader', config('dgn8')['model_config'])
        self.assertTrue(torch.equal(after_bn, torch.get_rng_state()))
        assert_equal(self, dict(bn.named_parameters()), dict(gn.named_parameters()))
        assert_equal(self, bn.encoder.state_dict(), gn.encoder.state_dict())
        assert_equal(self, bn.vq.state_dict(), gn.vq.state_dict())
        bn_layers = sum(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in bn.decoder.modules())
        self.assertGreater(bn_layers, 0)
        self.assertEqual(sum(isinstance(m, nn.GroupNorm) for m in gn.decoder.modules()), bn_layers)
        self.assertFalse(any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in gn.decoder.modules()))
        self.assertTrue(any('shortcut' in x['name'] for x in gn.decoder_normalization_layers))
        gn.load_state_dict(copy.deepcopy(gn.state_dict()), strict=True)
        with self.assertRaises(RuntimeError): gn.load_state_dict(bn.state_dict(), strict=True)
        # An omitted normalization field must exactly preserve historical BN state.
        old_cfg = dict(cfg); old_cfg.pop('decoder_normalization')
        torch.manual_seed(0)
        old = get_model('uppercase_letters_dataloader', old_cfg)
        assert_equal(self, old.state_dict(), bn.state_dict())

    def test_gn_decoder_mode_invariance_gradients_and_bn_diagnostic_noop(self):
        # Match Trainer's seeded deterministic cuDNN path before bitwise checks.
        previous_deterministic = torch.backends.cudnn.deterministic
        self.addCleanup(setattr, torch.backends.cudnn, 'deterministic', previous_deterministic)
        setup_seed(0)
        model = get_model('uppercase_letters_dataloader', config('dgn8')['model_config']).cuda()
        model.set_decoder_epoch(50)
        c = torch.randn(2, 3, 512, device='cuda', requires_grad=True)
        s = torch.randn_like(c, requires_grad=True)
        model.decoder.eval(); a = model.decode(c, s)
        model.decoder.train(); b = model.decode(c, s)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        b.square().mean().backward()
        self.assertGreater(float(c.grad.abs().sum()), 0)
        self.assertGreater(float(s.grad.abs().sum()), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.decoder.parameters() if p.grad is not None))
        batch = torch.randn(2, 3, 3, 48, 32, device='cuda')
        labels = torch.tensor([[0, 1, 2]]*2, device='cuda')
        styles = torch.tensor([[0, 0, 0], [1, 1, 1]], device='cuda')
        # Initialize VQ once, outside the state-neutral intervention.
        model.eval()
        with torch.inference_mode(): model(batch, freeze_codebook=True)
        before = copy.deepcopy(model.state_dict()); before_rng = rng_state()
        rows = evaluate_bn_modes(model, (batch, labels, styles), np.eye(26), ['a', 'b'], True)
        assert_equal(self, before, model.state_dict()); assert_equal(self, before_rng, rng_state())
        self.assertEqual(rows[2]['decoder_bn_layer_count'], 0)
        self.assertEqual(rows[2]['switched_bn_layer_count'], 0)
        for key in ('recon_loss', 'fixed_mapping_accuracy', 'code_changed_fraction', 'foreground_rgb_mae'):
            self.assertEqual(rows[0][key], rows[2][key])

    def test_only_pair_difference_is_decoder_norm_and_historical_change_is_lr(self):
        bn, gn = config('bn'), config('dgn8')
        self.assertEqual(bn['optimizer_config']['lr'], .0001)
        self.assertEqual(bn['epochs'], 200)
        self.assertEqual(bn['model_config']['decoder_style_warmup_epochs'], 50)
        for cfg in (bn, gn):
            self.assertEqual(cfg['model_config']['d_emb_c'], 512)
            self.assertEqual(cfg['model_config']['d_emb_s'], 512)
            self.assertNotIn('vq_codebook_dim', cfg['model_config'])
            self.assertEqual(cfg['snapshot_every_n_epochs'], 25)
            self.assertIsNone(cfg['load_checkpoint'])
        bn.pop('name'); gn.pop('name')
        for cfg in (bn, gn):
            cfg['model_config'].pop('decoder_normalization')
            cfg['model_config'].pop('decoder_groupnorm_groups', None)
        self.assertEqual(bn, gn)
        historical = yaml.safe_load((ROOT/'configs/uppercase/rq2/v2/cfg_vvi_rq2_v2_c512_k26_meanwarm50_seed0.yaml').read_text())
        historical.pop('name'); historical['optimizer_config']['lr'] = .0001
        historical['bn_diagnostic']['report_normalization_layers'] = True
        self.assertEqual(bn, historical)


if __name__ == '__main__': unittest.main()
