import copy
import csv
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
import yaml

from utils.training_diagnostics import (TrainingUsageMonitor, BatchNormDiagnostic,
    evaluate_bn_modes, usage_metrics, BN_MODES)


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def assert_equal(test, a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys())
        for key in a:
            assert_equal(test, a[key], b[key])
    elif isinstance(a, (tuple, list)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            assert_equal(test, x, y)
    else:
        test.assertEqual(a, b)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 1), torch.nn.BatchNorm2d(3))
        self.decoder = torch.nn.Sequential(torch.nn.BatchNorm2d(3), torch.nn.Conv2d(3, 3, 1))
        self.vq = torch.nn.Module()
        self.vq.register_buffer('ema', torch.ones(2))
        self.fail = False

    def forward(self, x, freeze_codebook=False):
        assert freeze_codebook and not self.vq.training
        z = self.encoder(x.flatten(0, 1))
        if self.fail:
            random.random(); np.random.random(); torch.rand(1, device=x.device)
            raise RuntimeError('injected forward failure')
        codes = (z[:, 0].mean((1, 2)) > 0).long().reshape(x.shape[:2])
        return self.decoder(z).reshape_as(x), z, z, codes, z.new_tensor(0), z


class TrainingDiagnosticsTests(unittest.TestCase):
    def test_usage_windows_and_epoch_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            monitor = TrainingUsageMonitor(temp, 3, 100)
            before = rng_state()
            monitor.begin(7, 'cpu')
            for step in range(1, 251):
                monitor.collect(torch.tensor([0, 0, 1]), step)
            result = monitor.finish(250)
            assert_equal(self, before, rng_state())
            with (Path(temp)/'training_codebook_usage.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(r['fragment_count']) for r in rows], [300, 300, 150, 750])
            self.assertEqual([r['scope'] for r in rows], ['window']*3+['epoch'])
            self.assertEqual(result['active_codes'], 2)
            self.assertAlmostEqual(result['max_code_usage_fraction'], 2/3)
            self.assertTrue((Path(temp)/'training_codebook_usage.png').exists())
            with self.assertRaises(ValueError): usage_metrics([0, 0])
            with self.assertRaises(ValueError): TrainingUsageMonitor(temp, 3, 0)

    def test_bn_state_rng_optimizer_and_fixed_mapping(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = TinyModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters())
        sum(p.square().sum() for p in model.parameters()).backward()
        optimizer.step(); optimizer.zero_grad()
        model.train(); model.decoder[0].eval()  # heterogeneous original modes
        state = copy.deepcopy(model.state_dict())
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        flags = [m.training for m in model.modules()]
        x = torch.randn(4, 2, 3, 8, 8, device=device)
        c = torch.tensor([[0, 1]]*4, device=device)
        s = torch.tensor([[0, 0], [1, 1]]*2, device=device)
        before = rng_state()
        rows = evaluate_bn_modes(model, (x, c, s), np.eye(2)*100, ['a', 'b'])
        self.assertEqual([r['mode'] for r in rows], list(BN_MODES))
        self.assertEqual(rows[2]['code_changed_fraction'], 0)
        self.assertEqual(rows[2]['fixed_mapping_accuracy'], rows[0]['fixed_mapping_accuracy'])
        assert_equal(self, model.state_dict(), state)
        assert_equal(self, optimizer.state_dict(), optimizer_state)
        assert_equal(self, before, rng_state())
        self.assertEqual(flags, [m.training for m in model.modules()])
        inverse = evaluate_bn_modes(model, (x, c, s), np.fliplr(np.eye(2))*100, ['a', 'b'])
        for row, other in zip(rows, inverse):
            self.assertAlmostEqual(row['fixed_mapping_accuracy'] + other['fixed_mapping_accuracy'], 1)
        model.fail = True
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            evaluate_bn_modes(model, (x, c, s), np.eye(2), ['a', 'b'])
        assert_equal(self, before, rng_state())
        assert_equal(self, model.state_dict(), state)
        self.assertEqual(flags, [m.training for m in model.modules()])

    def test_fixed_balanced_pages_without_rng_draws(self):
        class Dataset:
            s_list = ['a', 'b']
            png_paths = [f'/unused/{i:02d}_{style}.png' for style in ['a', 'b'] for i in range(5)]
            def __len__(self): return len(self.png_paths)
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            before = rng_state()
            first = BatchNormDiagnostic(a, Dataset(), {'pages_per_style': 4, 'seed': 0})
            second = BatchNormDiagnostic(b, Dataset(), {'pages_per_style': 4, 'seed': 0})
            self.assertEqual(first.ids, second.ids)
            self.assertEqual(first.fingerprint, second.fingerprint)
            assert_equal(self, before, rng_state())
            first.begin(0, 1)
            with self.assertRaisesRegex(RuntimeError, 'complete validation'):
                first.finish(TinyModel(), np.eye(2))

    def test_matched_c512_configs(self):
        root = Path(__file__).resolve().parents[1]/'configs/uppercase/rq2/v2'
        configs = [yaml.safe_load((root/f'cfg_vvi_rq2_v2_c512_k26_{suffix}_seed0.yaml').read_text())
                   for suffix in ['control', 'meanwarm50']]
        historical = yaml.safe_load((root/'cfg_vvi_rq2_v2_c128_k26_meanwarm50_seed0.yaml').read_text())
        for cfg in configs:
            for key in ['loss_config', 'optimizer_config', 'macro_best_health_gate',
                        'dataset_manifest_sha256', 'epochs', 'batch_size', 'precision']:
                self.assertEqual(cfg[key], historical[key])
            self.assertTrue(cfg['save_best_validation_loss'])
            self.assertEqual(cfg['model_config']['d_emb_c'], 512)
            self.assertEqual(cfg['model_config']['d_emb_s'], 512)
            self.assertNotIn('vq_codebook_dim', cfg['model_config'])
            self.assertEqual(cfg['snapshot_every_n_epochs'], 25)
        self.assertEqual(configs[0]['model_config']['decoder_style_mode'], 'fragment')
        self.assertEqual(configs[1]['model_config']['decoder_style_warmup_epochs'], 50)
        for key in ['training_usage_monitor', 'bn_diagnostic', 'disentanglement_probes']:
            a, b = copy.deepcopy(configs[0][key]), copy.deepcopy(configs[1][key])
            a.pop('switch_epochs', None); b.pop('switch_epochs', None)
            self.assertEqual(a, b)
