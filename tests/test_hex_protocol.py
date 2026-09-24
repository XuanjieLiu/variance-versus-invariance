import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import yaml
from model.factory import get_model
from model.v3_loss import V3Loss
from utils.codebook_metrics import grouped_mapping_metrics
from utils.normfree_diagnostics import PerStyleCodebookMonitor
from utils.training_utils import setup_seed
from test_training_diagnostics import assert_equal, rng_state

ROOT = Path(__file__).resolve().parents[1]


def config(mean=False):
    suffix = '_mean' if mean else ''
    return yaml.safe_load((ROOT / f'configs/hex/v2/cfg_vvi_hex_k16_repo{suffix}_seed0.yaml').read_text())


class HexProtocolTests(unittest.TestCase):
    def test_exact_matched_repo_bundle_and_modes(self):
        a, b = config(), config(True)
        for c in (a, b):
            self.assertEqual(c['epochs'], 200)
            self.assertEqual(c['batch_size'], 32)
            self.assertIsNone(c['load_checkpoint'])
            self.assertIsNone(c['active_checkpoint'])
            self.assertEqual(c['macro_best_health_gate']['min'], {
                'active_codes': 15, 'codebook_purity': .2,
                'dominant_label_coverage': 12, 'usage_perplexity': 11.2})
            self.assertEqual(c['snapshot_every_n_epochs'], 25)
        repo = yaml.safe_load((ROOT / 'configs/phonenums/v2/cfg_vvi_pn_k10_repo_seed0.yaml').read_text())
        self.assertEqual(a['loss_config'], repo['loss_config'])
        self.assertEqual(a['optimizer_config'], repo['optimizer_config'])
        mc = copy.deepcopy(repo['model_config']); mc.update(n_atoms=16, n_fragments=16,
            encoder_normalization='batch', decoder_normalization='batch')
        self.assertEqual(a['model_config'], mc)
        a.pop('name'); b.pop('name'); a['model_config']['decoder_style_mode'] = 'page_mean'
        self.assertEqual(a, b)

    def test_shape_bn_strict_load_raw_style_v3_and_gradient(self):
        models, states = [], []
        for mean in (False, True):
            setup_seed(0)
            model = get_model('hex_digits_dataloader', config(mean)['model_config']).cuda().eval()
            models.append(model); states.append(rng_state())
            self.assertEqual(sum(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.encoder.modules()), 32)
            self.assertEqual(sum(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.decoder.modules()), 28)
            for epoch in (0, 49, 50, 199):
                model.set_decoder_epoch(epoch)
                self.assertEqual(model.active_decoder_style_mode, 'page_mean' if mean else 'fragment')
            model.set_decoder_epoch(0)
        assert_equal(self, models[0].state_dict(), models[1].state_dict())
        assert_equal(self, states[0], states[1])
        x = torch.randn(2, 16, 3, 48, 32, device='cuda')
        # Initialize VQ only once, then copy its state, keeping both arms matched.
        with torch.inference_mode(): models[0](x, freeze_codebook=True)
        models[1].load_state_dict(models[0].state_dict(), strict=True)
        outputs = [m(x, freeze_codebook=True) for m in models]
        for out in outputs:
            self.assertEqual(tuple(out[0].shape), tuple(x.shape))
            self.assertEqual(tuple(out[3].shape), (2, 16))
        for i in (1, 2, 3, 5): torch.testing.assert_close(outputs[0][i], outputs[1][i])
        zs = outputs[1][5]
        torch.testing.assert_close(models[1].decoder_style_input(zs), zs.mean(1, keepdim=True).expand_as(zs))
        losses = [V3Loss(config()['loss_config']).compute_loss(o[0], o[1], o[2], o[4], o[5], x) for o in outputs]
        for name in ('v3_loss', 'content_loss', 'style_loss', 'sample_loss', 'fragment_loss'):
            torch.testing.assert_close(losses[0][name], losses[1][name])
        losses[1]['total_loss'].backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in models[1].parameters() if p.grad is not None))
        self.assertGreater(float(models[1].encoder.linear_s.weight.grad.abs().sum()), 0)

    def test_group_mapping_never_rematches(self):
        # The global optimum reserves q0 for label1. Rematching only label0 would falsely give 1.
        counts = np.array([[4, 6], [0, 1]])
        before = rng_state()
        result = grouped_mapping_metrics(counts, {'digits': [0], 'letters': [1]})
        self.assertEqual(result['groups']['digits']['accuracy'], 0)
        self.assertAlmostEqual(result['groups']['letters']['accuracy'], 6 / 7)
        assert_equal(self, before, rng_state())
        perfect = grouped_mapping_metrics(np.eye(16), {'digits': list(range(10)), 'letters': list(range(10, 16))})
        self.assertEqual(perfect['groups']['digits']['accuracy'], 1)
        self.assertEqual(perfect['groups']['letters']['accuracy'], 1)
        inactive = grouped_mapping_metrics(np.array([[3, 0, 0], [0, 0, 0]]), {'missing': [2]})
        self.assertIsNone(inactive['groups']['missing']['accuracy'])
        with self.assertRaises(ValueError): grouped_mapping_metrics(counts, {'a': [0], 'b': [0]})
        with self.assertRaises(ValueError): grouped_mapping_metrics(counts, {'a': [2]})

    def test_group_csv_json_plot_and_fixed_mapping(self):
        q = torch.tensor([[0, 0], [0, 1]])
        y = torch.tensor([[0, 1], [0, 1]])
        s = torch.tensor([[0, 0], [1, 1]])
        with tempfile.TemporaryDirectory() as root:
            mon = PerStyleCodebookMonitor(root, 2, ['0', 'a'], ['black', 'blue'], {'digits': [0], 'letters': [1]})
            before = rng_state(); mon.begin(0, 1, 'cpu'); mon.collect(q, y, s)
            mon.finish(np.array([[2, 1], [0, 1]]))
            assert_equal(self, before, rng_state())
            rows = list(csv.DictReader((Path(root) / 'content_group_epoch_history.csv').open()))
            self.assertEqual([float(r['accuracy']) for r in rows], [1., .5])
            self.assertTrue((Path(root) / 'content_group_metrics.png').is_file())
            data = json.loads((Path(root) / 'per_style_codebook/counts_epoch000__val.json').read_text())
            self.assertEqual(data['content_group_metrics']['per_content_accuracy'], [1., .5])


if __name__ == '__main__': unittest.main()
