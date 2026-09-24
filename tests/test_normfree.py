import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
import yaml

from model.factory import get_model
from model.normalization import configure_backbone_normalization
from utils.normfree_diagnostics import (NORM_TYPES, non_normalization_parameters,
    OptimizationScaleMonitor, PerStyleCodebookMonitor)
from utils.training_diagnostics import evaluate_bn_modes
from utils.training_utils import setup_seed
from test_training_diagnostics import assert_equal, rng_state

ROOT = Path(__file__).resolve().parents[1]


def config(arm):
    return yaml.safe_load((ROOT/'configs/uppercase/rq2/v2/normalization'/
        f'cfg_vvi_rq2_v2_c512_k26_w50_lr1e4_{arm}_seed0.yaml').read_text())


class NormFreeTests(unittest.TestCase):
    def test_literal_identity_and_validation(self):
        m = nn.Sequential(nn.Conv2d(3,16,1), nn.BatchNorm2d(16), nn.Sequential(nn.BatchNorm2d(6)))
        m[1].eval()
        weight = m[0].weight
        before = rng_state()
        rows = configure_backbone_normalization(m, 'none', scope='encoder')
        self.assertEqual(len(rows), 2)
        self.assertIs(m[0].weight, weight)
        self.assertIsInstance(m[1], nn.Identity)
        self.assertFalse(m[1].training)
        self.assertFalse(any(isinstance(x, NORM_TYPES) for x in m.modules()))
        assert_equal(self,before,rng_state())
        with self.assertRaises(ValueError): configure_backbone_normalization(m,'layer')
        with self.assertRaises(ValueError): configure_backbone_normalization(m,'group',scope='encoder')

    def test_shared_initialization_and_strict_load(self):
        models, states = {}, {}
        for arm in ('bn','dnonorm','allnonorm'):
            torch.manual_seed(0)
            models[arm] = get_model('uppercase_letters_dataloader', config(arm)['model_config'])
            states[arm] = torch.get_rng_state()
        for arm in ('dnonorm','allnonorm'):
            assert_equal(self, states['bn'],states[arm])
            assert_equal(self, dict(non_normalization_parameters(models['bn'])),
                         dict(non_normalization_parameters(models[arm])))
            assert_equal(self, models['bn'].vq.state_dict(), models[arm].vq.state_dict())
            self.assertFalse(any(isinstance(m,NORM_TYPES) for m in models[arm].decoder.modules()))
            self.assertEqual(len(models[arm].decoder_normalization_layers),28)
            self.assertTrue(any('shortcut' in x['name'] for x in models[arm].decoder_normalization_layers))
            models[arm].load_state_dict(copy.deepcopy(models[arm].state_dict()),strict=True)
            with self.assertRaises(RuntimeError): models[arm].load_state_dict(models['bn'].state_dict(),strict=True)
        self.assertEqual(sum(isinstance(m,NORM_TYPES) for m in models['dnonorm'].encoder.modules()),32)
        self.assertFalse(any(isinstance(m,NORM_TYPES) for m in models['allnonorm'].modules()))
        self.assertEqual(len(models['allnonorm'].encoder_normalization_layers),32)

    def test_all_bn_interventions_noop_state_neutral(self):
        setup_seed(0)
        m = get_model('uppercase_letters_dataloader', config('allnonorm')['model_config']).cuda().eval()
        x = torch.randn(2,3,3,48,32,device='cuda')
        labels = torch.tensor([[0,1,2]]*2,device='cuda')
        styles = torch.tensor([[0]*3,[1]*3],device='cuda')
        with torch.inference_mode(): m(x,freeze_codebook=True)
        state, rng = copy.deepcopy(m.state_dict()), rng_state()
        rows = evaluate_bn_modes(m,(x,labels,styles),np.eye(26),['a','b'],True)
        for row in rows:
            self.assertEqual(row['switched_bn_layer_count'],0)
            self.assertEqual(row['recon_loss'],rows[0]['recon_loss'])
            self.assertEqual(row['code_changed_fraction'],0)
        assert_equal(self,state,m.state_dict()); assert_equal(self,rng,rng_state())

    def test_optimization_monitor_does_not_change_training(self):
        setup_seed(0)
        a = get_model('uppercase_letters_dataloader',config('allnonorm')['model_config']).cuda()
        x = torch.randn(2,3,3,48,32,device='cuda')
        a.eval()
        with torch.inference_mode(): a(x,freeze_codebook=True)
        b = copy.deepcopy(a)
        a.train(); b.train()
        oa, ob = torch.optim.AdamW(a.parameters(),lr=1e-4), torch.optim.AdamW(b.parameters(),lr=1e-4)
        rng = rng_state()
        out_a = a(x,freeze_codebook=True)
        (out_a[0]-x).square().mean().backward(); oa.step()
        after_a = rng_state()
        with tempfile.TemporaryDirectory() as temp:
            mon = OptimizationScaleMonitor(temp,b,100); mon.begin(0); mon.before_forward(1,True)
            out_b = b(x,freeze_codebook=True)
            (out_b[0]-x).square().mean().backward(); ob.step()
            row = mon.collect(1,out_b[1],out_b[2],out_b[5])
            self.assertEqual(row['all_scales_finite'],1)
            self.assertGreater(row['encoder_grad_l2'],0)
            self.assertGreater(row['decoder_grad_l2'],0)
            summary = mon.finish(1)
            self.assertEqual(summary['sampled_steps'],1)
            self.assertTrue((Path(temp)/'optimization_scales.png').is_file())
            mon.close()
        assert_equal(self,a.state_dict(),b.state_dict())
        assert_equal(self,oa.state_dict(),ob.state_dict())
        assert_equal(self,after_a,rng_state())
        assert_equal(self,rng,after_a)

    def test_per_style_shared_mapping_and_balance(self):
        # All codes are globally correct except style b whose two labels share q0.
        q=torch.tensor([[0,1],[0,1],[0,0],[0,0]])
        y=torch.tensor([[0,1]]*4); s=torch.tensor([[0,0]]*2+[[1,1]]*2)
        counts=np.array([[4,2],[0,2]])
        with tempfile.TemporaryDirectory() as temp:
            mon=PerStyleCodebookMonitor(temp,2,['A','B'],['a','b'])
            before=rng_state(); mon.begin(0,5,'cpu'); mon.collect(q,y,s)
            rows=mon.finish(counts)
            self.assertEqual(rows[0]['global_mapping_accuracy'],1.)
            self.assertEqual(rows[1]['global_mapping_accuracy'],.5)
            self.assertEqual(rows[1]['largest_code_fraction'],1.)
            self.assertEqual(rows[1]['usage_perplexity'],1.)
            assert_equal(self,before,rng_state())
            mon.begin(1,6,'cpu'); mon.collect(q[:1],y[:1],s[:1])
            with self.assertRaises(ValueError): mon.finish(np.eye(2))
            mon.begin(1,6,'cpu')
            with self.assertRaises(ValueError): mon.collect(q+5,y,s)

    def test_pair_changes_only_encoder_norm(self):
        a,b=config('dnonorm'),config('allnonorm')
        for cfg in (a,b):
            self.assertEqual(cfg['epochs'],200)
            self.assertEqual(cfg['optimizer_config']['lr'],1e-4)
            self.assertEqual(cfg['model_config']['decoder_style_warmup_epochs'],50)
            self.assertEqual(cfg['model_config']['decoder_normalization'],'none')
            self.assertEqual(cfg['snapshot_every_n_epochs'],25)
            self.assertIsNone(cfg['load_checkpoint'])
        a.pop('name'); b.pop('name')
        a['model_config']['encoder_normalization']='none'
        self.assertEqual(a,b)
        old=config('bn'); old.pop('name')
        for key in ('optimization_diagnostics','per_style_codebook_monitor'): a.pop(key)
        a['model_config'].pop('encoder_normalization'); a['model_config']['decoder_normalization']='batch'
        self.assertEqual(a,old)


if __name__ == '__main__': unittest.main()
