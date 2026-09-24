"""Uppercase replication must keep the Hex objective, optimizer and BN protocol."""
import copy
from pathlib import Path
import unittest
import torch
import yaml
from model.factory import get_model
from utils.training_utils import setup_seed

ROOT=Path(__file__).resolve().parents[1]


def config(suffix=''):
    return yaml.safe_load((ROOT/f'configs/uppercase/rq2/v2/cfg_vvi_rq2_v2_c512_k26_hexrepo{suffix}_seed0.yaml').read_text())


class S10ProtocolTests(unittest.TestCase):
    def test_only_registered_differences(self):
        a,b=config(),config('_mean')
        for c in (a,b):
            self.assertEqual(c['epochs'],200)
            self.assertIsNone(c['load_checkpoint'])
            self.assertIsNone(c['active_checkpoint'])
            self.assertEqual(c['macro_best_health_gate']['min'],{'active_codes':24,'codebook_purity':.2,'dominant_label_coverage':18,'usage_perplexity':18})
        a.pop('name'); b.pop('name'); a['model_config']['decoder_style_mode']='page_mean'
        self.assertEqual(a,b)
        h=yaml.safe_load((ROOT/'configs/hex/v2/cfg_vvi_hex_k16_repo_seed0.yaml').read_text())
        for key in ('loss_config','optimizer_config','precision','random_seed','batch_size',
                    'epochs','snapshot_every_n_epochs','bn_diagnostic','disentanglement_probes'):
            self.assertEqual(config()[key],h[key],key)
        model=copy.deepcopy(h['model_config']); model.update(n_atoms=26,n_fragments=26)
        self.assertEqual(config()['model_config'],model)
        old=yaml.safe_load((ROOT/'configs/uppercase/rq2/v2/cfg_vvi_rq2_v2_c512_k26_control_seed0.yaml').read_text())
        for key in ('loss_config','optimizer_config','data_dir','dataset_manifest_sha256','epochs','batch_size'):
            self.assertEqual(config()[key],old[key],key)

    def test_explicit_bn_and_page_mean_do_not_change_initialization(self):
        states=[]
        for suffix in ('','_mean'):
            setup_seed(0)
            model=get_model('uppercase_letters_dataloader',config(suffix)['model_config']).cuda()
            states.append({k:v.cpu().clone() for k,v in model.state_dict().items()})
            self.assertEqual(sum(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()),60)
            for epoch in (0,49,50,199):
                model.set_decoder_epoch(epoch)
                self.assertEqual(model.active_decoder_style_mode,'page_mean' if suffix else 'fragment')
            del model
        self.assertEqual(set(states[0]),set(states[1]))
        for key in states[0]: torch.testing.assert_close(states[0][key],states[1][key],rtol=0,atol=0)
        original=config()['model_config']; original.pop('encoder_normalization'); original.pop('decoder_normalization')
        setup_seed(0); old=get_model('uppercase_letters_dataloader',original)
        for key,value in old.state_dict().items(): torch.testing.assert_close(states[0][key],value,rtol=0,atol=0)


if __name__=='__main__': unittest.main()
