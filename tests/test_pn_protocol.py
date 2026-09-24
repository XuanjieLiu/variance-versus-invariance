import json
from pathlib import Path
import tempfile
import unittest
import torch
import yaml
from trainer import Trainer
from model.v3_loss import V3Loss
from utils.evaluation_paths import resolve_evaluation_paths
from utils.confusion_diagnostics import format_probability
from run_codebook_health import select_best_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def tiny_trainer(directory):
    t = Trainer.__new__(Trainer)
    t.log_dir, t.config = str(directory), {}
    t.model = torch.nn.Linear(2,2)
    t.optimizer = torch.optim.Adam(t.model.parameters(),lr=.001)
    t.scheduler = torch.optim.lr_scheduler.StepLR(t.optimizer,1)
    t.scaler = torch.amp.GradScaler('cuda',enabled=False)
    t.best_macro_atom_purity = t.best_macro_val_loss = t.best_macro_epoch = None
    t.best_validation_loss = t.best_validation_epoch = None
    return t


class BestValidationTest(unittest.TestCase):
    def test_rotation_finite_ties_alias_and_restore(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); first=root/'first'; first.mkdir()
            t=tiny_trainer(first)
            for e,value in [(0,.3),(1,.2),(2,.2),(3,.4),(4,float('nan')),(5,float('inf'))]:
                if t._consider_best_validation_loss(e,value):
                    t._persist_best_validation_checkpoint(e)
            self.assertEqual(t.best_validation_epoch,1)
            self.assertEqual(len(list(first.glob('cp*.pt'))),1)
            t._save_current_checkpoint(5,.4,.5)
            (first/'config.yaml').write_text('{}')
            resolved=resolve_evaluation_paths(run=first,active_checkpoint='best_val')
            self.assertEqual(Path(resolved['active_checkpoint']).name,'cp_best_validation_loss_epoch1.pt')
            self.assertEqual(select_best_checkpoint(first)[1:],(1,.2))
            state=torch.load(first/'cp_current_epoch5.pt',weights_only=False)
            second=root/'second'; second.mkdir()
            r=tiny_trainer(second)
            r._restore_best_validation_state(state,first/'cp_current_epoch5.pt')
            self.assertEqual(r.best_validation_loss,.2)
            self.assertEqual(r.best_validation_epoch,1)
            self.assertEqual((second/'cp_best_validation_loss_epoch1.pt').read_bytes(),
                             (first/'cp_best_validation_loss_epoch1.pt').read_bytes())
            self.assertFalse(r._consider_best_validation_loss(6,.2))
            r._restore_best_validation_state({},first/'old.pt')
            self.assertIsNone(r.best_validation_loss)
            state['best_validation_epoch']=99
            with self.assertRaises(FileNotFoundError):
                r._restore_best_validation_state(state,first/'cp_current_epoch5.pt')

    def test_200_epoch_retention_bounds(self):
        for enable_val,limit in [(True,11),(False,10)]:
            with tempfile.TemporaryDirectory() as temp:
                t=tiny_trainer(temp)
                for e in range(200):
                    value=1/(e+1)
                    if enable_val and t._consider_best_validation_loss(e,value):
                        t._persist_best_validation_checkpoint(e)
                    t._save_best_macro_checkpoint(e,value,1-value)
                    t._save_current_checkpoint(e,value,1-value)
                    if (e+1)%25==0:
                        t._save_periodic_checkpoint(e)
                    self.assertLessEqual(len(list(Path(temp).glob('cp*.pt'))),limit)
                self.assertEqual(len(list(Path(temp).glob('cp*.pt'))),limit)


class ProtocolTest(unittest.TestCase):
    def test_paper_repo_loss_optimizer_and_vq(self):
        for arm,cls,weight,alpha,decay,threshold,wd in [
            ('paper',torch.optim.Adam,.1,.01,.95,3.2,0),
            ('repo',torch.optim.AdamW,1,.1,.98,16,.1)]:
            config=yaml.safe_load((ROOT/f'configs/phonenums/v2/cfg_vvi_pn_k10_{arm}_seed0.yaml').read_text())
            self.assertEqual(config['model_config']['vq_ema_decay'],decay)
            self.assertEqual(config['model_config']['threshold_ema_dead_code'],threshold)
            w=config['loss_config']['weights']
            self.assertEqual(set(w),{'recon_loss','content_loss','style_loss','sample_loss','fragment_loss','commit_loss'})
            self.assertEqual(w['commit_loss'],alpha)
            for key in ('content_loss','style_loss','sample_loss','fragment_loss'):
                self.assertEqual(w[key],weight)
            x=torch.randn(3,10,2); c=torch.randn(3,10,4); s=torch.randn(3,10,4)
            result=V3Loss(config['loss_config']).compute_loss(x,c,c,torch.tensor(.7),s,torch.zeros_like(x))
            expected=result['recon_loss']+alpha*.7+weight*result['v3_loss']
            torch.testing.assert_close(result['total_loss'],expected)
            with tempfile.TemporaryDirectory() as temp:
                config.update(log_dir=temp,name=arm,record_initial_model_hash=False)
                trainer=Trainer(config);trainer.build_model()
                self.assertIs(type(trainer.optimizer),cls)
                self.assertEqual(trainer.optimizer.param_groups[0]['weight_decay'],wd)
                self.assertEqual(trainer.model.vq._codebook.decay,decay)
                self.assertEqual(trainer.model.vq._codebook.threshold_ema_dead_code,threshold)
                del trainer

    def test_w50_matches_w100_and_annotation_format(self):
        folder=ROOT/'configs/uppercase/rq2/v2'
        a=yaml.safe_load((folder/'cfg_vvi_rq2_v2_c128_k26_meanwarm50_seed0.yaml').read_text())
        b=yaml.safe_load((folder/'cfg_vvi_rq2_v2_c128_k26_meanwarm100_seed0.yaml').read_text())
        for key in ('optimizer_config','loss_config','macro_best_health_gate','dataset_manifest_sha256','epochs'):
            self.assertEqual(a[key],b[key])
        self.assertEqual(a['model_config']['decoder_style_warmup_epochs'],50)
        self.assertFalse(a.get('save_best_validation_loss',False))
        self.assertEqual(format_probability(0),'')
        self.assertEqual(format_probability(88),'.88')
        self.assertEqual(format_probability(1),'.01')
        self.assertEqual(format_probability(100),'1.00')
