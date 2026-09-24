"""GPU preflight for PAPER/REPO/W50; retain dirs only until visual inspection."""
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import yaml
from trainer import Trainer
from utils.codebook_metrics import file_sha256
from smoke_v2 import TwoBatches, rng_state, assert_rng_equal


def main():
    assert socket.gethostname().startswith('ws-') and torch.cuda.is_available()
    torch.set_num_threads(4)
    root=Path(__file__).resolve().parents[1]; os.chdir(root)
    data=root.parent/'data/PhoneNumsV2'
    manifest=json.loads((data/'manifest.json').read_text())
    assert manifest['page_count']==100000
    for i,e in enumerate(manifest['entries'],1):
        assert file_sha256(data/e['path'])==e['sha256'],e['path']
        if i%20000==0:print('PNG_HASH_VERIFIED',i,flush=True)
    cases=[('paper','configs/phonenums/v2/cfg_vvi_pn_k10_paper_seed0.yaml',10,1250),
           ('repo','configs/phonenums/v2/cfg_vvi_pn_k10_repo_seed0.yaml',10,1250),
           ('w50','configs/uppercase/rq2/v2/cfg_vvi_rq2_v2_c128_k26_meanwarm50_seed0.yaml',26,325)]
    report={}
    for arm,path,n_content,per_pair in cases:
        config=yaml.safe_load((root/path).read_text())
        assert config['dataset_manifest_sha256']==file_sha256(Path(config['data_dir'])/'manifest.json')
        config.update(name=f'smoke-pn-w50-{arm}',debug=True,debug_portion=1,snapshot_every_n_epochs=1)
        directory=root/'logs'/config['name']; assert not directory.exists()
        for h in logging.root.handlers[:]:
            logging.root.removeHandler(h); h.close()
        trainer=Trainer(config);trainer.prepare_data();trainer.build_model()
        initial=json.loads((directory/'initial_model_state.json').read_text())['sha256']
        before=rng_state()
        order=np.asarray(list(iter(trainer.train_loader.sampler)),dtype=np.int64)
        order_hash=hashlib.sha256(order.tobytes()).hexdigest()
        random.setstate(before[0]);np.random.set_state(before[1]);torch.set_rng_state(before[2]);torch.cuda.set_rng_state_all(before[3])
        if arm=='w50':
            trainer.model.set_decoder_epoch(49)
            assert trainer.model.active_decoder_style_mode=='page_mean'
            trainer.model.set_decoder_epoch(50)
            assert trainer.model.active_decoder_style_mode=='fragment'
            trainer.model.set_decoder_epoch(0)
            historical=json.loads((root/'logs/20260911-1522__VVI-RQ2-V2-C128-K26-W100-S0/initial_model_state.json').read_text())
            assert initial==historical['sha256']
        monitor=trainer.disentanglement_monitor
        for method in ('begin','collect','finish'):
            original=getattr(monitor,method)
            def checked(*args,_name=method,_original=original,**kwargs):
                before=rng_state()
                state={k:v.detach().clone() for k,v in trainer.model.state_dict().items()} if _name=='finish' else None
                result=_original(*args,**kwargs)
                assert_rng_equal(before,rng_state())
                if state is not None:
                    for k,v in state.items():torch.testing.assert_close(v,trainer.model.state_dict()[k],rtol=0,atol=0)
                return result
            setattr(monitor,method,checked)
        trainer.train_loader=TwoBatches(trainer.train_loader)
        trainer.train()
        for filename in ('loss_epoch_history.csv','codebook_epoch_history.csv','v3_ratio_epoch_history.csv',
                         'disentanglement_probe_history.csv','foreground_color_epoch_history.csv'):
            with (directory/filename).open() as handle:rows=list(csv.DictReader(handle))
            assert int(rows[-1]['epoch'])==0
        for filename in ('loss_curves.png','codebook_metrics.png','v3_ratios.png','disentanglement_probes.png',
                         'foreground_color_metrics.png','cp_current_epoch0.pt','cp_snapshot_epoch0.pt'):
            assert (directory/filename).is_file(),filename
        if arm!='w50':
            assert (directory/'cp_best_validation_loss_epoch0.pt').is_file()
            state=torch.load(directory/'cp_current_epoch0.pt',map_location='cpu',weights_only=False)
            assert state['best_validation_epoch']==0
            assert state['best_validation_loss']==trainer.best_validation_loss
            del state
        assert len(list(directory.glob('cp*.pt'))) <= (4 if arm!='w50' else 3)
        matrices=list((directory/'reconstruction_diagnostics').glob('*column-normalized*.json'))
        assert len(matrices)==1
        matrix=json.loads(matrices[0].read_text())
        assert matrix['annotation_version']==2
        np.testing.assert_array_equal(matrix['content_style_counts'],np.full((n_content,8),per_pair))
        np.testing.assert_allclose(np.asarray(matrix['probabilities']).sum(0),1)
        np.testing.assert_array_equal(np.asarray(matrix['display_hundredths']).sum(0),100)
        assert matrix['fragment_count']==n_content*8*per_pair
        for suffix in ('.svg','.png','.csv'):assert matrices[0].with_suffix(suffix).is_file()
        report[arm]={'initial_model_sha256':initial,'first_epoch_sampler_sha256':order_hash,
                     'full_validation_fragments':matrix['fragment_count'],'checkpoints':len(list(directory.glob('cp*.pt')))}
        print('SMOKE_OK',arm,json.dumps(report[arm]),flush=True)
        del trainer;torch.cuda.empty_cache()
    assert report['paper']['initial_model_sha256']==report['repo']['initial_model_sha256']
    assert report['paper']['first_epoch_sampler_sha256']==report['repo']['first_epoch_sampler_sha256']
    subprocess.run([sys.executable,'run_reconstruction_grid.py','--run','smoke-pn-w50-paper',
                    '--active_checkpoint','best_val','--figure17'],check=True)
    assert list((root/'logs/smoke-pn-w50-paper/vis').glob('code_style_recombination*.png'))
    output=root/'logs/diagnostics/20260912_pn_w50_preflight'
    output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('ALL_THREE_PREFLIGHTS_OK',flush=True)


if __name__=='__main__':main()
