"""Persist completed V2 CTRL/W100 full-test acceptance; GPU only, idempotent."""
import datetime
import json
from pathlib import Path
import socket
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import yaml
from run_codebook_health import evaluate
from utils.codebook_metrics import file_sha256


def main():
    assert socket.gethostname().startswith('ws-') and torch.cuda.is_available()
    for arm in ('CTRL','W100'):
        root=Path('logs')/f'20260911-1522__VVI-RQ2-V2-C128-K26-{arm}-S0'
        config=yaml.safe_load((root/'config.yaml').read_text())
        directory=root/'acceptance_20260912'
        directory.mkdir(exist_ok=True)
        for role in ('best','current'):
            metadata=json.loads((root/('best_macro_atom_purity.json' if role=='best' else 'current_checkpoint.json')).read_text())
            cp=root/metadata['checkpoint']
            path=directory/f'health__{cp.stem}__test-full.json'
            identity={'checkpoint_sha256':file_sha256(cp),'config_sha256':file_sha256(root/'config.yaml'),
                      'dataset_manifest_sha256':config['dataset_manifest_sha256']}
            if path.exists():
                old=json.loads(path.read_text())
                assert all(old[k]==v for k,v in identity.items()),'Existing acceptance identity mismatch'
            else:
                result=evaluate(config,cp)
                payload={**result,**identity,'checkpoint':str(cp.resolve()),'selection':role,
                         'epoch':metadata['epoch'],'evaluated_at':datetime.datetime.now().astimezone().isoformat()}
                path.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
            grid=directory/f'reconstruction_grid__{cp.stem}__test-full-map__8styles__viz-v2.json'
            if not grid.exists():
                subprocess.run([sys.executable,'run_reconstruction_grid.py','--run',str(root),
                                '--active_checkpoint',role,'--output_dir',str(directory)],check=True)
            result=json.loads(path.read_text())
            print('ACCEPTED_ARTIFACTS',arm,role,'macro',result['codebook']['macro_atom_purity'],
                  'one_to_one',result['codebook']['one_to_one_accuracy'],'recon',result['losses']['recon_loss'],flush=True)


if __name__=='__main__':main()
