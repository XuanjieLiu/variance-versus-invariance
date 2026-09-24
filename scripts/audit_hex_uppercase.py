"""Read-only data/config/source comparison for HEX-S1 versus uppercase S7."""
import difflib
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch
import yaml
from importlib import import_module
from dataset.lowercase_letters.letter_office import LetterOffice
from utils.foreground_color import fragment_color_metrics
from utils.subset_sampling import select_subset_indices


def main():
    assert socket.gethostname().startswith('ws-') and torch.cuda.is_available()
    os.chdir(ROOT); torch.set_num_threads(4)
    runs = {'uppercase': ROOT/'logs/20260913-1106__VVI-RQ2-V2-C512-K26-W0-S0',
            'hex': ROOT/'logs/20260915-1121__VVI-HEX-K16-REPO-S0'}
    output = ROOT/'logs/diagnostics/20260916_hex_uppercase_audit'; output.mkdir(exist_ok=True)
    report = {'node': socket.gethostname(), 'datasets': {}, 'config_differences': {}}
    configs = {key: yaml.safe_load((run/'config.yaml').read_text()) for key,run in runs.items()}
    def compare(a,b,path=''):
        for key in sorted(set(a)|set(b)):
            full = path+'.'+key if path else key
            if isinstance(a.get(key),dict) and isinstance(b.get(key),dict): compare(a[key],b[key],full)
            elif a.get(key)!=b.get(key): report['config_differences'][full]={'uppercase':a.get(key),'hex':b.get(key)}
    compare(configs['uppercase'],configs['hex'])
    for key,c in configs.items():
        manifest_path=Path(c['data_dir'])/'manifest.json'
        manifest=json.loads(manifest_path.read_text())
        module=import_module('dataloader.'+c['dataloader'])
        dataset=module.get_dataloader(Path(c['data_dir'])/'val',batch_size=32,shuffle=False,
            n_fragments=c['model_config']['n_fragments'],fragment_len=32).dataset
        # Fixed manifest-order32 pages per style; does not use model predictions.
        selected=[]
        for style in module.S_LIST:
            selected.extend([i for i,p in enumerate(dataset.png_paths) if Path(p).stem.rsplit('_',1)[-1]==style][:32])
        x=torch.stack([dataset[i][0] for i in selected]).cuda()
        colors=fragment_color_metrics(x.flatten(0,1),x.flatten(0,1))
        foreground_fraction=colors['foreground_fraction'].cpu().numpy()
        reference=dataset[0][0]
        with Image.open(dataset.png_paths[0]) as im:
            pixels=torch.from_numpy(np.array(im)).permute(2,0,1).float()/127.5-1
            sliced=pixels.reshape(3,48,c['model_config']['n_fragments'],32).permute(2,0,1,3)
            torch.testing.assert_close(reference,sliced)
            mode=im.mode
        train_pages=sum(e['split']=='train' for e in manifest['entries'])
        steps=train_pages//c['batch_size']
        report['datasets'][key]={'dataset':manifest['dataset'], 'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'train_pages':train_pages,'val_pages':len(dataset),'steps_per_epoch':steps,
            'total_steps':steps*c['epochs'],'fragments_per_batch':32*c['model_config']['n_fragments'],
            'total_training_fragments':train_pages*c['epochs']*c['model_config']['n_fragments'],
            'color_probe_pages':256,'foreground_fraction_mean':float(foreground_fraction.mean()),
            'foreground_fraction_quantiles':np.quantile(foreground_fraction,[0,.1,.5,.9,1]).tolist(),
            'image_mode':mode,'rgb_loader_matches_manual_slicing':True,
            'font_sha256':manifest['font_sha256'], 'generator_sha256':manifest['generator_sha256']}
        del x
    office=LetterOffice(font=str(ROOT/'dataset/phonenums/fonts/ITCKRIST.TTF'),alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    report['uppercase_renderer']={'font_size':office.font_size,'fit_to_bbox':True,'margin':office.margin}
    report['hex_renderer']={'font_size':48,'fit_to_bbox':False,'layout':'NumberOffice'}
    snapshots={}
    for key,run in runs.items():
        with tarfile.open(run/'reproducibility/source.tar.gz') as tar:
            snapshots[key]={m.name:tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
    names=['model/autoencoder.py','model/factory.py','model/modules/phonenums.py','model/v3_loss.py',
           'trainer.py','utils/training_utils.py','dataloader/lowercase_letters_dataloader.py']
    report['source_comparison']={}
    for name in names:
        a,b=snapshots['uppercase'][name],snapshots['hex'][name]
        report['source_comparison'][name]={'equal':a==b,'uppercase_sha256':hashlib.sha256(a).hexdigest(),'hex_sha256':hashlib.sha256(b).hexdigest()}
        if a!=b:
            diff=''.join(difflib.unified_diff(a.decode().splitlines(True),b.decode().splitlines(True),fromfile='S7/'+name,tofile='HEX/'+name))
            (output/(name.replace('/','__')+'.diff')).write_text(diff)
    (output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
