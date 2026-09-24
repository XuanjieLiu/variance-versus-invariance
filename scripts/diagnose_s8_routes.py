"""Read-only full-test per-style counts and small fixed-grid routing probes."""
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import yaml
from dataloader.uppercase_letters_dataloader import S_LIST
from model.factory import get_model
from run_codebook_health import build_loader
from run_reconstruction_grid import compute_hungarian_mapping, select_one_sample_per_style
from utils.codebook_metrics import compute_assignment_metrics, file_sha256
from utils.training_utils import setup_seed


def main():
    assert socket.gethostname().startswith('ws-') and torch.cuda.is_available()
    torch.set_num_threads(4)
    for arm, role in [('BN', 'best_macro_atom_purity'), ('BN', 'current_checkpoint'),
                      ('DGN8', 'current_checkpoint')]:
        root = ROOT / 'logs' / f'20260913-2309__VVI-RQ2-V2-C512-K26-W50-LR1E4-{arm}-S0'
        cp = root / json.loads((root / (role + '.json')).read_text())['checkpoint']
        cfg = yaml.safe_load((root / 'config.yaml').read_text())
        assert file_sha256(Path(cfg['data_dir']) / 'manifest.json') == cfg['dataset_manifest_sha256']
        setup_seed(0)
        model = get_model(cfg['dataloader'], cfg['model_config']).cuda()
        state = torch.load(cp, map_location='cuda', weights_only=False)
        model.load_state_dict(state['model'], strict=True)
        model.set_decoder_epoch(state['epoch'])
        model.eval()
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        loader, labels, _ = build_loader(cfg)
        k, c, s = cfg['model_config']['n_atoms'], len(labels), len(S_LIST)
        counts = torch.zeros(s*k*c, dtype=torch.int64, device='cuda')
        with torch.inference_mode():
            for x, content, style in loader:
                out = model(x.cuda(), freeze_codebook=True)
                flat = ((style.cuda().long().flatten()*k + out[3].flatten())*c
                        + content.cuda().long().flatten())
                counts += torch.bincount(flat, minlength=s*k*c)
            counts = counts.reshape(s,k,c).cpu().numpy()
            assert np.all(counts.sum(axis=1) == 325)
            mapping, _, _ = compute_hungarian_mapping(counts.sum(axis=0))
            per_style = {}
            for i, name in enumerate(S_LIST):
                usage = counts[i].sum(axis=1)
                per_style[name] = {**compute_assignment_metrics(counts[i]),
                    'fixed_global_mapping_accuracy': float(sum(counts[i,q,y] for q,y in mapping.items())/counts[i].sum()),
                    'largest_code_id': int(usage.argmax()),
                    'largest_code_fraction': float(usage.max()/usage.sum())}
            items = [loader.dataset[i] for i,_ in select_one_sample_per_style(loader.dataset,S_LIST)]
            x = torch.stack([t[0] for t in items]).cuda()
            out = model(x, freeze_codebook=True)
            interventions = {'original':out[0],
                'roll_content_within_page':model.decode(out[2].roll(1,dims=1),out[5]),
                'roll_style_within_page':model.decode(out[2],out[5].roll(1,dims=1)),
                'replace_style_with_page_mean':model.decode(out[2],out[5].mean(dim=1,keepdim=True).expand_as(out[5]))}
            routing = {name:{'recon_mse':float((y-x).square().mean()),
                'output_change_mse':float((y-out[0]).square().mean())} for name,y in interventions.items()}
        assert all(torch.equal(v, model.state_dict()[key]) for key,v in before.items())
        result = {'checkpoint':str(cp),'checkpoint_sha256':file_sha256(cp),
            'config_sha256':file_sha256(root/'config.yaml'),
            'manifest_sha256':cfg['dataset_manifest_sha256'],
            'decoder_style_mode':model.active_decoder_style_mode,'epoch':state['epoch'],
            'per_style_scope':'full test;325 pages/style;325 fragments/content/style',
            'per_style':per_style,'counts_style_code_content':counts.tolist(),
            'routing_scope':'fixed8 test pages, one/style; diagnostic only; cyclic within-page permutation; no fine-tuning',
            'routing':routing,'state_unchanged':True}
        output = root/'acceptance_20260914_s8'/f'routing__{cp.stem}__test.json'
        output.write_text(json.dumps(result,indent=2)+'\n')
        print('S8_ROUTING',arm,state['epoch'],json.dumps({
            'styles':{name:{key:r[key] for key in ['fixed_global_mapping_accuracy','active_codes','usage_perplexity','largest_code_id','largest_code_fraction']} for name,r in per_style.items()},
            'routing':routing}),flush=True)


if __name__ == '__main__':
    main()
