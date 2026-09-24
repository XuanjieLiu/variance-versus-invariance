"""Read-only retained-checkpoint EMA audit; predicted replacements are not replay."""
import json
from pathlib import Path
import socket
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import yaml
from dataloader.uppercase_letters_dataloader import get_dataloader
from model.factory import get_model
from utils.subset_sampling import select_subset_indices
from utils.training_diagnostics import preserve_model_and_rng, usage_metrics
from utils.codebook_metrics import file_sha256


def main():
    assert socket.gethostname().startswith('ws-') and torch.cuda.is_available()
    torch.set_num_threads(4)
    for arm in ('W0', 'W50'):
        root = ROOT/'logs'/f'20260913-1106__VVI-RQ2-V2-C512-K26-{arm}-S0'
        config = yaml.safe_load((root/'config.yaml').read_text())
        loader = get_dataloader(str(ROOT/config['data_dir']/'val'), batch_size=32,
                                num_workers=0, n_fragments=26, fragment_len=32, shuffle=False)
        ids, _ = select_subset_indices(loader.dataset, 32, 0, 'style_stratified')
        batch = torch.stack([loader.dataset[i][0] for i in ids]).cuda()
        reports = []
        for epoch in (24, 49, 99, 199):
            cp = root/f'cp_snapshot_epoch{epoch}.pt'
            state = torch.load(cp, map_location='cuda', weights_only=False)
            model = get_model(config['dataloader'], config['model_config']).cuda()
            model.load_state_dict(state['model']); model.set_decoder_epoch(epoch); model.eval()
            cb = model.vq._codebook
            cluster = cb.cluster_size.detach().clone().flatten()
            norms = model.vq.codebook.detach().norm(dim=-1).flatten()
            report = {'epoch': epoch, 'checkpoint_sha256': file_sha256(cp),
                      'cluster_mass': cluster.tolist(), 'codes_at_reset_mass16': int((cluster == 16).sum()),
                      'atom_norm_min_median_max': [float(norms.min()), float(norms.median()), float(norms.max())],
                      'modes': {}}
            for mode in ('eval', 'encoder_batch_stats'):
                with torch.inference_mode(), preserve_model_and_rng(model):
                    model.eval()
                    if mode == 'encoder_batch_stats':
                        for module in model.encoder.modules():
                            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm): module.train()
                    out = model(batch, freeze_codebook=True)
                    counts = torch.bincount(out[3].flatten().long(), minlength=26)
                    next_mass = .98*cluster + .02*counts
                    report['modes'][mode] = {**usage_metrics(counts),
                        'raw_quantization_mse': float((out[1]-out[2]).square().mean()),
                        'would_expire_next_ema_step_at16': int((next_mass < 16).sum()),
                        'would_expire_next_ema_step_at3_2': int((next_mass < 3.2).sum())}
            reports.append(report)
            print('EMA_AUDIT', arm, json.dumps(report), flush=True)
            del model, state; torch.cuda.empty_cache()
        output = root/'acceptance_20260913_s7'/'ema_audit.json'
        output.write_text(json.dumps({'protocol': 'Frozen snapshot on fixed32 balanced val pages; no updates. Next-step expiration is hypothetical, not measured training history. Lower threshold comparison uses original stored EMA mass, not a retrained checkpoint.',
                                      'checkpoints': reports}, indent=2)+'\n')


if __name__ == '__main__': main()
