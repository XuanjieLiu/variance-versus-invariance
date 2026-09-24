"""GPU-only S8 acceptance; does not mutate training checkpoints or configuration."""
import datetime
import json
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import yaml
from scripts.accept_s7 import read_rows, summarize
from run_codebook_health import evaluate
from utils.codebook_metrics import file_sha256
from utils.training_utils import setup_seed


def trajectories(root):
    result = summarize(root)
    codes = read_rows(root, 'codebook_epoch_history.csv')
    loss = read_rows(root, 'loss_epoch_history.csv')
    bn = read_rows(root, 'bn_diagnostic_epoch_history.csv')
    probes = read_rows(root, 'disentanglement_probe_history.csv')
    result['phases'] = {}
    for name, start, end in [('warmup', 0, 50), ('released', 50, 200), ('last50', 150, 200)]:
        select = lambda rows: [r for r in rows if start <= int(r['epoch']) < end]
        v = [float(r['recon_loss']) for r in select(loss) if r['partition'] == 'val']
        c = select(codes)
        result['phases'][name] = {
            'val_recon_median_p95_max': np.quantile(v, [.5, .95, 1]).tolist(),
            'val_recon_over1_epochs': sum(x > 1 for x in v),
            **{key + '_mean': float(np.mean([float(r[key]) for r in c]))
               for key in ['macro_atom_purity', 'one_to_one_accuracy', 'active_codes',
                           'usage_perplexity', 'dominant_label_coverage']}}
    result['compact_timeline'] = []
    for c in codes:
        epoch = int(c['epoch'])
        modes = {r['mode']: r for r in bn if int(r['epoch']) == epoch}
        p = next(r for r in probes if int(r['epoch']) == epoch)
        l = {r['partition']: r for r in loss if int(r['epoch']) == epoch}
        row = {'epoch': epoch, **{k: float(c[k]) for k in ['macro_atom_purity',
               'one_to_one_accuracy', 'codebook_purity', 'active_codes', 'usage_perplexity',
               'dominant_label_coverage']},
               **{part + '_' + k: float(l[part][k]) for part in ['train', 'val']
                  for k in ['recon_loss', 'commit_loss', 'v3_loss']},
               **{k: float(p[k]) for k in ['ec_vq_style_accuracy', 'zs_content_raw_accuracy',
                  'zs_content_whitened_accuracy', 'foreground_rgb_mae', 'foreground_chroma_rmse']},
               'bn': {mode: {k: float(r[k]) for k in ['recon_loss', 'fixed_mapping_accuracy',
                     'code_changed_fraction', 'usage_perplexity', 'active_codes']}
                      for mode, r in modes.items()}}
        result['compact_timeline'].append(row)
        if 'DGN8' in root.name:
            assert int(modes['decoder_batch_stats']['switched_bn_layer_count']) == 0
            for k in ['recon_loss', 'fixed_mapping_accuracy', 'code_changed_fraction']:
                assert modes['eval'][k] == modes['decoder_batch_stats'][k]
    return result


def main():
    assert Path.cwd() == ROOT
    assert socket.gethostname().startswith('ws-') and torch.cuda.is_available()
    torch.set_num_threads(4)
    print('S8_ACCEPTANCE_NODE', socket.gethostname(), flush=True)
    for arm in ('BN', 'DGN8'):
        root = ROOT / 'logs' / f'20260913-2309__VVI-RQ2-V2-C512-K26-W50-LR1E4-{arm}-S0'
        directory = root / 'acceptance_20260914_s8'
        directory.mkdir(exist_ok=True)
        summary = trajectories(root)
        (directory / 'trajectory_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print('TRAJECTORY', arm, json.dumps({k: v for k, v in summary.items()
              if k not in ('epochs', 'compact_timeline')}), flush=True)
        config = yaml.safe_load((root / 'config.yaml').read_text())
        accepted = []
        for role, metadata_file in [('best', 'best_macro_atom_purity.json'),
                                    ('best_val', 'best_validation_loss.json'),
                                    ('current', 'current_checkpoint.json')]:
            if not (root / metadata_file).exists():
                assert role == 'best' and summary['healthy_epochs'] == 0
                accepted.append({'role': role, 'missing_reason': 'No validation epoch passes registered macro health gate'})
                print('NO_HEALTHY_MACRO_BEST', arm, flush=True)
                continue
            metadata = json.loads((root / metadata_file).read_text())
            cp = root / metadata['checkpoint']
            output = directory / f'health__{cp.stem}__test-full.json'
            identity = {'checkpoint_sha256': file_sha256(cp),
                        'config_sha256': file_sha256(root / 'config.yaml'),
                        'dataset_manifest_sha256': config['dataset_manifest_sha256']}
            if output.exists():
                result = json.loads(output.read_text())
                assert all(result[k] == v for k, v in identity.items())
            else:
                setup_seed(0)
                result = {**evaluate(config, cp), **identity, 'selection': role,
                          'checkpoint': str(cp), 'epoch': metadata['epoch'],
                          'evaluated_at': datetime.datetime.now().astimezone().isoformat()}
                assert result['sample_count'] == 2600 and result['fragment_count'] == 67600
                output.write_text(json.dumps(result, indent=2) + '\n')
            accepted.append({'role': role, 'report': output.name, **result})
            print('FULL_TEST', arm, role, metadata['epoch'], json.dumps({
                'codebook': result['codebook'], 'losses': result['losses'],
                'fixed_batch': result['fixed_batch'],
                'color': {k: result['foreground_color'][k] for k in
                          ['foreground_rgb_mae', 'foreground_chroma_rmse']}}), flush=True)
            grid = directory / f'reconstruction_grid__{cp.stem}__test-full-map__8styles__viz-v2.json'
            if not grid.exists():
                subprocess.run([sys.executable, 'run_reconstruction_grid.py', '--run', str(root),
                                '--active_checkpoint', role, '--output_dir', str(directory)], check=True)
        (directory / 'acceptance_index.json').write_text(json.dumps(accepted, indent=2) + '\n')
        print('S8_ACCEPTANCE_DONE', arm, flush=True)


if __name__ == '__main__':
    main()
