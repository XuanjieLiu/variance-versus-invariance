"""GPU-only full-test acceptance and read-only trajectory summary for S7."""
import csv
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
from run_codebook_health import evaluate
from utils.codebook_metrics import file_sha256


def read_rows(root, name):
    with (root / name).open() as handle:
        return list(csv.DictReader(handle))


def number(row, key):
    return float(row[key])


def summarize(root):
    codes = read_rows(root, 'codebook_epoch_history.csv')
    losses = read_rows(root, 'loss_epoch_history.csv')
    probes = read_rows(root, 'disentanglement_probe_history.csv')
    bn = read_rows(root, 'bn_diagnostic_epoch_history.csv')
    usage = read_rows(root, 'training_codebook_usage.csv')
    ratios = read_rows(root, 'v3_ratio_epoch_history.csv')
    healthy = lambda r: (number(r, 'active_codes') >= 24 and
                         number(r, 'usage_perplexity') >= 18 and
                         number(r, 'dominant_label_coverage') >= 18 and
                         number(r, 'codebook_purity') >= .20)
    stable = lambda r: (number(r, 'macro_atom_purity') >= .70 and
                        number(r, 'active_codes') == 26 and
                        number(r, 'usage_perplexity') >= 20 and
                        number(r, 'dominant_label_coverage') >= 24)
    windows = [int(codes[i]['epoch']) for i in range(len(codes)-9)
               if sum(stable(r) for r in codes[i:i+10]) >= 8]
    val = [r for r in losses if r['partition'] == 'val']
    checkpoints = list(root.glob('cp*.pt'))
    summary = {'run': root.name, 'completed_epoch': int(codes[-1]['epoch']),
               'validation_epochs': len(codes), 'healthy_epochs': sum(map(healthy, codes)),
               'stable_window_starts': windows, 'checkpoint_count': len(checkpoints),
               'max_macro': max(codes, key=lambda r: number(r, 'macro_atom_purity')),
               'max_hungarian': max(codes, key=lambda r: number(r, 'one_to_one_accuracy')),
               'peak_val_recon': max(val, key=lambda r: number(r, 'recon_loss')),
               'last_50_mean_macro': float(np.mean([number(r, 'macro_atom_purity') for r in codes[-50:]])),
               'last_50_mean_hungarian': float(np.mean([number(r, 'one_to_one_accuracy') for r in codes[-50:]]))}
    epochs = set(range(16)) | {20, 24, 28, 40, 49, 50, 51, 60, 74, 99, 100, 102, 124, 149, 174, 199}
    selected = {}
    for epoch in sorted(epochs):
        match = lambda row: int(row['epoch']) == epoch
        selected[str(epoch)] = {
            'codebook': next(r for r in codes if match(r)),
            'loss': {r['partition']: r for r in losses if match(r)},
            'bn': {r['mode']: r for r in bn if match(r)},
            'probes': next(r for r in probes if match(r)),
            'usage': next(r for r in usage if match(r) and r['scope'] == 'epoch'),
            'ratios': {r['partition']: r for r in ratios if match(r)}}
    summary['epochs'] = selected
    assert len(codes) == 200 and int(codes[-1]['epoch']) == 199
    assert all(float(r['code_changed_fraction']) == 0 for r in bn if r['mode'] == 'decoder_batch_stats')
    assert len(checkpoints) <= 11
    return summary


def main():
    assert Path.cwd() == ROOT
    assert socket.gethostname().startswith('ws-') and torch.cuda.is_available()
    torch.set_num_threads(4)
    for arm in ('W0', 'W50'):
        root = ROOT / 'logs' / f'20260913-1106__VVI-RQ2-V2-C512-K26-{arm}-S0'
        directory = root / 'acceptance_20260913_s7'
        directory.mkdir(exist_ok=True)
        summary = summarize(root)
        (directory / 'trajectory_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print('TRAJECTORY', arm, json.dumps({k: v for k, v in summary.items() if k != 'epochs'}), flush=True)
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
                result = {**evaluate(config, cp), **identity, 'selection': role,
                          'checkpoint': str(cp), 'epoch': metadata['epoch'],
                          'evaluated_at': datetime.datetime.now().astimezone().isoformat()}
                assert result['sample_count'] == 2600 and result['fragment_count'] == 67600
                output.write_text(json.dumps(result, indent=2) + '\n')
            accepted.append({'role': role, 'report': output.name, **result})
            print('FULL_TEST', arm, role, metadata['epoch'], json.dumps(result), flush=True)
            grid = directory / f'reconstruction_grid__{cp.stem}__test-full-map__8styles__viz-v2.json'
            if not grid.exists():
                subprocess.run([sys.executable, 'run_reconstruction_grid.py', '--run', str(root),
                                '--active_checkpoint', role, '--output_dir', str(directory)], check=True)
        (directory / 'acceptance_index.json').write_text(json.dumps(accepted, indent=2) + '\n')
        print('S7_ACCEPTANCE_DONE', arm, flush=True)


if __name__ == '__main__':
    main()
