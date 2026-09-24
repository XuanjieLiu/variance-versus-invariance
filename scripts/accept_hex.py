"""Full-test acceptance of HEX-S1; only execute on a GPU compute node."""
import argparse
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
from utils.training_utils import setup_seed


def rows(root, name):
    with (root / name).open() as handle:
        return list(csv.DictReader(handle))


def trajectories(root):
    codes = rows(root, 'codebook_epoch_history.csv')
    losses = rows(root, 'loss_epoch_history.csv')
    probes = rows(root, 'disentanglement_probe_history.csv')
    bn = rows(root, 'bn_diagnostic_epoch_history.csv')
    stable = lambda r: (float(r['macro_atom_purity']) >= .70 and
        float(r['one_to_one_accuracy']) >= .70 and int(r['active_codes']) == 16 and
        float(r['usage_perplexity']) >= 13 and int(r['dominant_label_coverage']) >= 15)
    assert [int(r['epoch']) for r in codes] == list(range(200))
    windows = [i for i in range(191) if sum(stable(r) for r in codes[i:i+10]) >= 8]
    val = [r for r in losses if r['partition'] == 'val']
    result = {'run': root.name, 'completed_epoch': 199, 'optimizer_steps': int(codes[-1]['global_step']),
        'checkpoint_count': len(list(root.glob('cp*.pt'))), 'stable_window_starts': windows,
        'first_stable_window': [windows[0], windows[0]+9] if windows else None,
        'last50': {key: float(np.mean([float(r[key]) for r in codes[-50:]]))
                   for key in ['macro_atom_purity', 'one_to_one_accuracy', 'usage_perplexity']},
        'last_validation': codes[-1], 'last_probe': probes[-1],
        'max_validation_recon': max(val, key=lambda r: float(r['recon_loss'])),
        'validation_recon_over1_epochs': sum(float(r['recon_loss']) > 1 for r in val),
        'codebook_timeline': codes, 'loss_timeline': losses, 'bn_timeline': bn,
        'group_timeline': rows(root, 'content_group_epoch_history.csv'),
        'per_style_timeline': rows(root, 'per_style_codebook_epoch_history.csv')}
    assert result['checkpoint_count'] <= 11 and result['optimizer_steps'] == 500000
    assert all(float(r['code_changed_fraction']) == 0 for r in bn if r['mode'] == 'decoder_batch_stats')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['repo', 'mean', 'both'], default='both')
    args = parser.parse_args()
    assert Path.cwd() == ROOT and socket.gethostname().startswith('ws-') and torch.cuda.is_available()
    torch.set_num_threads(4)
    for arm in (['repo', 'mean'] if args.arm == 'both' else [args.arm]):
        suffix = '-MEAN' if arm == 'mean' else ''
        root = ROOT / f'logs/20260915-1121__VVI-HEX-K16-REPO{suffix}-S0'
        output = root / 'acceptance_20260916_hex'
        output.mkdir(exist_ok=True)
        summary = trajectories(root)
        (output / 'trajectory_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print('TRAJECTORY', arm, json.dumps({k:v for k,v in summary.items() if not k.endswith('timeline')}), flush=True)
        config = yaml.safe_load((root / 'config.yaml').read_text())
        accepted = []
        for role, name in [('best', 'best_macro_atom_purity.json'), ('best_val', 'best_validation_loss.json'), ('current', 'current_checkpoint.json')]:
            meta = json.loads((root / name).read_text()); cp = root / meta['checkpoint']
            report = output / f'health__{cp.stem}__test-full.json'
            identity = {'checkpoint_sha256': file_sha256(cp), 'config_sha256': file_sha256(root / 'config.yaml'),
                        'dataset_manifest_sha256': config['dataset_manifest_sha256']}
            if report.exists():
                result = json.loads(report.read_text())
                assert all(result[k] == v for k,v in identity.items())
            else:
                setup_seed(0)
                result = {**evaluate(config, cp), **identity, 'checkpoint': str(cp), 'epoch': meta['epoch'],
                    'evaluated_at': datetime.datetime.now().astimezone().isoformat()}
                assert result['sample_count'] == 10000 and result['fragment_count'] == 160000
                np.testing.assert_array_equal(result['per_style_codebook']['content_style_counts'], np.full((16,8),1250))
                report.write_text(json.dumps(result, indent=2) + '\n')
            c, loss = result['codebook'], result['losses']
            passed = (c['macro_atom_purity'] >= .75 and c['codebook_purity'] >= .75 and
                c['one_to_one_accuracy'] >= .75 and c['active_codes'] == 16 and
                c['usage_perplexity'] >= 13 and c['dominant_label_coverage'] >= 15 and loss['recon_loss'] <= .30)
            accepted.append({'role': role, 'report': report.name, 'semantic_gate_passed': passed, **result})
            print('FULL_TEST', arm, role, meta['epoch'], json.dumps({'codebook':c, 'loss':loss,
                'color': {k:result['foreground_color'][k] for k in ['foreground_rgb_mae','foreground_chroma_rmse']},
                'groups':result['content_group_metrics'], 'gate':passed}), flush=True)
            grid = output / f'reconstruction_grid__{cp.stem}__test-full-map__8styles__viz-v2.json'
            if not grid.exists():
                subprocess.run([sys.executable,'run_reconstruction_grid.py','--run',str(root),
                    '--active_checkpoint',role,'--output_dir',str(output)],check=True)
        (output / 'acceptance_index.json').write_text(json.dumps(accepted, indent=2) + '\n')
        print('HEX_ACCEPTANCE_DONE', arm, flush=True)


if __name__ == '__main__': main()
