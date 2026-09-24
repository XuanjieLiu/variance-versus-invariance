"""Persist full-test acceptance for the completed PhoneNums/C128-W50 round."""
import datetime
import json
from pathlib import Path
import socket
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import yaml
from run_codebook_health import evaluate
from utils.codebook_metrics import file_sha256


def main():
    assert socket.gethostname().startswith('ws-') and torch.cuda.is_available()
    for suffix, roles in (
        ('VVI-PN-K10-REPO-S0', ('best', 'best_val', 'current')),
        ('VVI-PN-K10-PAPER-S0', ('best', 'best_val', 'current')),
        ('VVI-RQ2-V2-C128-K26-W50-S0', ('best', 'current')),
    ):
        root = Path('logs') / f'20260912-1420__{suffix}'
        config = yaml.safe_load((root / 'config.yaml').read_text())
        directory = root / 'acceptance_20260913'
        directory.mkdir(exist_ok=True)
        for role in roles:
            metadata_file = {'best': 'best_macro_atom_purity.json',
                             'best_val': 'best_validation_loss.json',
                             'current': 'current_checkpoint.json'}[role]
            metadata = json.loads((root / metadata_file).read_text())
            cp = root / metadata['checkpoint']
            path = directory / f'health__{cp.stem}__test-full.json'
            identity = {'checkpoint_sha256': file_sha256(cp),
                        'config_sha256': file_sha256(root / 'config.yaml'),
                        'dataset_manifest_sha256': config['dataset_manifest_sha256']}
            if path.exists():
                result = json.loads(path.read_text())
                assert all(result[k] == v for k, v in identity.items())
            else:
                result = {**evaluate(config, cp), **identity,
                          'checkpoint': str(cp.resolve()), 'selection': role,
                          'epoch': metadata['epoch'],
                          'evaluated_at': datetime.datetime.now().astimezone().isoformat()}
                path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
            grid = directory / f'reconstruction_grid__{cp.stem}__test-full-map__8styles__viz-v2.json'
            if not grid.exists():
                args = [sys.executable, 'run_reconstruction_grid.py', '--run', str(root),
                        '--active_checkpoint', role, '--output_dir', str(directory)]
                if suffix.startswith('VVI-PN'):
                    args.append('--figure17')
                subprocess.run(args, check=True)
            print('ACCEPTED', suffix, role, metadata['epoch'],
                  'macro', result['codebook']['macro_atom_purity'],
                  'one_to_one', result['codebook']['one_to_one_accuracy'],
                  'recon', result['losses']['recon_loss'], flush=True)


if __name__ == '__main__':
    main()
