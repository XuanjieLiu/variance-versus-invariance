"""GPU-only full-epoch preflight for S10 epoch199 -> 200 continuations."""
import argparse
import csv
import json
import os
from pathlib import Path
import socket
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
import numpy as np
import torch
import yaml
from trainer import Trainer
from test_training_diagnostics import assert_equal
from utils.codebook_metrics import file_sha256
from utils.resume_artifacts import HISTORIES


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu(v) for v in value)
    return value


def rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['repo', 'mean'], required=True)
    args = parser.parse_args()
    assert socket.gethostname().startswith('ws-') and torch.cuda.is_available()
    os.chdir(ROOT)
    torch.set_num_threads(4)
    suffix = '_mean' if args.arm == 'mean' else ''
    config_path = ROOT / f'configs/uppercase/rq2/v2/cfg_vvi_rq2_v2_c512_k26_hexrepo{suffix}_seed0_resume1.yaml'
    config = yaml.safe_load(config_path.read_text())
    parent = Path(config['load_checkpoint']).resolve().parent
    original = yaml.safe_load((parent / 'config.yaml').read_text())
    allowed = {'name', 'epochs', 'load_checkpoint', 'inherit_resume_history'}
    for key in (set(config) | set(original)) - allowed:
        assert config.get(key) == original.get(key), key
    assert config['epochs'] == 570 and config['snapshot_every_n_epochs'] == 25
    assert file_sha256(Path(config['data_dir']) / 'manifest.json') == config['dataset_manifest_sha256']
    source = torch.load(config['load_checkpoint'], map_location='cpu', weights_only=False)
    assert source['epoch'] == 199
    assert {int(s['step']) for s in source['optimizer']['state'].values()} == {130000}
    report = {'arm': args.arm, 'node': socket.gethostname(),
              'config_sha256': file_sha256(config_path),
              'source_checkpoint': config['load_checkpoint'],
              'source_checkpoint_sha256': file_sha256(config['load_checkpoint']),
              'dataset_manifest_sha256': config['dataset_manifest_sha256'],
              'source_lr': source['optimizer']['param_groups'][0]['lr'],
              'source_scheduler': source['scheduler'], 'source_scaler': source['scaler'],
              'start_epoch': 200, 'target_epoch': 769, 'target_steps': 500500,
              'rng_restored': False, 'note': 'Source checkpoint lacks RNG; not bitwise uninterrupted.'}
    history = rows(parent / 'codebook_epoch_history.csv')
    healthy = [float(r['macro_atom_purity']) >= .70 and int(r['active_codes']) == 26
               and float(r['usage_perplexity']) >= 20 and int(r['dominant_label_coverage']) >= 24
               for r in history]
    report['stable_windows'] = [[i, i + 9] for i in range(len(healthy) - 9) if sum(healthy[i:i+10]) >= 8]
    report['parent_current_validation'] = history[-1]
    report['parent_macro_best'] = json.loads((parent / 'best_macro_atom_purity.json').read_text())
    config.update(name=f'smoke-s10-resume-{args.arm}', epochs=1)
    directory = ROOT / 'logs' / config['name']
    assert not directory.exists(), directory
    trainer = Trainer(config)
    trainer.prepare_data()
    trainer.build_model()
    # Keep the inherited protocol identical; force one extra smoke-only drawing
    # after its compatibility check, without changing diagnostic samples/metrics.
    original_begin = trainer.disentanglement_monitor.begin
    def smoke_begin(*args, **kwargs):
        original_begin(*args, **kwargs)
        trainer.disentanglement_monitor.draw = True
    trainer.disentanglement_monitor.begin = smoke_begin
    assert trainer.start_epoch == 200
    assert len(trainer.train_loader) == 650 and len(trainer.val_loader.dataset) == 2600
    case = unittest.TestCase()
    assert_equal(case, source['model'], cpu(trainer.model.state_dict()))
    assert_equal(case, source['optimizer'], cpu(trainer.optimizer.state_dict()))
    assert_equal(case, source['scheduler'], trainer.scheduler.state_dict())
    assert_equal(case, source['scaler'], trainer.scaler.state_dict())
    for metadata in ('best_macro_atom_purity.json', 'best_validation_loss.json'):
        a = json.loads((parent / metadata).read_text())
        b = json.loads((directory / metadata).read_text())
        assert a['epoch'] == b['epoch']
        assert file_sha256(parent / a['checkpoint']) == file_sha256(directory / b['checkpoint'])
    for name in HISTORIES:
        if (parent / name).exists():
            assert rows(parent / name) == rows(directory / name), name
    assert trainer.start_epoch + 570 - 1 == 769
    del source
    trainer.train()
    expected_mode = 'page_mean' if args.arm == 'mean' else 'fragment'
    assert trainer.model.active_decoder_style_mode == expected_mode
    assert (directory / 'cp_current_epoch200.pt').is_file()
    assert not list(directory.glob('cp_snapshot_epoch*.pt'))
    assert len(list(directory.glob('cp*.pt'))) <= 3
    saved = torch.load(directory / 'cp_current_epoch200.pt', map_location='cpu', weights_only=False)
    assert saved['epoch'] == 200
    assert {int(s['step']) for s in saved['optimizer']['state'].values()} == {130650}
    assert saved['scheduler']['last_epoch'] == 201
    for name in HISTORIES:
        if (parent / name).exists():
            values = rows(directory / name)
            assert int(values[-1]['epoch']) == 200, name
            assert int(values[-1]['global_step']) == 130650, name
    for stem in ('loss_curves', 'codebook_metrics', 'v3_ratios', 'foreground_color_metrics',
                 'disentanglement_probes', 'training_codebook_usage', 'bn_diagnostics',
                 'optimization_scales', 'per_style_codebook_metrics'):
        assert (directory / f'{stem}.png').is_file(), stem
    matrices = list((directory / 'reconstruction_diagnostics').glob('*epoch200*column-normalized*.json'))
    assert len(matrices) == 1
    matrix = json.loads(matrices[0].read_text())
    np.testing.assert_array_equal(matrix['content_style_counts'], np.full((26, 8), 325))
    for ext in ('.png', '.svg', '.csv'):
        assert matrices[0].with_suffix(ext).is_file()
    bn_rows = [r for r in rows(directory / 'bn_diagnostic_epoch_history.csv') if int(r['epoch']) == 200]
    assert len(bn_rows) == 4
    assert next(r for r in bn_rows if r['mode'] == 'decoder_batch_stats')['code_changed_fraction'] == '0.0'
    report['smoke_current_validation'] = rows(directory / 'codebook_epoch_history.csv')[-1]
    report['restoration_exact'] = ['model', 'optimizer', 'scheduler', 'scaler', 'macro_best_weights', 'best_val_weights']
    report['inherited_histories'] = json.loads((directory / 'resume_lineage.json').read_text())['inherited_history_rows']
    report['smoke_completed_epoch'] = 200
    report['smoke_completed_steps'] = 130650
    report['checkpoints'] = sorted(p.name for p in directory.glob('cp*.pt'))
    output = ROOT / 'logs/diagnostics/20260917_s10_resume_preflight'
    output.mkdir(parents=True, exist_ok=True)
    (output / f'{args.arm}.json').write_text(json.dumps(report, indent=2) + '\n')
    trainer.optimization_scale_monitor.close()
    print('S10_RESUME_SMOKE_VALIDATED', args.arm, str(directory), flush=True)


if __name__ == '__main__':
    main()
