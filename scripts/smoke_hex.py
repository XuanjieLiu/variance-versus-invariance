"""One complete train/validation epoch per Hex arm, on a compute GPU only."""
import argparse
import copy
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
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
import numpy as np
import torch
import yaml
from trainer import Trainer
from model.factory import get_model
from smoke_v2 import rng_state, assert_rng_equal
from test_training_diagnostics import assert_equal
from utils.codebook_metrics import file_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['repo', 'mean'], required=True)
    args = parser.parse_args()
    assert socket.gethostname().startswith('ws-') and torch.cuda.is_available()
    os.chdir(ROOT); torch.set_num_threads(4)
    suffix = '_mean' if args.arm == 'mean' else ''
    config_path = ROOT / f'configs/hex/v2/cfg_vvi_hex_k16_repo{suffix}_seed0.yaml'
    config = yaml.safe_load(config_path.read_text())
    assert file_sha256(Path(config['data_dir']) / 'manifest.json') == config['dataset_manifest_sha256']
    config.update(name=f'smoke-hex-s1-{args.arm}', debug=False, epochs=1,
                  debug_portion=1, snapshot_every_n_epochs=1)
    directory = ROOT / 'logs' / config['name']
    assert not directory.exists(), directory
    trainer = Trainer(config); trainer.prepare_data(); trainer.build_model()
    assert len(trainer.train_loader.dataset) == 80000 and len(trainer.train_loader) == 2500
    assert len(trainer.val_loader.dataset) == 10000
    initial = json.loads((directory / 'initial_model_state.json').read_text())
    norms = json.loads((directory / 'normalization_config.json').read_text())
    assert norms['encoder_bn_layers'] == 32 and norms['decoder_bn_layers'] == 28
    assert norms['encoder'] == norms['decoder'] == 'batch'
    assert not norms['converted_layers'] and not norms['encoder_converted_layers']
    state = rng_state()
    order = np.asarray(list(iter(trainer.train_loader.sampler)), dtype=np.int64)
    order_hash = hashlib.sha256(order.tobytes()).hexdigest()
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2]); torch.cuda.set_rng_state_all(state[3])
    expected_mode = 'page_mean' if args.arm == 'mean' else 'fragment'
    style = torch.arange(2 * 16 * 512, dtype=torch.float32, device='cuda').reshape(2, 16, 512)
    for epoch in (0, 49, 50, 199):
        trainer.model.set_decoder_epoch(epoch)
        assert trainer.model.active_decoder_style_mode == expected_mode
        expected = style.mean(1, keepdim=True).expand_as(style) if args.arm == 'mean' else style
        torch.testing.assert_close(trainer.model.decoder_style_input(style), expected)
    trainer.model.set_decoder_epoch(0)
    # Wrap all expensive validation diagnostic finishes, verifying they do not
    # perturb model/optimizer/RNG or training module modes.
    for monitor in (trainer.bn_diagnostic, trainer.disentanglement_monitor, trainer.per_style_codebook_monitor):
        original = monitor.finish
        def checked(*a, _original=original, **kw):
            before = rng_state(); model_state = copy.deepcopy(trainer.model.state_dict())
            optimizer = copy.deepcopy(trainer.optimizer.state_dict())
            flags = [m.training for m in trainer.model.modules()]
            result = _original(*a, **kw)
            assert_rng_equal(before, rng_state())
            assert_equal(unittest.TestCase(), model_state, trainer.model.state_dict())
            assert_equal(unittest.TestCase(), optimizer, trainer.optimizer.state_dict())
            assert flags == [m.training for m in trainer.model.modules()]
            return result
        monitor.finish = checked
    trainer.train()
    tables = ('loss_epoch_history', 'codebook_epoch_history', 'v3_ratio_epoch_history',
              'foreground_color_epoch_history', 'disentanglement_probe_history',
              'training_codebook_usage', 'bn_diagnostic_epoch_history',
              'optimization_scale_epoch_history', 'per_style_codebook_epoch_history',
              'content_group_epoch_history')
    records = {}
    for table in tables:
        with (directory / f'{table}.csv').open() as handle: rows = list(csv.DictReader(handle))
        assert rows and int(rows[-1]['epoch']) == 0, table
        records[table] = rows
    assert int(records['training_codebook_usage'][-1]['fragment_count']) == 80000 * 16
    assert int(records['optimization_scale_epoch_history'][-1]['sampled_steps']) == 25
    assert int(records['optimization_scale_epoch_history'][-1]['all_scales_finite']) == 1
    assert len(records['per_style_codebook_epoch_history']) == 8
    assert all(int(r['fragment_count']) == 20000 for r in records['per_style_codebook_epoch_history'])
    assert {r['group'] for r in records['content_group_epoch_history']} == {'digits_0_9', 'letters_a_f'}
    assert sum(int(r['fragment_count']) for r in records['content_group_epoch_history']) == 160000
    modes = records['bn_diagnostic_epoch_history']
    assert len(modes) == 4 and all(int(r['page_count']) == 32 for r in modes)
    assert next(r for r in modes if r['mode'] == 'decoder_batch_stats')['code_changed_fraction'] == '0.0'
    for filename in ('loss_curves.png', 'codebook_metrics.png', 'v3_ratios.png',
                     'foreground_color_metrics.png', 'disentanglement_probes.png',
                     'training_codebook_usage.png', 'bn_diagnostics.png', 'optimization_scales.png',
                     'per_style_codebook_metrics.png', 'content_group_metrics.png',
                     'cp_current_epoch0.pt', 'cp_best_validation_loss_epoch0.pt', 'cp_snapshot_epoch0.pt'):
        assert (directory / filename).is_file(), filename
    checkpoints = sorted(p.name for p in directory.glob('cp*.pt'))
    assert len(checkpoints) <= 4
    matrix_files = list((directory / 'reconstruction_diagnostics').glob('*column-normalized*.json'))
    assert len(matrix_files) == 1
    matrix = json.loads(matrix_files[0].read_text())
    np.testing.assert_array_equal(matrix['content_style_counts'], np.full((16, 8), 1250))
    np.testing.assert_array_equal(np.asarray(matrix['display_hundredths']).sum(0), np.full(16, 100))
    assert list(matrix['content_group_metrics']['groups']) == ['digits_0_9', 'letters_a_f']
    for ext in ('.svg', '.png', '.csv'): assert matrix_files[0].with_suffix(ext).is_file()
    checkpoint = torch.load(directory / 'cp_current_epoch0.pt', map_location='cpu', weights_only=False)
    restored = get_model(config['dataloader'], config['model_config'])
    restored.load_state_dict(checkpoint['model'], strict=True)
    assert restored.active_decoder_style_mode == expected_mode
    assert_equal(unittest.TestCase(), trainer.model.state_dict(),
                 {key: value.to(trainer.device) for key, value in restored.state_dict().items()})
    trainer.optimization_scale_monitor.close()
    bn_protocol = json.loads((directory / 'bn_diagnostic_protocol.json').read_text())
    report = {'arm': args.arm, 'node': socket.gethostname(), 'initial_model': initial,
              'normalization': norms, 'first_epoch_sampler_sha256': order_hash,
              'bn_pages_sha256': bn_protocol['pages_sha256'], 'checkpoints': checkpoints,
              'config_sha256': file_sha256(config_path), 'dataset_manifest_sha256': config['dataset_manifest_sha256'],
              'training_steps': 2500, 'validation_pages': 10000, 'validation_fragments': 160000,
              'content_group_validation': records['content_group_epoch_history']}
    del restored, checkpoint, trainer; torch.cuda.empty_cache()
    subprocess.run([sys.executable, 'run_evaluation.py', '--run', str(directory),
                    '--active_checkpoint', 'current', '--confusion_mtx', '--test_subset_size', '512',
                    '--test_subset_seed', '0', '--test_subset_strategy', 'style_stratified'], check=True)
    evaluation = list((directory / 'vis').glob('*test-n512*.json'))
    assert len(evaluation) == 1
    data = json.loads(evaluation[0].read_text())
    assert set(data['content_group_metrics']['groups']) == {'digits_0_9', 'letters_a_f'}
    subprocess.run([sys.executable, 'run_codebook_health.py', '--config', str(directory / 'config.yaml'),
                    '--checkpoint', str(directory / 'cp_current_epoch0.pt'),
                    '--test-subset-size', '512', '--test-subset-seed', '0',
                    '--test-subset-strategy', 'style_stratified', '--output', str(directory / 'health_quick.json')], check=True)
    health = json.loads((directory / 'health_quick.json').read_text())
    assert health['sample_count'] == 512 and health['fragment_count'] == 8192
    np.testing.assert_array_equal(health['per_style_codebook']['content_style_counts'], np.full((16, 8), 64))
    assert all(m['one_to_one_accuracy'] == m['global_mapping_accuracy']
               for m in health['per_style_codebook']['metrics'].values())
    assert set(health['content_group_metrics']['groups']) == {'digits_0_9', 'letters_a_f'}
    output = ROOT / 'logs/diagnostics/20260915_hex_preflight'
    output.mkdir(parents=True, exist_ok=True)
    (output / f'{args.arm}.json').write_text(json.dumps(report, indent=2) + '\n')
    print('HEX_SMOKE_VALIDATED', args.arm, str(directory), flush=True)
    print('Inspect reconstruction/matrix then delete this exact smoke directory.', flush=True)


if __name__ == '__main__': main()
