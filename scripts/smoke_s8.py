"""Full-validation GPU preflight for the matched decoder BN / GN8 pair."""
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
from smoke_v2 import TwoBatches, rng_state, assert_rng_equal
from test_training_diagnostics import assert_equal
from utils.codebook_metrics import file_sha256


def main():
    assert socket.gethostname().startswith('ws-') and torch.cuda.is_available()
    os.chdir(ROOT)
    torch.set_num_threads(4)
    reports = {}
    for arm in ('bn', 'dgn8'):
        config_path = ROOT/f'configs/uppercase/rq2/v2/normalization/cfg_vvi_rq2_v2_c512_k26_w50_lr1e4_{arm}_seed0.yaml'
        config = yaml.safe_load(config_path.read_text())
        assert file_sha256(Path(config['data_dir'])/'manifest.json') == config['dataset_manifest_sha256']
        # Keep the production evaluation path: Tester(debug=True) silently uses
        # 10% data, which is incompatible with a balanced512-page CLI check.
        config.update(name=f'smoke-s8-{arm}', debug=False, epochs=1,
                      debug_portion=1, snapshot_every_n_epochs=1)
        directory = ROOT/'logs'/config['name']
        assert not directory.exists(), directory
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler); handler.close()
        trainer = Trainer(config); trainer.prepare_data(); trainer.build_model()
        initial_info = json.loads((directory/'initial_model_state.json').read_text())
        initial = initial_info['sha256']
        parameters_hash = initial_info['parameters_sha256']
        normalization = json.loads((directory/'normalization_config.json').read_text())
        assert normalization['encoder_bn_layers'] > 0
        if arm == 'dgn8':
            assert normalization['decoder_bn_layers'] == 0
            assert normalization['decoder_groupnorm_layers'] > 0
            assert all(x['groups'] == 8 for x in normalization['converted_layers'])
        else:
            assert normalization['decoder_bn_layers'] > 0 and normalization['decoder_groupnorm_layers'] == 0
        before = rng_state()
        order = np.asarray(list(iter(trainer.train_loader.sampler)), dtype=np.int64)
        order_hash = hashlib.sha256(order.tobytes()).hexdigest()
        random.setstate(before[0]); np.random.set_state(before[1])
        torch.set_rng_state(before[2]); torch.cuda.set_rng_state_all(before[3])
        if arm in ('bn', 'dgn8'):
            style = torch.arange(2*26*512, dtype=torch.float32, device='cuda').reshape(2, 26, 512)
            trainer.model.set_decoder_epoch(49)
            assert trainer.model.active_decoder_style_mode == 'page_mean'
            torch.testing.assert_close(trainer.model.decoder_style_input(style), style.mean(1, keepdim=True).expand_as(style))
            trainer.model.set_decoder_epoch(50)
            assert trainer.model.active_decoder_style_mode == 'fragment'
            torch.testing.assert_close(trainer.model.decoder_style_input(style), style)
            trainer.model.set_decoder_epoch(0)
        bn = trainer.bn_diagnostic
        original = bn.finish
        def checked_finish(*args, **kwargs):
            before = rng_state()
            state = copy.deepcopy(trainer.model.state_dict())
            optimizer = copy.deepcopy(trainer.optimizer.state_dict())
            flags = [m.training for m in trainer.model.modules()]
            result = original(*args, **kwargs)
            assert_rng_equal(before, rng_state())
            assert_equal(unittest.TestCase(), state, trainer.model.state_dict())
            assert_equal(unittest.TestCase(), optimizer, trainer.optimizer.state_dict())
            assert flags == [m.training for m in trainer.model.modules()]
            assert len(result) == 4 and result[2]['code_changed_fraction'] == 0
            if arm == 'dgn8':
                assert result[2]['switched_bn_layer_count'] == 0
                assert result[2]['recon_loss'] == result[0]['recon_loss']
            return result
        bn.finish = checked_finish
        trainer.train_loader = TwoBatches(trainer.train_loader)
        trainer.train()
        for filename in ('loss_epoch_history.csv', 'codebook_epoch_history.csv', 'v3_ratio_epoch_history.csv',
                         'foreground_color_epoch_history.csv', 'disentanglement_probe_history.csv',
                         'training_codebook_usage.csv', 'bn_diagnostic_epoch_history.csv'):
            with (directory/filename).open() as handle: rows = list(csv.DictReader(handle))
            assert int(rows[-1]['epoch']) == 0
            if filename == 'bn_diagnostic_epoch_history.csv':
                assert len(rows) == 4 and all(int(row['page_count']) == 32 for row in rows)
                assert all(int(row['fragment_count']) == 832 for row in rows)
            if filename == 'training_codebook_usage.csv':
                assert len(rows) == 2 and int(rows[-1]['fragment_count']) == 2*32*26
        for filename in ('loss_curves.png', 'codebook_metrics.png', 'v3_ratios.png',
                         'foreground_color_metrics.png', 'disentanglement_probes.png',
                         'training_codebook_usage.png', 'bn_diagnostics.png',
                         'cp_current_epoch0.pt', 'cp_best_validation_loss_epoch0.pt', 'cp_snapshot_epoch0.pt'):
            assert (directory/filename).is_file(), filename
        checkpoints = [p.name for p in directory.glob('cp*.pt')]
        assert len(checkpoints) <= 4
        matrix_files = list((directory/'reconstruction_diagnostics').glob('*column-normalized*.json'))
        assert len(matrix_files) == 1
        matrix = json.loads(matrix_files[0].read_text())
        np.testing.assert_array_equal(matrix['content_style_counts'], np.full((26, 8), 325))
        for ext in ('.svg', '.png', '.csv'): assert matrix_files[0].with_suffix(ext).is_file()
        bn_protocol = json.loads((directory/'bn_diagnostic_protocol.json').read_text())
        reports[arm] = {'initial_model_sha256': initial, 'parameters_sha256': parameters_hash,
                        'normalization': normalization, 'first_epoch_sampler_sha256': order_hash,
                        'bn_pages_sha256': bn_protocol['pages_sha256'], 'config_sha256': file_sha256(config_path),
                        'dataset_manifest_sha256': config['dataset_manifest_sha256'],
                        'checkpoints': checkpoints, 'full_validation_fragments': 67600}
        print('S8_SMOKE_OK', arm, json.dumps(reports[arm]), flush=True)
        checkpoint = torch.load(directory/'cp_current_epoch0.pt', map_location='cpu', weights_only=False)
        restored = get_model(config['dataloader'], config['model_config'])
        restored.load_state_dict(checkpoint['model'], strict=True)
        assert_equal(unittest.TestCase(), trainer.model.state_dict(),
                     {key: value.to(trainer.device) for key, value in restored.state_dict().items()})
        assert restored.active_decoder_style_mode == 'page_mean'
        del restored, checkpoint, trainer; torch.cuda.empty_cache()
        if arm == 'dgn8':
            subprocess.run([sys.executable, 'run_evaluation.py', '--run', str(directory),
                '--active_checkpoint', 'current', '--confusion_mtx', '--test_subset_size', '512',
                '--test_subset_seed', '0', '--test_subset_strategy', 'style_stratified'], check=True)
    for field in ('parameters_sha256', 'first_epoch_sampler_sha256', 'bn_pages_sha256'):
        assert reports['bn'][field] == reports['dgn8'][field], field
    output = ROOT/'logs/diagnostics/20260913_s8_preflight'
    output.mkdir(parents=True, exist_ok=True)
    (output/'summary.json').write_text(json.dumps({'node': socket.gethostname(), 'runs': reports}, indent=2)+'\n')
    print('S8_BOTH_PREFLIGHTS_OK; delete smoke-s8-bn and smoke-s8-dgn8 after visual check', flush=True)


if __name__ == '__main__':
    main()
