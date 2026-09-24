"""GPU-only V2 integrity + full-validation matched diagnostics preflight.

Deliberately keeps two smoke dirs for human image inspection; delete them directly
after acceptance. Never add these runs to the formal ledger.
"""
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import yaml
from trainer import Trainer
from utils.codebook_metrics import file_sha256


class TwoBatches:
    def __init__(self, loader):
        self.loader = loader
    def __len__(self):
        return len(self.loader)
    def __iter__(self):
        return itertools.islice(iter(self.loader), 2)


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone(), torch.cuda.get_rng_state_all()


def assert_rng_equal(a, b):
    assert a[0] == b[0]
    assert a[1][0] == b[1][0] and a[1][2:] == b[1][2:]
    np.testing.assert_array_equal(a[1][1], b[1][1])
    torch.testing.assert_close(a[2], b[2])
    for x, y in zip(a[3], b[3]):
        torch.testing.assert_close(x, y)


def main():
    assert socket.gethostname().startswith('ws-') and torch.cuda.is_available()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    torch.set_num_threads(4)
    data = root.parent / 'data/UppercaseLettersV2'
    manifest = json.loads((data / 'manifest.json').read_text())
    assert len(manifest['entries']) == 26000
    for entry in manifest['entries']:
        assert file_sha256(data / entry['path']) == entry['sha256'], entry['path']
    print('ALL_26000_PNG_HASHES_VERIFIED', flush=True)
    initial_hashes, orders = [], []
    for arm, suffix in [('ctrl', 'control'), ('w100', 'meanwarm100')]:
        config = yaml.safe_load((root / f'configs/uppercase/rq2/v2/cfg_vvi_rq2_v2_c128_k26_{suffix}_seed0.yaml').read_text())
        old = yaml.safe_load((root / f'configs/uppercase/rq2/cfg_vvi_rq2_c128_k26_{suffix}_seed0.yaml').read_text())
        for key in ('model_config', 'loss_config', 'optimizer_config', 'macro_best_health_gate', 'precision', 'batch_size', 'epochs'):
            assert config[key] == old[key], key
        config.update(name=f'smoke-v2-{arm}', debug=True, debug_portion=1, snapshot_every_n_epochs=1)
        assert not (root / 'logs' / config['name']).exists(), 'Refuse smoke overwrite'
        trainer = Trainer(config)
        trainer.prepare_data()
        trainer.build_model()
        directory = Path(trainer.log_dir)
        initial_hashes.append(json.loads((directory / 'initial_model_state.json').read_text())['sha256'])
        before = rng_state()
        order = list(iter(trainer.train_loader.sampler))
        orders.append(hashlib.sha256(np.asarray(order, dtype=np.int64).tobytes()).hexdigest())
        random.setstate(before[0]); np.random.set_state(before[1])
        torch.set_rng_state(before[2]); torch.cuda.set_rng_state_all(before[3])
        if arm == 'w100':
            trainer.model.set_decoder_epoch(99)
            assert trainer.model.active_decoder_style_mode == 'page_mean'
            trainer.model.set_decoder_epoch(100)
            assert trainer.model.active_decoder_style_mode == 'fragment'
            trainer.model.set_decoder_epoch(0)
        monitor = trainer.disentanglement_monitor
        for method in ('begin', 'collect', 'finish'):
            original = getattr(monitor, method)
            def checked(*args, _method=method, _original=original, **kwargs):
                before = rng_state()
                state = ({k: v.detach().clone() for k, v in trainer.model.state_dict().items()}
                         if _method == 'finish' else None)
                result = _original(*args, **kwargs)
                assert_rng_equal(before, rng_state())
                if state is not None:
                    for k, v in state.items():
                        torch.testing.assert_close(v, trainer.model.state_dict()[k], rtol=0, atol=0)
                return result
            setattr(monitor, method, checked)
        trainer.train_loader = TwoBatches(trainer.train_loader)
        trainer.train()
        for filename in ('loss_epoch_history.csv', 'codebook_epoch_history.csv', 'v3_ratio_epoch_history.csv',
                         'disentanglement_probe_history.csv', 'foreground_color_epoch_history.csv'):
            with (directory / filename).open() as handle:
                rows = list(csv.DictReader(handle))
            assert int(rows[-1]['epoch']) == 0
        for filename in ('loss_curves.png', 'codebook_metrics.png', 'v3_ratios.png',
                         'disentanglement_probes.png', 'foreground_color_metrics.png',
                         'cp_current_epoch0.pt', 'cp_snapshot_epoch0.pt'):
            assert (directory / filename).exists(), filename
        assert len(list(directory.glob('cp*.pt'))) <= 3
        matrices = list((directory / 'reconstruction_diagnostics').glob('*column-normalized*.json'))
        assert len(matrices) == 1
        matrix = json.loads(matrices[0].read_text())
        np.testing.assert_array_equal(matrix['content_style_counts'], np.full((26, 8), 325))
        np.testing.assert_allclose(np.asarray(matrix['probabilities']).sum(0), 1)
        np.testing.assert_array_equal(np.asarray(matrix['display_hundredths']).sum(0), 100)
        assert matrix['fragment_count'] == 67600
        for suffix in ('.svg', '.png', '.csv'):
            assert matrices[0].with_suffix(suffix).exists()
        assert list((directory / 'reconstruction_diagnostics').glob('foreground_mask*.png'))
        assert list((directory / 'reconstruction_diagnostics').glob('reconstruction_epoch*.png'))
        print('V2_SMOKE_OK', arm, directory, flush=True)
        del trainer
        torch.cuda.empty_cache()
    assert initial_hashes[0] == initial_hashes[1], initial_hashes
    assert orders[0] == orders[1], orders
    print('MATCHED_INITIAL_MODEL_SHA256', initial_hashes[0], flush=True)
    print('MATCHED_TRAIN_ORDER_SHA256', orders[0], flush=True)


if __name__ == '__main__':
    main()
