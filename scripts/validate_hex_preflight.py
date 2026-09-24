"""Compare matched smoke provenance and inspect final grid layout on a GPU node."""
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import yaml
from dataloader.hex_digits_dataloader import HexDigitsDataset, C_LIST, S_LIST
from model.factory import get_model
from run_reconstruction_grid import render_grid, compute_hungarian_mapping
from utils.codebook_metrics import file_sha256


def main():
    assert socket.gethostname().startswith('ws-') and torch.cuda.is_available()
    os.chdir(ROOT); torch.set_num_threads(4)
    output = ROOT / 'logs/diagnostics/20260915_hex_preflight'
    a, b = [json.loads((output / f'{arm}.json').read_text()) for arm in ('repo', 'mean')]
    for key in ('initial_model', 'normalization', 'first_epoch_sampler_sha256',
                'bn_pages_sha256', 'dataset_manifest_sha256', 'training_steps',
                'validation_pages', 'validation_fragments'):
        assert a[key] == b[key], key
    for arm in ('repo', 'mean'):
        root = ROOT / f'logs/smoke-hex-s1-{arm}'
        config = yaml.safe_load((root / 'config.yaml').read_text())
        mode = config['model_config']['decoder_style_mode']
        diagnostic = root / 'reconstruction_diagnostics'
        meta = json.loads((diagnostic / f'reconstruction_epoch000__val-full-map__{mode}.json').read_text())
        counts = json.loads((diagnostic / f'codebook_confusion_matrix__epoch000__val-full-balanced-column-normalized__{mode}.json').read_text())
        mapping, _, accuracy = compute_hungarian_mapping(counts['raw_counts'])
        dataset = HexDigitsDataset(Path(config['data_dir']) / 'val')
        indices = {p: i for i, p in enumerate(dataset.png_paths)}
        items = [dataset[indices[p]] for p in meta['grid_pages']]
        x = torch.stack([item[0] for item in items]).cuda()
        labels = torch.stack([item[1] for item in items])
        model = get_model(config['dataloader'], config['model_config']).cuda().eval()
        checkpoint = torch.load(root / 'cp_current_epoch0.pt', map_location='cuda', weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=True)
        model.set_decoder_epoch(0)
        with torch.inference_mode(): result = model(x, freeze_codebook=True)
        cells = render_grid(output / f'{arm}_layout_check.png', x.cpu(), result[0].cpu(), labels,
            result[3].cpu(), C_LIST, S_LIST, mapping,
            f'Hex {arm} preflight layout check | epoch 0 | {mode}', accuracy,
            mapping_scope='cached full validation (10,000 pages)')
        assert len(cells) == 128
        assert [c['code_index'] for c in cells] == [c['code_index'] for c in meta['cells']]
        del result, model, checkpoint, x
        torch.cuda.empty_cache()
    (output / 'matched_validation.json').write_text(json.dumps({
        'node': socket.gethostname(), 'matched_initial_model_and_order': True,
        'validation_pages': 10000, 'validation_fragments': 160000,
        'grid_layout_source_sha256': file_sha256(ROOT / 'run_reconstruction_grid.py'),
        'mapping_scope': 'cached complete validation; no subset rematching',
        'forward_pages_per_arm_for_layout_only': 8,
    }, indent=2) + '\n')
    print('HEX_MATCHED_PREFLIGHT_OK', flush=True)


if __name__ == '__main__': main()
