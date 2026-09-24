"""Opt-in online usage and state-neutral, balanced BatchNorm diagnostics."""
from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from utils.foreground_color import ForegroundColorAccumulator
from utils.subset_sampling import select_subset_indices


def append_row(path, row):
    exists = path.exists()
    with path.open('a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def usage_metrics(counts):
    counts = torch.as_tensor(counts).detach().double()
    total = counts.sum()
    if total <= 0:
        raise ValueError('Usage diagnostic requires nonempty counts')
    probabilities = counts[counts > 0] / total
    return {'active_codes': int((counts > 0).sum()),
            'usage_perplexity': float((-probabilities.mul(probabilities.log()).sum()).exp()),
            'max_code_usage_fraction': float(probabilities.max()),
            'fragment_count': int(total)}


class TrainingUsageMonitor:
    """Aggregate assignments made by changing training models, not a frozen eval."""
    def __init__(self, root, n_atoms, every_n_steps=100):
        self.root, self.n_atoms = Path(root), int(n_atoms)
        self.every = int(every_n_steps)
        if self.every <= 0:
            raise ValueError('training_usage_monitor.every_n_steps must be positive')
        (self.root / 'training_codebook_usage_protocol.json').write_text(json.dumps({
            'scope': 'online changing training model; no frozen-checkpoint interpretation',
            'every_n_steps': self.every, 'codebook_size': self.n_atoms,
            'window_resets_at_epoch_boundary': True,
        }, indent=2) + '\n')

    def begin(self, epoch, device):
        self.epoch, self.steps = int(epoch), 0
        self.window = torch.zeros(self.n_atoms, dtype=torch.int64, device=device)
        self.total = torch.zeros_like(self.window)

    @torch.no_grad()
    def collect(self, indices, global_step):
        counts = torch.bincount(indices.detach().long().flatten(), minlength=self.n_atoms)
        if len(counts) != self.n_atoms:
            raise ValueError('Out-of-range training VQ index')
        self.window += counts
        self.total += counts
        self.steps += 1
        if self.steps % self.every == 0:
            self._write('window', self.window, global_step)
            self.window.zero_()

    def _write(self, scope, counts, global_step):
        row = {'epoch': self.epoch, 'global_step': int(global_step), 'scope': scope,
               **usage_metrics(counts)}
        append_row(self.root / 'training_codebook_usage.csv', row)
        return row

    def finish(self, global_step):
        if bool(self.window.sum()):
            self._write('window', self.window, global_step)
        row = self._write('epoch', self.total, global_step)
        _plot_csv(self.root, 'training_codebook_usage',
                  ('active_codes', 'usage_perplexity', 'max_code_usage_fraction'),
                  'Online training usage (changing model; not checkpoint evaluation)',
                  group='scope', x='global_step')
        return row


@contextmanager
def preserve_model_and_rng(model):
    """Also restore on failed forwards; do not touch parameters or optimizer."""
    modes = [(module, module.training) for module in model.modules()]
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    py_rng, np_rng = random.getstate(), np.random.get_state()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield buffers
    finally:
        with torch.no_grad():
            for buffer, value in buffers:
                buffer.copy_(value)
        for module, training in modes:
            module.training = training
        random.setstate(py_rng)
        np.random.set_state(np_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


BN_MODES = ('eval', 'encoder_batch_stats', 'decoder_batch_stats', 'both_batch_stats')


@torch.inference_mode()
def evaluate_bn_modes(model, batch, full_validation_counts, style_names, report_normalization_layers=False):
    """A fixed mapping from full validation; never refit mapping on the small batch."""
    device = next(model.parameters()).device
    originals, contents, styles = [value.to(device) for value in batch]
    counts = np.asarray(full_validation_counts)
    if counts.ndim != 2 or counts.sum() <= 0:
        raise ValueError('BN probe requires nonempty full-validation confusion counts')
    code_ids, labels = linear_sum_assignment(-counts)
    mapping = torch.full((counts.shape[0],), -1, dtype=torch.long, device=device)
    mapping[torch.as_tensor(code_ids, device=device)] = torch.as_tensor(labels, device=device)
    results, normal_codes = [], None
    with preserve_model_and_rng(model) as buffers:
        for mode in BN_MODES:
            for buffer, value in buffers:
                buffer.copy_(value)
            model.eval()
            encoder_bn_count = decoder_bn_count = switched_bn_count = 0
            for name, module in model.named_modules():
                if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    continue
                encoder_bn_count += int(name.startswith('encoder.'))
                decoder_bn_count += int(name.startswith('decoder.'))
                if (mode == 'both_batch_stats'
                        or (mode == 'encoder_batch_stats' and name.startswith('encoder.'))
                        or (mode == 'decoder_batch_stats' and name.startswith('decoder.'))):
                    module.train()
                    switched_bn_count += 1
            outputs = model(originals, freeze_codebook=True)
            codes = outputs[3].detach().long()
            if normal_codes is None:
                normal_codes = codes.clone()
            if mode == 'decoder_batch_stats' and not torch.equal(codes, normal_codes):
                raise RuntimeError('Decoder-only BN intervention unexpectedly changed VQ indices')
            color = ForegroundColorAccumulator(style_names)
            color.update(originals, outputs[0], styles)
            color_metrics = color.result()
            row = {'mode': mode,
                   'recon_loss': float(torch.nn.functional.mse_loss(outputs[0], originals)),
                   'fixed_mapping_accuracy': float((mapping[codes] == contents).double().mean()),
                   'code_changed_fraction': float((codes != normal_codes).double().mean()),
                   **usage_metrics(torch.bincount(codes.flatten(), minlength=counts.shape[0]))}
            if report_normalization_layers:
                row.update(encoder_bn_layer_count=encoder_bn_count,
                           decoder_bn_layer_count=decoder_bn_count,
                           switched_bn_layer_count=switched_bn_count)
            row.update({key: value for key, value in color_metrics.items()
                        if key not in ('protocol', 'per_style')})
            for style, metrics in color_metrics['per_style'].items():
                row.update({f'{style}__{key}': value for key, value in metrics.items()})
            results.append(row)
    return results


class BatchNormDiagnostic:
    def __init__(self, root, dataset, config):
        self.root, self.dataset = Path(root), dataset
        self.every = int(config.get('every_n_epochs', 1))
        self.report_normalization_layers = bool(config.get('report_normalization_layers', False))
        per_style = int(config.get('pages_per_style', 4))
        if self.every <= 0 or per_style <= 0:
            raise ValueError('BN diagnostic period and pages_per_style must be positive')
        size = per_style * len(dataset.s_list)
        self.ids, counts = select_subset_indices(dataset, size, config.get('seed', 0), 'style_stratified')
        if len(self.ids) != size or any(n != per_style for n in counts.values()):
            raise ValueError('BN diagnostic requires exactly pages_per_style from every style')
        self.id_set = set(self.ids)
        paths = [str(Path(dataset.png_paths[i]).resolve()) for i in self.ids]
        self.fingerprint = hashlib.sha256(json.dumps(paths).encode()).hexdigest()
        (self.root / 'bn_diagnostic_protocol.json').write_text(json.dumps({
            'split': 'validation', 'pages': paths, 'pages_per_style': per_style,
            'seed': config.get('seed', 0), 'pages_sha256': self.fingerprint,
            'every_n_epochs': self.every, 'modes': BN_MODES,
            'report_normalization_layers': self.report_normalization_layers,
            'mapping': 'normal full-validation Hungarian; shared by all four modes',
            'selection': 'diagnostic only; never checkpoint or objective selection',
        }, indent=2) + '\n')

    def begin(self, epoch, global_step):
        self.epoch, self.global_step = epoch, global_step
        self.enabled_now = epoch % self.every == 0
        self.offset, self.records = 0, {}

    def collect(self, originals, contents, styles):
        if self.enabled_now:
            for local in range(len(originals)):
                page_id = self.offset + local
                if page_id in self.id_set:
                    self.records[page_id] = tuple(value[local].detach().cpu().clone()
                                                  for value in (originals, contents, styles))
        self.offset += len(originals)

    def finish(self, model, counts):
        if not self.enabled_now:
            return []
        if self.offset != len(self.dataset) or set(self.records) != self.id_set:
            raise RuntimeError('BN diagnostic requires complete validation capture')
        batch = tuple(torch.stack([self.records[i][j] for i in self.ids]) for j in range(3))
        n_styles, n_contents = len(self.dataset.s_list), counts.shape[1]
        pairs = torch.bincount((batch[1].long() * n_styles + batch[2].long()).flatten(),
                               minlength=n_contents*n_styles).reshape(n_contents, n_styles)
        if not bool((pairs == len(self.ids) // n_styles).all()):
            raise ValueError('BN probe actual content/style counts are not balanced')
        rows = []
        for result in evaluate_bn_modes(model, batch, counts, self.dataset.s_list,
                                       self.report_normalization_layers):
            row = {'epoch': self.epoch, 'global_step': self.global_step,
                   'split': 'val-balanced-small-probe', 'pages_sha256': self.fingerprint,
                   'page_count': len(self.ids), **result}
            append_row(self.root / 'bn_diagnostic_epoch_history.csv', row)
            rows.append(row)
        self.records.clear()
        _plot_csv(self.root, 'bn_diagnostic_epoch_history',
                  ('recon_loss', 'fixed_mapping_accuracy', 'code_changed_fraction',
                   'usage_perplexity', 'foreground_rgb_mae', 'foreground_chroma_rmse'),
                  'Fixed balanced validation BN intervention (diagnostic only)', group='mode',
                  figure_name='bn_diagnostics.png')
        return rows


def _plot_csv(root, stem, metrics, title, group, x='epoch', figure_name=None):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    with (root / f'{stem}.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    fig, axes = plt.subplots(len(metrics), 1, figsize=(11, 2.6*len(metrics)), sharex=True)
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        for label in dict.fromkeys(row[group] for row in rows):
            selected = [row for row in rows if row[group] == label]
            display_label = label
            if selected[0].get('switched_bn_layer_count') == '0' and label != 'eval':
                display_label += ' (no-op: no matching BN)'
            ax.plot([int(row[x]) for row in selected], [float(row[metric]) for row in selected],
                    label=display_label, linewidth=1.2, marker='.', markersize=4)
        ax.set_ylabel(metric)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        if metric in ('recon_loss', 'foreground_rgb_mae', 'foreground_chroma_rmse'):
            ax.set_yscale('symlog', linthresh=.01)
    np.atleast_1d(axes)[-1].set_xlabel(x)
    fig.suptitle(title)
    fig.tight_layout()
    filename = figure_name or f'{stem}.png'
    temp = root / f'{filename}.tmp.png'
    fig.savefig(temp, dpi=150)
    plt.close(fig)
    temp.replace(root / filename)
