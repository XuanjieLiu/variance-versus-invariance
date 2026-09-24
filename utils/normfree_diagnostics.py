"""Opt-in detached diagnostics; never alter objectives, gradients or selection."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn

from utils.codebook_metrics import compute_assignment_metrics, grouped_mapping_metrics
from utils.training_diagnostics import append_row, _plot_csv


NORM_TYPES = (nn.modules.batchnorm._BatchNorm, nn.GroupNorm, nn.LayerNorm,
              nn.modules.instancenorm._InstanceNorm)


def non_normalization_parameters(model):
    excluded = {id(p) for m in model.modules() if isinstance(m, NORM_TYPES)
                for p in m.parameters(recurse=False)}
    return [(name, p) for name, p in model.named_parameters() if id(p) not in excluded]


@torch.no_grad()
def tensor_scale(x):
    x = x.detach().double()
    return {'rms': float(x.square().mean().sqrt()), 'absmax': float(x.abs().max())}


class OptimizationScaleMonitor:
    HOOKS = ('encoder.cnn_1', 'encoder.cnn_2', 'encoder.linear_1',
             'decoder.linear_0', 'decoder.cnn_transpose_2', 'decoder.cnn_transpose_1')

    def __init__(self, root, model, every_n_steps=100):
        self.root, self.model, self.every = Path(root), model, int(every_n_steps)
        if self.every <= 0:
            raise ValueError('optimization_diagnostics.every_n_steps must be positive')
        self.active, self.activations, self.handles = False, {}, []
        for name in self.HOOKS:
            module = model.get_submodule(name)
            self.handles.append(module.register_forward_hook(self._hook(name)))
        (self.root / 'optimization_scale_protocol.json').write_text(json.dumps({
            'every_n_steps': self.every, 'also_record_last_step_of_epoch': True,
            'hooks': self.HOOKS, 'activation_and_latent_scope': 'pre-optimizer sampled training forward',
            'gradient_scope': 'unscaled gradients retained after GradScaler.step, before zero_grad',
            'epoch_scope': 'arithmetic mean of sampled steps, not every training step',
            'intervention': 'none; no clipping, rescaling or parameter/RNG mutation'}, indent=2)+'\n')

    def _hook(self, name):
        def hook(module, args, output):
            if self.active:
                self.activations[name] = tensor_scale(output)
        return hook

    def begin(self, epoch):
        self.epoch, self.rows = int(epoch), []

    def before_forward(self, global_step, last_step=False):
        self.active = global_step % self.every == 0 or last_step
        self.activations = {}

    @torch.no_grad()
    def collect(self, global_step, zc, ec, zs):
        if not self.active:
            return None
        self.active = False  # never collect validation or diagnostic forwards
        row = {'epoch': self.epoch, 'global_step': int(global_step), 'scope': 'sampled_step'}
        for name, tensor in [('zc', zc), ('ec', ec), ('zs', zs), ('codebook', self.model.vq.codebook)]:
            row.update({f'{name}_{key}': value for key, value in tensor_scale(tensor).items()})
            row[f'{name}_mean_l2'] = float(tensor.detach().double().norm(dim=-1).mean())
        if set(self.activations) != set(self.HOOKS):
            raise RuntimeError('Incomplete activation capture')
        for name in self.HOOKS:
            row.update({f'{name}_{key}': value for key, value in self.activations[name].items()})
        for name in ('encoder', 'decoder'):
            grads = [p.grad.detach() for p in getattr(self.model, name).parameters() if p.grad is not None]
            row[f'{name}_grad_l2'] = float(torch.stack([g.double().square().sum() for g in grads]).sum().sqrt()) if grads else 0.
            row[f'{name}_grad_absmax'] = max((float(g.abs().max()) for g in grads), default=0.)
        row['all_scales_finite'] = int(all(np.isfinite(v) for k,v in row.items() if k != 'scope'))
        append_row(self.root/'optimization_scale_history.csv', row)
        self.rows.append(row)
        self.activations = {}
        return row

    def finish(self, global_step):
        if not self.rows:
            raise RuntimeError('No optimization-scale observations this epoch')
        row = {'epoch': self.epoch, 'global_step': int(global_step), 'scope': 'sampled_epoch_mean',
               'sampled_steps': len(self.rows)}
        row.update({k: float(np.mean([r[k] for r in self.rows])) for k in self.rows[0]
                    if k not in ('epoch', 'global_step', 'scope', 'all_scales_finite')})
        row['all_scales_finite'] = min(r['all_scales_finite'] for r in self.rows)
        append_row(self.root/'optimization_scale_epoch_history.csv', row)
        _plot_csv(self.root, 'optimization_scale_epoch_history',
            ('zc_rms', 'ec_rms', 'zs_rms', 'encoder.cnn_2_rms', 'decoder.cnn_transpose_1_rms',
             'encoder_grad_l2', 'decoder_grad_l2'),
            'Detached optimization scales (sampled training steps)', group='scope',
            figure_name='optimization_scales.png')
        return row

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class PerStyleCodebookMonitor:
    def __init__(self, root, n_codes, content_names, style_names, content_groups=None):
        self.root, self.k = Path(root), int(n_codes)
        self.contents, self.styles = list(content_names), list(style_names)
        self.groups = content_groups or {}
        self.c, self.s = len(self.contents), len(self.styles)
        (self.root/'per_style_codebook').mkdir(exist_ok=True)
        (self.root/'per_style_codebook_protocol.json').write_text(json.dumps({
            'split': 'full validation', 'mapping': 'one shared full-validation Hungarian assignment',
            'probability_scope': 'per-style counts; do not rematch or exclude a style',
            'content_names': self.contents, 'style_names': self.styles,
            'content_groups': self.groups,
            'selection': 'diagnostic only; does not change historical macro gate'}, indent=2)+'\n')

    def begin(self, epoch, global_step, device):
        self.epoch, self.step = int(epoch), int(global_step)
        self.counts = torch.zeros(self.s*self.k*self.c, dtype=torch.int64, device=device)

    @torch.no_grad()
    def collect(self, indices, contents, styles):
        q, y, s = [x.detach().to(device=self.counts.device, dtype=torch.long).flatten()
                   for x in (indices, contents, styles)]
        if not (q.numel() == y.numel() == s.numel()):
            raise ValueError('Per-style diagnostic shape mismatch')
        if not bool(((q>=0)&(q<self.k)&(y>=0)&(y<self.c)&(s>=0)&(s<self.s)).all()):
            raise ValueError('Out-of-range code/content/style')
        self.counts += torch.bincount((s*self.k+q)*self.c+y, minlength=self.counts.numel())

    def finish(self, global_counts):
        counts = self.counts.reshape(self.s,self.k,self.c).cpu().numpy()
        if not np.array_equal(counts.sum(axis=0), global_counts):
            raise ValueError('Per-style/global confusion counts disagree')
        pairs = counts.sum(axis=1)
        if pairs.min() <= 0 or not np.all(pairs == pairs[0,0]):
            raise ValueError('Full validation content/style counts must be strictly balanced')
        codes, labels = linear_sum_assignment(-np.asarray(global_counts))
        rows = []
        for i, name in enumerate(self.styles):
            metrics = compute_assignment_metrics(counts[i])
            usage = counts[i].sum(axis=1)
            row = {'epoch':self.epoch, 'global_step':self.step, 'style':name,
                   'fragment_count':int(usage.sum()),
                   'global_mapping_accuracy':float(counts[i,codes,labels].sum()/usage.sum()),
                   **{k:metrics[k] for k in ('macro_atom_purity','codebook_purity','active_codes',
                                            'usage_perplexity','dominant_label_coverage')},
                   'largest_code_id':int(usage.argmax()),
                   'largest_code_fraction':float(usage.max()/usage.sum())}
            append_row(self.root/'per_style_codebook_epoch_history.csv', row)
            rows.append(row)
        output = self.root/'per_style_codebook'/f'counts_epoch{self.epoch:03d}__val.json'
        grouped = grouped_mapping_metrics(global_counts, self.groups, dict(zip(codes, labels)))
        self.group_rows = []
        for name, values in grouped['groups'].items():
            row = {'epoch': self.epoch, 'global_step': self.step, 'group': name,
                   'accuracy': values['accuracy'], 'fragment_count': values['fragment_count'],
                   'correct_count': values['correct_count']}
            append_row(self.root/'content_group_epoch_history.csv', row)
            self.group_rows.append(row)
        if self.groups:
            _plot_csv(self.root, 'content_group_epoch_history', ('accuracy',),
                      'Content subsets under the same full-validation Hungarian mapping',
                      group='group', figure_name='content_group_metrics.png')
        output.write_text(json.dumps({'epoch':self.epoch,'split':'val-full-balanced',
            'style_names':self.styles,'content_names':self.contents,
            'counts_style_code_content':counts.tolist(),
            'mapping_code_to_label':dict(zip(map(str,codes.tolist()),labels.tolist())),
            'metrics':rows, 'content_group_metrics': grouped}, indent=2)+'\n')
        _plot_csv(self.root, 'per_style_codebook_epoch_history',
                  ('global_mapping_accuracy','macro_atom_purity','codebook_purity',
                   'active_codes','usage_perplexity','largest_code_fraction'),
                  'Full validation per-style codebook (shared global mapping)', group='style',
                  figure_name='per_style_codebook_metrics.png')
        return rows
