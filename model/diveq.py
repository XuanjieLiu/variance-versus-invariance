"""DiVeQ adapter for CSAE (ordinary stochastic DiVeQ, not SF/Detach).

Adapted from AaltoML/DiVeQ, MIT licensed; see licenses/DiVeQ.txt.
Reference commit: 7e8cc2c3885707576510273da097465949f04063
Reference: vqvae_compression/diveq.py, Eq.8 of arXiv:2509.26469.
Training uses directional reparameterization; evaluation is exact hard VQ.
"""
import logging
import math

import torch
from torch import nn
from torch.nn import functional as F


class _GradientCodebook(nn.Module):
    def __init__(self, size, dim):
        super().__init__()
        # Preserve the native atom state-key/shape used by geometry utilities.
        self.embed = nn.Parameter(torch.rand(1, size, dim) / size)


class DiVeQ(nn.Module):
    def __init__(self, dim, codebook_size, *, noise_std=0.001,
                 replacement_iters=100, discard_threshold=0.01, perturb_eps=1e-9):
        super().__init__()
        if not isinstance(dim, int) or dim < 1 or not isinstance(codebook_size, int) or codebook_size < 2:
            raise ValueError("DiVeQ needs positive dim and codebook_size >= 2")
        for name, value in (("noise_std", noise_std), ("perturb_eps", perturb_eps)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"DiVeQ {name} must be positive and finite")
        if type(replacement_iters) is not int or replacement_iters < 1:
            raise ValueError("DiVeQ replacement_iters must be a positive integer")
        if not math.isfinite(discard_threshold) or not 0 <= discard_threshold <= 1:
            raise ValueError("DiVeQ discard_threshold must be in [0,1]")
        self.dim, self.codebook_size = dim, codebook_size
        # Upstream calls this noise_var, but passes it as Normal's SCALE (std).
        self.noise_std = float(noise_std)
        self.replacement_iters = replacement_iters
        self.discard_threshold = float(discard_threshold)
        self.perturb_eps = float(perturb_eps)
        self.project_in = nn.Identity()
        self.project_out = nn.Identity()
        self._codebook = _GradientCodebook(codebook_size, dim)
        self.register_buffer("codebook_usage", torch.zeros(codebook_size, dtype=torch.long))
        self.register_buffer("iter_counter", torch.zeros((), dtype=torch.long))
        self.register_buffer("replacement_pending", torch.zeros((), dtype=torch.bool))
        self.register_buffer("total_replaced", torch.zeros((), dtype=torch.long))

    @property
    def codebook(self):
        return self._codebook.embed[0]

    def get_codes_from_indices(self, indices):
        return F.embedding(indices.long(), self.codebook)

    def get_output_from_indices(self, indices):
        return self.project_out(self.get_codes_from_indices(indices))

    def forward(self, x, freeze_codebook=False, ema_update_weight=None):
        if ema_update_weight is not None:
            raise ValueError("DiVeQ has no EMA; ema_update_weight is unsupported")
        if x.ndim not in (2, 3) or x.shape[-1] != self.dim or x.numel() == 0:
            raise ValueError(f"DiVeQ expects [N,{self.dim}] or [B,F,{self.dim}]")
        if not x.is_floating_point():
            raise ValueError("DiVeQ requires floating-point input")
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = self.project_in(x)
            z = z if z.dtype == torch.float64 else z.float()
            flat = z.reshape(-1, self.dim)
            atoms = self.codebook.detach() if freeze_codebook else self.codebook
            atoms = atoms.to(z.dtype)
            with torch.no_grad():
                distances = flat.square().sum(-1, keepdim=True) + atoms.square().sum(-1)
                distances = distances - 2 * flat @ atoms.T
                ids = distances.argmin(-1)
            hard = F.embedding(ids, atoms)
            if torch.is_grad_enabled():
                direction = hard - flat
                noisy = direction
                if self.training:
                    noisy = noisy + torch.randn_like(noisy) * self.noise_std
                unit = (noisy / noisy.norm(dim=-1, keepdim=True).clamp_min(1e-12)).detach()
                quantized = flat + direction.norm(dim=-1, keepdim=True) * unit
                if not self.training:
                    # Exact eval forward, while retaining DiVeQ (not STE) gradients
                    # when a downstream caller explicitly requests gradients.
                    quantized = hard.detach() + (quantized - quantized.detach())
            else:
                # Frozen/no-grad diagnostics must not sample noise or mutate VQ.
                quantized = hard
            if self.training and torch.is_grad_enabled() and not freeze_codebook:
                with torch.no_grad():
                    # Official replacement counts batch PRESENCE, not token mass.
                    self.codebook_usage[ids.unique()] += 1
                    self.iter_counter.add_(1)
                    self.replacement_pending.logical_or_(self.iter_counter.remainder(self.replacement_iters) == 0)
            quantized = self.project_out(quantized.reshape_as(z))
        # DiVeQ learns both encoder and atoms from the task objective, no aux loss.
        return quantized, ids.reshape(x.shape[:-1]), z.new_zeros(())

    @torch.no_grad()
    def after_optimizer_step(self):
        """Replacement after backward/step, never inside an active autograd graph.

        Same batch-presence frequency, proportional donor sampling and additive
        jitter as upstream. Optimizer moments are retained as in upstream. No
        replacement is performed during evaluation/frozen forwards. Counters and
        pending state are persistent for checkpoint/resume.
        """
        if not self.training or not bool(self.replacement_pending):
            return
        ratio = self.codebook_usage.float() / self.replacement_iters
        unused = (ratio < self.discard_threshold).nonzero().flatten()
        used = (ratio >= self.discard_threshold).nonzero().flatten()
        if unused.numel() and used.numel():
            probability = self.codebook_usage[used].float()
            donor = used[torch.multinomial(probability / probability.sum(), unused.numel(), replacement=True)]
            values = self.codebook[donor].clone()
            self.codebook[unused] = values + self.perturb_eps * torch.randn_like(values)
            self.total_replaced.add_(unused.numel())
            logging.info("DIVEQ replacement: iteration=%d replaced=%d", int(self.iter_counter), unused.numel())
        self.codebook_usage.zero_()
        self.replacement_pending.zero_()


def build_diveq(config):
    dim = config["d_emb_c"]
    if config.get("vq_codebook_dim", dim) not in (None, dim):
        raise ValueError("This DiVeQ adapter requires native dimension == d_emb_c")
    if "vq_ema_decay" in config or "threshold_ema_dead_code" in config:
        raise ValueError("Remove EMA options from DiVeQ config; use diveq_config instead")
    options = config.get("diveq_config", {})
    return DiVeQ(dim, config["n_atoms"], **options)
