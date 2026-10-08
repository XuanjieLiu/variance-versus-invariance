"""v3_rank: a label-free, per-page native-VQ participation-ratio floor.

This is not a log/SVD matrix rank or a VICReg whitening objective. It adds no
noise, variance floor, normalization layer or trainable parameters.
"""

import math

import torch
import torch.nn.functional as F


RANK_LOG_COLUMNS = (
    "rank_loss", "weighted_rank_loss", "rank_effective_rank", "rank_target",
    "rank_trace", "rank_active_fraction",
)


def v3_method_specs(config):
    """Keep historical V3 names; accept the explicit, enabled v3_rank alias."""
    if config["method"] == "v3_rank":
        if not RankRegularization(config["loss_config"].get("rank_regularization")).enabled:
            raise ValueError("method: v3_rank requires rank_regularization.enabled: true")
        return ["V3", "rank"]
    return config["method"].split("_")


class RankRegularization:
    def __init__(self, config=None):
        config = {} if config is None else config
        if not isinstance(config, dict):
            raise ValueError("rank_regularization must be a mapping")
        unknown = set(config) - {"enabled", "target_rank", "weight", "eps"}
        if unknown:
            raise ValueError(f"Unknown rank_regularization options: {sorted(unknown)}")
        self.enabled = config.get("enabled", False)
        if not isinstance(self.enabled, bool):
            raise ValueError("rank_regularization.enabled must be a boolean")
        self.target_rank = float(config.get("target_rank", 4.0))
        self.weight = float(config.get("weight", 0.01))
        self.eps = float(config.get("eps", 1e-12))
        if not math.isfinite(self.target_rank) or self.target_rank < 1:
            raise ValueError("target_rank must be finite and >= 1")
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("rank regularization weight must be finite and >= 0")
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("rank regularization eps must be finite and > 0")

    def validate_dimensions(self, fragments, native_dim, n_atoms=None):
        if not self.enabled:
            return
        limit = min(int(fragments) - 1, int(native_dim))
        if n_atoms is not None:
            limit = min(limit, int(n_atoms) - 1)
        if self.target_rank > limit:
            raise ValueError(
                f"V3-Rank target_rank={self.target_rank:g} exceeds theoretical "
                f"centered rank bound {limit} (fragments={fragments}, "
                f"native_dim={native_dim}, n_atoms={n_atoms})"
            )

    def __call__(self, native_vq):
        if native_vq is None:
            raise ValueError("V3-Rank requires native_vq from the same model forward")
        if native_vq.ndim != 3 or native_vq.shape[0] == 0 or not native_vq.is_floating_point():
            raise ValueError("native_vq must be a floating [pages, fragments, native_dim] tensor")
        _, fragments, dim = native_vq.shape
        self.validate_dimensions(fragments, dim)
        # Covariance/Gram products must not silently become half precision under
        # training autocast. Preserve float64 for numerical/gradient checks.
        with torch.autocast(device_type=native_vq.device.type, enabled=False):
            q = native_vq if native_vq.dtype == torch.float64 else native_vq.float()
            centered = q - q.mean(dim=1, keepdim=True)
            # C=Q^T Q/(F-1) and G=Q Q^T/(F-1) have the same nonzero spectrum.
            # Prefer F x F for the usual small-page / wide-code configuration.
            if fragments <= dim:
                covariance = centered @ centered.transpose(-1, -2)
            else:
                covariance = centered.transpose(-1, -2) @ centered
            covariance = covariance / (fragments - 1)
            trace = covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
            effective_rank = trace.square() / (covariance.square().sum((-2, -1)) + self.eps)
            page_losses = F.relu(1 - effective_rank / self.target_rank)
            loss = page_losses.mean()
        return {
            "rank_loss": loss,
            "weighted_rank_loss": self.weight * loss,
            "rank_effective_rank": effective_rank.detach().mean(),
            "rank_target": loss.new_tensor(self.target_rank),
            "rank_trace": trace.detach().mean(),
            "rank_active_fraction": (page_losses.detach() > 0).to(loss.dtype).mean(),
        }
