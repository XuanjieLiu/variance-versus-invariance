"""Input-only approximate foreground diagnostics, never a training objective."""
import csv
import json
from pathlib import Path

import torch

PROTOCOL = {"version": 1, "reference": "observed_noisy_input",
            "background": "per-channel median of outer 2px border",
            "border": 2, "threshold": 0.15, "min_pixels": 16,
            "chroma": "RGB minus channel mean (not CIELAB DeltaE)",
            "prediction_clipped": False, "aggregation": "equal weight per valid fragment"}


@torch.inference_mode()
def fragment_color_metrics(originals, predictions):
    if originals.shape != predictions.shape or originals.shape[-3] != 3:
        raise ValueError("Expected matching [...,3,H,W] RGB tensors")
    x = (originals.detach().float().reshape(-1, *originals.shape[-3:]) + 1) * 0.5
    y = (predictions.detach().float().reshape_as(x) + 1) * 0.5
    h, w = x.shape[-2:]
    if h <= 4 or w <= 4:
        raise ValueError("Foreground diagnostics need H,W > 4")
    border = torch.ones((h, w), device=x.device, dtype=torch.bool)
    border[2:-2, 2:-2] = False
    background = x[:, :, border].median(dim=-1).values[:, :, None, None]
    mask = ((x - background).abs().amax(dim=1) > PROTOCOL["threshold"]) & ~border
    counts = mask.sum((1, 2))
    valid = counts >= PROTOCOL["min_pixels"]
    denominator = counts.clamp_min(1) * 3
    diff = y - x
    chroma_diff = diff - diff.mean(dim=1, keepdim=True)
    rgb = (diff.abs() * mask[:, None]).sum((1, 2, 3)) / denominator
    chroma = ((chroma_diff.square() * mask[:, None]).sum((1, 2, 3)) / denominator).sqrt()
    return {"foreground_rgb_mae": rgb, "foreground_chroma_rmse": chroma,
            "foreground_fraction": mask.float().mean((1, 2)),
            "prediction_out_of_range_fraction": ((y < 0) | (y > 1)).float().mean((1, 2, 3)),
            "valid": valid, "mask": mask}


class ForegroundColorAccumulator:
    def __init__(self, styles):
        self.styles = list(styles)
        # count, valid count, RGB sum, chroma sum, mask fraction sum, range fraction sum
        self.sums = torch.zeros((len(styles), 6), dtype=torch.float64)

    @torch.inference_mode()
    def update(self, originals, predictions, styles):
        values = fragment_color_metrics(originals, predictions)
        ids = styles.detach().to(originals.device).long().reshape(-1)
        if ids.numel() != values["valid"].numel() or bool(((ids < 0) | (ids >= len(self.styles))).any()):
            raise ValueError("Invalid foreground style labels")
        valid = values["valid"]
        weights = (torch.ones_like(ids, dtype=torch.float64), valid.double(),
                   torch.where(valid, values["foreground_rgb_mae"], 0).double(),
                   torch.where(valid, values["foreground_chroma_rmse"], 0).double(),
                   values["foreground_fraction"].double(),
                   values["prediction_out_of_range_fraction"].double())
        self.sums += torch.stack([torch.bincount(ids, weights=v, minlength=len(self.styles))
                                  for v in weights], dim=1).cpu()

    def result(self):
        def unpack(row):
            count, valid, rgb, chroma, mask, outside = row.tolist()
            return {"fragment_count": int(count), "foreground_valid_count": int(valid),
                    "foreground_invalid_count": int(count-valid),
                    "foreground_rgb_mae": rgb / valid if valid else float("nan"),
                    "foreground_chroma_rmse": chroma / valid if valid else float("nan"),
                    "foreground_fraction": mask / count if count else float("nan"),
                    "prediction_out_of_range_fraction": outside / count if count else float("nan")}
        return {"protocol": PROTOCOL, **unpack(self.sums.sum(0)),
                "per_style": {style: unpack(self.sums[i]) for i, style in enumerate(self.styles)}}


def log_color_epoch(root, epoch, global_step, mode, result):
    root = Path(root)
    path = root / "foreground_color_epoch_history.csv"
    row = {"epoch": epoch, "completed_epochs": epoch+1, "global_step": global_step,
           "split": "val", "decoder_style_mode": mode}
    row.update({k: v for k, v in result.items() if k not in ("protocol", "per_style")})
    for style, metrics in result["per_style"].items():
        row.update({f"{style}__{key}": value for key, value in metrics.items()})
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    (root / "foreground_color_protocol.json").write_text(json.dumps(PROTOCOL, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    with path.open() as handle:
        history = list(csv.DictReader(handle))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, key in zip(axes.flat, ("foreground_rgb_mae", "foreground_chroma_rmse",
                                  "foreground_fraction", "prediction_out_of_range_fraction")):
        epochs = [int(r["epoch"]) for r in history]
        ax.plot(epochs, [float(r[key]) for r in history], color="black", label="all", linewidth=2)
        for style in result["per_style"]:
            ax.plot(epochs, [float(r[f"{style}__{key}"]) for r in history], label=style, alpha=.7)
        ax.set_title(key)
        ax.set_xlabel("epoch")
        ax.grid(alpha=.2)
        ax.legend(fontsize=7, ncol=3)
        if "fraction" not in key:
            ax.set_yscale("symlog", linthresh=.01)
    fig.suptitle("Observed-input foreground estimates; no clean target; predictions NOT clipped")
    fig.tight_layout()
    temp = root / "foreground_color_metrics.tmp.png"
    fig.savefig(temp, dpi=160)
    plt.close(fig)
    temp.replace(root / "foreground_color_metrics.png")
    return {k: v for k, v in row.items() if isinstance(v, (int, float))}


def save_mask_grid(path, originals, content_labels, content_names, style_names):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    values = fragment_color_metrics(originals, originals)
    masks = values["mask"].reshape(*originals.shape[:2], *originals.shape[-2:]).cpu()
    rgb = ((originals.detach().cpu().float()+1)*.5).permute(0, 1, 3, 4, 2).numpy()
    fig, axes = plt.subplots(len(content_names), len(style_names), figsize=(12, max(10, len(content_names))), squeeze=False)
    for s, style in enumerate(style_names):
        positions = {int(c): i for i, c in enumerate(content_labels[s])}
        for c, name in enumerate(content_names):
            i = positions[c]
            overlay = rgb[s, i].copy()
            mask = masks[s, i].numpy()
            overlay[mask] = overlay[mask] * .65 + [0, .35, 0]
            axes[c, s].imshow(overlay.clip(0, 1))
            axes[c, s].axis("off")
            if c == 0:
                axes[c, s].set_title(style)
            if s == 0:
                axes[c, s].text(-.12, .5, name, transform=axes[c, s].transAxes)
    fig.suptitle("Input-estimated foreground (green); approximate mask, not ground truth")
    fig.tight_layout(rect=(0, 0, 1, .985))
    fig.savefig(path, dpi=140)
    plt.close(fig)
