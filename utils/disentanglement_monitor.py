"""Detached, page-disjoint probes and fixed validation reconstruction grids.

No model forwards, optimizer updates, test labels, or global RNG draws occur
here. Features and reconstructions are captured from the existing validation.
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch

from run_reconstruction_grid import (
    compute_hungarian_mapping, render_grid, write_mapping_csv,
)


def fixed_page_split(dataset, page_count=512, seed=0):
    """Balance styles and split whole pages, independently of filesystem order."""
    styles = dataset.s_list
    if page_count < 2 * len(styles) or page_count % (2 * len(styles)):
        raise ValueError("probe pages must be a positive multiple of 2 * styles")
    rng = np.random.default_rng(seed)
    fit, score, grid = [], [], []
    per_style = page_count // len(styles)
    for style in styles:
        candidates = sorted(
            (str(Path(path).resolve()), i)
            for i, path in enumerate(dataset.png_paths)
            if Path(path).stem.rsplit("_", 1)[-1] == style
        )
        if len(candidates) < per_style:
            raise ValueError(f"Style {style} has fewer than {per_style} pages")
        grid.append(candidates[0][1])
        order = rng.permutation(len(candidates))[:per_style]
        chosen = [candidates[int(i)][1] for i in order]
        fit.extend(chosen[:per_style // 2])
        score.extend(chosen[per_style // 2:])
    return fit, score, grid


def code_style_probe(fit_codes, fit_styles, score_codes, score_styles, k, n_styles):
    counts = torch.bincount(
        fit_codes.long().flatten() * n_styles + fit_styles.long().flatten(),
        minlength=k * n_styles,
    ).reshape(k, n_styles).double()
    # Fixed Laplace prior; an unseen code predicts the (balanced) fit prior.
    probabilities = (counts + 1) / (counts.sum(1, keepdim=True) + n_styles)
    scores = probabilities[score_codes.long().flatten()]
    truth = score_styles.long().flatten()
    return {
        "ec_vq_style_accuracy": float((scores.argmax(1) == truth).double().mean()),
        "ec_vq_style_cross_entropy": float(-scores.gather(1, truth[:, None]).log().mean()),
        "ec_vq_unseen_code_fraction": float(
            (counts.sum(1)[score_codes.long().flatten()] == 0).double().mean()
        ),
    }


def linear_readouts(fit_x, score_x, targets, pca_components=64, ridge=0.01):
    """Fit once per feature space, with raw and scale-aware diagnostic readouts.

    targets: mapping name -> (fit_labels, score_labels, class_count).
    SVD/centering use fit pages only. Fixed floor and regularization are never
    tuned on the score set. Accuracy measures decodability, not independence.
    """
    x, t = fit_x.double(), score_x.double()
    mean = x.mean(0)
    x, t = x - mean, t - mean
    _, singular, vh = torch.linalg.svd(x, full_matrices=False)
    k = min(pca_components, vh.shape[0])
    scales = (singular[:k] / max(len(x) - 1, 1) ** 0.5).clamp_min(1e-7)
    global_scale = x.square().mean().sqrt().clamp_min(1e-7)
    variants = {
        "raw": (x / global_scale, t / global_scale),
        "whitened": ((x @ vh[:k].T) / scales, (t @ vh[:k].T) / scales),
    }
    result = {}
    for variant, (xx, tt) in variants.items():
        gram = xx.T @ xx / len(xx)
        gram += ridge * torch.eye(gram.shape[0], device=xx.device, dtype=xx.dtype)
        ys, means, widths = [], [], []
        for fit_y, _, n_classes in targets.values():
            y = torch.nn.functional.one_hot(fit_y.long(), n_classes).double()
            means.append(y.mean(0))
            ys.append(y - means[-1])
            widths.append(n_classes)
        weights = torch.linalg.solve(gram, xx.T @ torch.cat(ys, 1) / len(xx))
        outputs = (tt @ weights).split(widths, dim=1)
        for (name, (_, truth, _)), output, prior in zip(targets.items(), outputs, means):
            result[f"{name}_{variant}_accuracy"] = float(
                ((output + prior).argmax(1) == truth).double().mean()
            )
    energy = singular.square()
    result["pc1_variance_fraction"] = float(energy[0] / energy.sum().clamp_min(1e-30))
    result["whitening_floor_components"] = int((scales <= 1e-7).sum())
    return result


def append_csv(path, row):
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


class DisentanglementMonitor:
    def __init__(self, log_dir, dataset, config, n_atoms, content_names):
        self.root = Path(log_dir)
        self.config = config
        self.dataset = dataset
        self.n_atoms = n_atoms
        self.contents = list(content_names)
        self.styles = list(dataset.s_list)
        self.fit_ids, self.score_ids, self.grid_ids = fixed_page_split(
            dataset, config.get("pages", 512), config.get("seed", 0)
        )
        self.probe_ids = set(self.fit_ids + self.score_ids)
        self.grid_set = set(self.grid_ids)
        manifest = {
            "split": "validation", "seed": config.get("seed", 0),
            "protocol": config, "fit_score_unit": "whole page",
            "style_names": self.styles, "content_names": self.contents,
            "fit_pages": [dataset.png_paths[i] for i in self.fit_ids],
            "score_pages": [dataset.png_paths[i] for i in self.score_ids],
            "grid_pages": [dataset.png_paths[i] for i in self.grid_ids],
            "labels_do_not_enter_model_training_or_checkpoint_selection": True,
        }
        (self.root / "disentanglement_probe_protocol.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )

    def begin(self, epoch, global_step, mode):
        self.epoch, self.global_step, self.mode = epoch, global_step, mode
        self.records, self.grid_records, self.stats = {}, {}, []
        self.offset = 0
        every = self.config.get("reconstruction_every_n_epochs", 25)
        self.draw = (epoch + 1) % every == 0 or epoch in self.config.get("extra_grid_epochs", [])

    def collect(self, originals, outputs, ec, zs, codes, content, style, stats):
        for local in range(len(originals)):
            page_id = self.offset + local
            if page_id in self.probe_ids:
                self.records[page_id] = tuple(
                    value[local].detach().cpu()
                    for value in (ec, zs, codes, content, style)
                )
            if self.draw and page_id in self.grid_set:
                self.grid_records[page_id] = tuple(
                    value[local].detach().cpu()
                    for value in (originals, outputs, codes, content)
                )
        self.offset += len(originals)
        self.stats.append({name: float(value) for name, value in stats.items()})

    @torch.inference_mode()
    def finish(self, confusion_counts, device):
        if set(self.records) != self.probe_ids:
            raise RuntimeError("Incomplete validation probe pages; check loader ordering")
        def pack(ids):
            return [torch.cat([self.records[i][j] for i in ids]).to(device) for j in range(5)]
        ec_fit, zs_fit, q_fit, c_fit, s_fit = pack(self.fit_ids)
        ec_score, zs_score, q_score, c_score, s_score = pack(self.score_ids)
        metrics = code_style_probe(q_fit, s_fit, q_score, s_score, self.n_atoms, len(self.styles))
        kwargs = {"pca_components": self.config.get("pca_components", 64),
                  "ridge": self.config.get("ridge", 0.01)}
        for prefix, fit_x, score_x, targets in (
            ("zs", zs_fit, zs_score, {
                "zs_content": (c_fit, c_score, len(self.contents)),
                "zs_style": (s_fit, s_score, len(self.styles)),
            }),
            ("zc_pre_vq", ec_fit, ec_score, {
                "zc_pre_vq_style": (s_fit, s_score, len(self.styles)),
            }),
        ):
            values = linear_readouts(fit_x, score_x, targets, **kwargs)
            for name in ("pc1_variance_fraction", "whitening_floor_components"):
                values[f"{prefix}_{name}"] = values.pop(name)
            metrics.update(values)
        for name in self.stats[0]:
            metrics[name] = float(np.mean([batch[name] for batch in self.stats]))
        row = {"epoch": self.epoch, "completed_epochs": self.epoch + 1,
               "global_step": self.global_step, "decoder_style_mode": self.mode,
               "fit_pages": len(self.fit_ids), "score_pages": len(self.score_ids), **metrics}
        append_csv(self.root / "disentanglement_probe_history.csv", row)
        if self.draw:
            self.save_grid(confusion_counts)
        self.plot()
        self.records.clear()
        self.grid_records.clear()
        return metrics

    def save_grid(self, counts):
        if set(self.grid_records) != self.grid_set:
            raise RuntimeError("Incomplete fixed reconstruction grid")
        originals, outputs, codes, content = [
            torch.stack([self.grid_records[i][j] for i in self.grid_ids]) for j in range(4)
        ]
        code_to_label, _, accuracy = compute_hungarian_mapping(counts)
        directory = self.root / "reconstruction_diagnostics"
        directory.mkdir(exist_ok=True)
        stem = f"reconstruction_epoch{self.epoch:03d}__val-full-map__{self.mode}"
        cells = render_grid(
            directory / f"{stem}.png", originals, outputs, content, codes,
            self.contents, self.styles, code_to_label,
            f"{self.root.name} | epoch {self.epoch} ({self.epoch + 1} completed) | {self.mode}",
            accuracy, mapping_scope="full validation",
        )
        write_mapping_csv(directory / f"{stem}.csv", counts, code_to_label, self.contents)
        periodic = (self.epoch + 1) % self.config.get("reconstruction_every_n_epochs", 25) == 0
        metadata = {
            "epoch": self.epoch, "completed_epochs": self.epoch + 1,
            "decoder_style_mode": self.mode, "mapping_split": "full validation",
            "mapping_fragments": int(np.asarray(counts).sum()),
            "hungarian_accuracy": accuracy, "cells": cells,
            "checkpoint": f"cp_snapshot_epoch{self.epoch}.pt" if periodic else None,
            "grid_pages": [self.dataset.png_paths[i] for i in self.grid_ids],
        }
        (directory / f"{stem}.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def plot(self):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        with (self.root / "disentanglement_probe_history.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        epochs = [int(row["epoch"]) for row in rows]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        groups = (
            ("Style leakage into content (lower is better)", (
                "ec_vq_style_accuracy", "zc_pre_vq_style_raw_accuracy", "zc_pre_vq_style_whitened_accuracy"), 1 / len(self.styles)),
            ("Content leakage into raw style (lower is better)", (
                "zs_content_raw_accuracy", "zs_content_whitened_accuracy"), 1 / len(self.contents)),
            ("Style retained in raw style (higher is better)", (
                "zs_style_raw_accuracy", "zs_style_whitened_accuracy"), 1 / len(self.styles)),
            ("Raw MPD (full-validation batch means)", (
                "content_fragment_mpd", "content_sample_mpd", "style_fragment_mpd", "style_sample_mpd"), None),
        )
        for ax, (title, names, chance) in zip(axes.flat, groups):
            for name in names:
                ax.plot(epochs, [float(row[name]) for row in rows], label=name, linewidth=1)
            if chance is not None:
                ax.axhline(chance, color="gray", linestyle="--", label="chance")
                ax.set_ylim(0, 1.02)
            else:
                ax.set_yscale("symlog", linthresh=1e-5)
            for boundary in self.config.get("switch_epochs", []):
                ax.axvline(boundary, color="black", linestyle=":", linewidth=1)
            ax.set_title(title)
            ax.set_xlabel("epoch (zero-based)")
            ax.legend(fontsize=7)
            ax.grid(alpha=.2)
        fig.suptitle("Detached probes: fixed, page-disjoint validation fit / score")
        fig.tight_layout()
        temp = self.root / "disentanglement_probes.tmp.png"
        fig.savefig(temp, dpi=140)
        plt.close(fig)
        temp.replace(self.root / "disentanglement_probes.png")
