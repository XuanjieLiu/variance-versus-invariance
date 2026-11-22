"""Scheduled InsNotes validation probes, with recording-disjoint fit/score sets.

Consumes an unshuffled, complete validation traversal; never reads samples or
calls a model. ``final_epoch`` is absolute and zero-based, and can be updated
in the supplied config after construction. Probe accuracy measures linear
decodability, not a guarantee of disentanglement or OOD generalization.
"""

import csv
import json
import operator
from pathlib import Path
import re

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from utils.codebook_metrics import batch_confusion_counts, compute_assignment_metrics
from utils.confusion_diagnostics import content_style_counts, validate_balanced_counts
from utils.disentanglement_monitor import code_style_probe, linear_readouts


_STAT_NAMES = (
    "content_fragment_mpd", "content_sample_mpd", "style_fragment_mpd",
    "style_sample_mpd", "content_hinge_active", "style_hinge_active",
    "sample_hinge_active", "fragment_hinge_active",
)


def _recording_partition(dataset, n_styles):
    if getattr(dataset, "training", False) or getattr(dataset, "split", "val") != "val":
        raise ValueError("Audio diagnostics require deterministic validation data")
    groups = operator.index(dataset.groups_per_take)
    if groups != 10 or dataset.n_fragments != 12:
        raise ValueError("Expected ten nonoverlapping 12-note groups per recording")
    paths = [str(Path(path).resolve()) for path in dataset.npy_paths]
    if paths != sorted(paths) or len(set(paths)) != len(paths):
        raise ValueError("npy_paths must contain unique, sorted recording paths")
    styles = []
    for path in paths:
        match = re.fullmatch(r"ins([0-9]+)_([0-9]+)\.npy", Path(path).name)
        if match is None or not 0 <= int(match.group(1)) < n_styles:
            raise ValueError(f"Invalid InsNotes recording name: {path}")
        styles.append(int(match.group(1)))

    explicit_items = getattr(dataset, "eval_items", None)
    if explicit_items is not None:
        items = [(operator.index(file_id), operator.index(group_id))
                 for file_id, group_id in explicit_items]
    elif callable(getattr(dataset, "recording_index", None)):
        # A recording-index-only loader supplies groups in visit order.
        visits = [0] * len(paths)
        items = []
        for index in range(len(dataset)):
            file_id = operator.index(dataset.recording_index(index))
            if not 0 <= file_id < len(paths):
                raise ValueError("Invalid evaluation recording index")
            items.append((file_id, visits[file_id]))
            visits[file_id] += 1
    else:
        raise ValueError("Dataset must expose eval_items or recording_index(index)")
    expected = {(file_id, group_id) for file_id in range(len(paths))
                for group_id in range(groups)}
    if len(items) != len(dataset) or len(items) != len(expected) or set(items) != expected:
        raise ValueError("Evaluation items must cover every recording/group exactly once")

    fit_files, score_files, grid_files = set(), set(), []
    for style in range(n_styles):
        files = [i for i, value in enumerate(styles) if value == style]
        if len(files) != 10:
            raise ValueError("Audio diagnostics require exactly ten validation recordings per style")
        fit_files.update(files[:5])
        score_files.update(files[5:])
        grid_files.append(files[5])
    fit_ids = [i for i, (file_id, _) in enumerate(items) if file_id in fit_files]
    score_ids = [i for i, (file_id, _) in enumerate(items) if file_id in score_files]
    item_ids = {item: i for i, item in enumerate(items)}
    grid_ids = [item_ids[(file_id, 0)] for file_id in grid_files]
    return paths, items, styles, fit_ids, score_ids, grid_ids


class AudioDiagnostics:
    def __init__(self, log_dir, dataset, config, n_atoms, content_names):
        self.root = Path(log_dir)
        self.config = config
        self.dataset = dataset
        self.contents = list(content_names)
        self.styles = list(dataset.s_list)
        self.n_atoms = int(n_atoms)
        if (self.n_atoms, len(self.contents), len(self.styles)) != (12, 12, 12):
            raise ValueError("InsNotes audio diagnostics require 12 codes, pitches and styles")
        self.every = operator.index(config.get("every_n_epochs", 25))
        self.pca_components = operator.index(config.get("pca_components", 64))
        self.ridge = float(config.get("ridge", 0.01))
        if self.every <= 0 or self.pca_components <= 0 or not np.isfinite(self.ridge) or self.ridge <= 0:
            raise ValueError("Diagnostic period, PCA width and ridge must be positive")
        self.plot_content_ids = list(config.get("plot_content_ids", [0, 3, 6, 9]))
        if (not 1 <= len(self.plot_content_ids) <= 4
                or len(set(self.plot_content_ids)) != len(self.plot_content_ids)
                or any(not isinstance(i, int) or not 0 <= i < 12 for i in self.plot_content_ids)):
            raise ValueError("plot_content_ids must contain one to four distinct pitch IDs")
        (self.paths, self.items, self.recording_styles, self.fit_ids,
         self.score_ids, self.grid_ids) = _recording_partition(dataset, len(self.styles))
        self.grid_set = set(self.grid_ids)
        self.records, self.grid_records = {}, {}
        self.active = False
        self.protocol = {
            "split": "validation", "fit_score_unit": "whole recording",
            "fit_rule": "first five sorted recordings per style",
            "score_rule": "last five sorted recordings per style",
            "fit_recordings": self._paths_for(self.fit_ids),
            "score_recordings": self._paths_for(self.score_ids),
            "groups_per_recording": 10, "notes_per_group": 12,
            "expected_group_count": len(self.items),
            "expected_fragment_count": len(self.items) * 12,
            "expected_count_per_pitch_style": 100,
            "evaluation_items": [list(item) for item in self.items],
            "recording_paths": self.paths,
            "grid_items": [{"dataset_index": i, "recording": self.paths[self.items[i][0]],
                            "group_index": self.items[i][1]} for i in self.grid_ids],
            "content_names": self.contents, "style_names": self.styles,
            "dataset_manifest_sha256": getattr(dataset, "manifest_sha256", None),
            "every_n_epochs": self.every,
            "schedule": "epoch % every_n_epochs == 0 or epoch == final_epoch (zero-based)",
            "probes": "raw/whitened ridge pitch and style readouts from z_c and raw z_s; code-to-style probe",
            "probe_fit": "centering, SVD, scale and readouts fit on fit recordings only",
            "pca_components": self.pca_components, "ridge": self.ridge,
            "statistics": "sample-weighted batch means of raw mean-pairwise distances; not global MPD",
            "spectrogram_scale": "unchanged input feature values; no RGB scaling or exponentiation",
            "interpretation": "decodability diagnostics, not disentanglement or OOD guarantees",
            "selection": "diagnostic only; no model updates or checkpoint selection",
        }
        if config.get("enabled", False):
            self.root.mkdir(parents=True, exist_ok=True)
            self._write_json(self.root / "audio_diagnostic_protocol.json", self.protocol)
        self.playback = None
        playback_config = config.get("playback", {})
        if playback_config.get("enabled", False):
            from utils.audio_playback import AudioPlayback
            if not config.get("enabled", False):
                raise ValueError("Audio playback requires audio diagnostics")
            self.playback = AudioPlayback(self.root, dataset, self.protocol, playback_config)
            if self.playback.every % self.every:
                raise ValueError("Playback interval must be a multiple of audio diagnostic interval")

    def _paths_for(self, ids):
        return [self.paths[i] for i in sorted({self.items[index][0] for index in ids})]

    @staticmethod
    def _write_json(path, payload):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    def begin(self, epoch, global_step, mode):
        self.epoch, self.global_step, self.mode = int(epoch), int(global_step), str(mode)
        final_epoch = self.config.get("final_epoch")
        if final_epoch is not None:
            final_epoch = operator.index(final_epoch)
            if final_epoch < 0:
                raise ValueError("final_epoch must be an absolute zero-based epoch")
        self.active = bool(self.config.get("enabled", False) and epoch >= 0
                           and (epoch % self.every == 0 or epoch == final_epoch))
        self.records.clear()
        self.grid_records.clear()
        self.offset = 0
        if not self.active:
            return
        self.counts = torch.zeros((12, 12), dtype=torch.int64)
        self.pairs = torch.zeros((12, 12), dtype=torch.int64)
        self.recon_sse, self.recon_elements = 0.0, 0
        self.stat_sums = {name: 0.0 for name in _STAT_NAMES}
        self.stat_weights = {name: 0 for name in _STAT_NAMES}

    @torch.no_grad()
    def collect(self, originals, outputs, ec, zs, codes, content, style, stats):
        if not self.active:
            return
        if originals.ndim != 4 or originals.shape != outputs.shape or originals.shape[1] != 12:
            raise ValueError("Expected matching [groups, 12, mel_bins, frames] spectrograms")
        batch_size = len(originals)
        if batch_size == 0 or self.offset + batch_size > len(self.items):
            raise ValueError("Validation traversal exceeds the full evaluation item list")
        if (ec.ndim != 3 or zs.ndim != 3 or ec.shape[:2] != originals.shape[:2]
                or zs.shape[:2] != originals.shape[:2] or min(ec.shape[-1], zs.shape[-1]) < 1):
            raise ValueError("Embeddings must preserve the group and note axes")
        if any(value.shape != originals.shape[:2] for value in (codes, content, style)):
            raise ValueError("Code and label shapes must match the group and note axes")
        if any(not bool(torch.isfinite(value).all()) for value in (originals, outputs, ec, zs)):
            raise ValueError("Nonfinite audio diagnostic input")
        q, c, s = [value.detach().cpu().clone() for value in (codes, content, style)]
        if not bool(((q >= 0) & (q < 12) & (q == q.long())).all()):
            raise ValueError("Code IDs must be integers in [0, 12)")
        expected_styles = torch.tensor([
            self.recording_styles[self.items[i][0]]
            for i in range(self.offset, self.offset + batch_size)
        ])[:, None].expand(batch_size, 12)
        if not torch.equal(s, expected_styles):
            raise ValueError("Style labels disagree with evaluation item order")
        if not torch.equal(c, torch.arange(12).expand(batch_size, 12)):
            raise ValueError("Each evaluation group must contain ordered pitches 0..11 exactly once")
        self.counts += batch_confusion_counts(q, c, 12, 12)
        self.pairs += content_style_counts(c, s, 12, 12)
        difference = outputs.detach().double() - originals.detach().double()
        self.recon_sse += float(difference.square().sum())
        self.recon_elements += difference.numel()
        ec_cpu, zs_cpu = ec.detach().cpu(), zs.detach().cpu()
        for local in range(batch_size):
            index = self.offset + local
            self.records[index] = tuple(value[local].clone() for value in (ec_cpu, zs_cpu, q, c, s))
            if index in self.grid_set:
                self.grid_records[index] = tuple(value[local].detach().cpu().clone()
                                                for value in (originals, outputs, q, c))
        self.offset += batch_size
        for name in _STAT_NAMES:
            if stats is not None and name in stats:
                value = float(stats[name].detach() if torch.is_tensor(stats[name]) else stats[name])
                if not np.isfinite(value):
                    raise ValueError(f"Nonfinite diagnostic statistic: {name}")
                self.stat_sums[name] += value * batch_size
                self.stat_weights[name] += batch_size

    @torch.inference_mode()
    def finish(self, confusion_counts, device):
        if not self.active:
            return {}
        try:
            if self.offset != len(self.items) or len(self.records) != len(self.items):
                raise RuntimeError("Audio diagnostics require the complete validation traversal")
            if torch.is_tensor(confusion_counts):
                confusion_counts = confusion_counts.detach().cpu().numpy()
            counts = np.asarray(confusion_counts)
            if counts.shape != (12, 12) or not np.array_equal(counts, self.counts.numpy()):
                raise ValueError("Supplied confusion counts disagree with captured validation")
            validate_balanced_counts(self.pairs.numpy(), counts)
            if not bool((self.pairs == 100).all()):
                raise ValueError("Expected exactly 100 notes per pitch/style pair")
            if set(self.grid_records) != self.grid_set:
                raise RuntimeError("Incomplete fixed audio reconstruction examples")

            def pack(ids):
                return [torch.cat([self.records[i][j] for i in ids]).to(device) for j in range(5)]

            ec_fit, zs_fit, q_fit, c_fit, s_fit = pack(self.fit_ids)
            ec_score, zs_score, q_score, c_score, s_score = pack(self.score_ids)
            metrics = code_style_probe(q_fit, s_fit, q_score, s_score, 12, 12)
            for prefix, fit, score in (("zc_pre_vq", ec_fit, ec_score), ("zs", zs_fit, zs_score)):
                readouts = linear_readouts(fit, score, {
                    f"{prefix}_content": (c_fit, c_score, 12),
                    f"{prefix}_style": (s_fit, s_score, 12),
                }, pca_components=self.pca_components, ridge=self.ridge)
                for name in ("pc1_variance_fraction", "whitening_floor_components"):
                    readouts[f"{prefix}_{name}"] = readouts.pop(name)
                metrics.update(readouts)
            metrics.update({name: value for name, value in compute_assignment_metrics(counts).items()
                            if isinstance(value, (int, float))})
            metrics["recon_loss"] = self.recon_sse / self.recon_elements
            metrics.update({name: self.stat_sums[name] / self.stat_weights[name]
                            if self.stat_weights[name] else None for name in _STAT_NAMES})
            row = {"epoch": self.epoch, "completed_epochs": self.epoch + 1,
                   "global_step": self.global_step, "decoder_style_mode": self.mode,
                   "recording_count": len(self.paths), "group_count": self.offset,
                   "fragment_count": int(counts.sum()), "fit_recordings": 60,
                   "score_recordings": 60, "fit_fragments": len(q_fit),
                   "score_fragments": len(q_score), **metrics}
            self._write_artifacts(counts, row)
            history = self.root / "audio_diagnostic_history.csv"
            if history.exists():
                with history.open(newline="") as handle:
                    if next(csv.reader(handle), None) != list(row):
                        raise ValueError("Existing audio diagnostic CSV has an incompatible schema")
            with history.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if handle.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
            self._plot_curves(history)
            return metrics
        finally:
            self.active = False
            self.records.clear()
            self.grid_records.clear()

    def _write_artifacts(self, counts, row):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        directory = self.root / "audio_diagnostics" / f"epoch{self.epoch:05d}"
        directory.mkdir(parents=True, exist_ok=True)
        code_ids, labels = linear_sum_assignment(-counts)
        mapping = {int(code): int(label) for code, label in zip(code_ids, labels)}
        order = sorted(mapping, key=mapping.get)
        probabilities = counts / counts.sum(0, keepdims=True)
        self._write_json(directory / "metrics.json", {
            **row, "final_epoch": self.config.get("final_epoch"),
            "raw_counts": counts.tolist(), "content_style_counts": self.pairs.tolist(),
            "probabilities": probabilities.tolist(), "normalization": "P(code|pitch)",
            "code_to_label": mapping, "row_code_order": order,
            "mapping_scope": "one Hungarian assignment from full validation",
            "grid_items": self.protocol["grid_items"],
            "interpretation": self.protocol["interpretation"],
        })
        with (directory / "confusion.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["code_id", "hungarian_pitch", *self.contents])
            for code in order:
                writer.writerow([code, self.contents[mapping[code]], *probabilities[code]])
        fig, ax = plt.subplots(figsize=(9, 8))
        try:
            image = ax.imshow(probabilities[order], vmin=0, vmax=1, cmap="Blues")
            ax.set_xticks(range(12), self.contents)
            ax.set_yticks(range(12), [f"q{q:02d}: {self.contents[mapping[q]]}" for q in order])
            for y, code in enumerate(order):
                for x, probability in enumerate(probabilities[code]):
                    if probability > 0:
                        ax.text(x, y, f"{probability:.2f}", ha="center", va="center", fontsize=8,
                                color="white" if probability > 0.5 else "black")
            ax.set(xlabel="True pitch", ylabel="Code (global Hungarian order)",
                   title=f"Epoch {self.epoch} | full validation | P(code | pitch)\n"
                         f"14,400 notes; 100 per pitch/style | accuracy {row['one_to_one_accuracy']:.4f}")
            fig.colorbar(image, ax=ax, fraction=0.04)
            fig.tight_layout()
            fig.savefig(directory / "confusion.png", dpi=140)
        finally:
            plt.close(fig)

        originals, outputs, codes, content = [
            torch.stack([self.grid_records[i][j] for i in self.grid_ids]).numpy()
            for j in range(4)
        ]
        np.savez_compressed(directory / "reconstructions.npz", originals=originals,
                            outputs=outputs, codes=codes, content=content,
                            dataset_indices=np.asarray(self.grid_ids))
        if self.playback is not None and self.playback.due(self.epoch, self.config.get("final_epoch")):
            self.playback.write(f"epoch{self.epoch:05d}", originals, outputs, codes,
                                epoch=self.epoch, mode=self.mode)
        for style, name in enumerate(self.styles):
            fig, axes = plt.subplots(2, len(self.plot_content_ids), squeeze=False,
                                     figsize=(3.2 * len(self.plot_content_ids), 5.2),
                                     constrained_layout=True)
            try:
                # Fixed original-derived scale, shared with unclamped reconstructions.
                vmin, vmax = float(originals[style].min()), float(originals[style].max())
                vmax = max(vmax, vmin + 1e-6)
                for column, pitch in enumerate(self.plot_content_ids):
                    note = int(np.flatnonzero(content[style] == pitch)[0])
                    code = int(codes[style, note])
                    for r, values in enumerate((originals, outputs)):
                        image = axes[r, column].imshow(values[style, note], origin="lower",
                            aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
                        axes[r, column].set_xlabel("Frame")
                    axes[0, column].set_title(self.contents[pitch])
                    axes[1, column].set_title(f"q{code:02d} -> {self.contents[mapping[code]]}")
                axes[0, 0].set_ylabel("Original\nMel bin")
                axes[1, 0].set_ylabel("Reconstruction\nMel bin")
                fig.colorbar(image, ax=axes.ravel().tolist(), label="Stored log-mel value", fraction=0.025)
                path = Path(self.paths[self.items[self.grid_ids[style]][0]]).name
                fig.suptitle(f"{name} | {path} | group 0 | epoch {self.epoch}\n"
                             "Shared original scale; out-of-range colors saturate; archive retains raw values",
                             fontsize=10)
                fig.savefig(directory / f"reconstruction_style{style:02d}.png", dpi=120)
            finally:
                plt.close(fig)

    @torch.inference_mode()
    def export_best_validation_audio(self, training_config, device):
        """A fresh model, immutable final grid, and restored RNG; no trainer state mutation."""
        if self.playback is None or not self.playback.config.get("export_best_validation", True):
            return
        from model.factory import get_model
        from utils.audio_playback import preserve_audio_rng
        from utils.codebook_metrics import file_sha256

        metadata = json.loads((self.root / "best_validation_loss.json").read_text())
        checkpoint = self.root / metadata["checkpoint"]
        final_epoch = self.config["final_epoch"]
        with np.load(self.root / "audio_diagnostics" / f"epoch{final_epoch:05d}" / "reconstructions.npz") as grid:
            originals = grid["originals"].copy()
            if not np.array_equal(grid["dataset_indices"], self.grid_ids):
                raise RuntimeError("Best audio grid does not match the fixed validation protocol")
        with preserve_audio_rng():
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if int(state["epoch"]) != metadata["epoch"]:
                raise RuntimeError("Best checkpoint metadata disagrees with checkpoint epoch")
            model = get_model(training_config["dataloader"], training_config["model_config"]).to(device).eval()
            model.load_state_dict(state["model"], strict=True)
            model.requires_grad_(False)
            values = model(torch.from_numpy(originals).to(device), freeze_codebook=True)
            path = self.playback.write("best_validation", originals, values[0].cpu().numpy(),
                values[3].cpu().numpy(), epoch=int(state["epoch"]), mode=model.active_decoder_style_mode,
                checkpoint={**metadata, "sha256": file_sha256(checkpoint)})
        return path

    def _plot_curves(self, history):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        with history.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        epochs = [int(row["epoch"]) for row in rows]
        groups = (
            ("Style decodability from content", ("ec_vq_style_accuracy",
                "zc_pre_vq_style_raw_accuracy", "zc_pre_vq_style_whitened_accuracy")),
            ("Pitch decodability from raw style", ("zs_content_raw_accuracy", "zs_content_whitened_accuracy")),
            ("Style decodability from raw style", ("zs_style_raw_accuracy", "zs_style_whitened_accuracy")),
            ("Pitch decodability from content", ("one_to_one_accuracy",
                "zc_pre_vq_content_raw_accuracy", "zc_pre_vq_content_whitened_accuracy")),
            ("Reconstruction MSE", ("recon_loss",)),
            ("Raw MPD (sample-weighted batch means)", _STAT_NAMES[:4]),
        )
        fig, axes = plt.subplots(3, 2, figsize=(13, 11))
        try:
            for index, (ax, (title, names)) in enumerate(zip(axes.flat, groups)):
                for name in names:
                    values = [float(row[name]) if row[name] else float("nan") for row in rows]
                    if np.isfinite(values).any():
                        ax.plot(epochs, values, label=name, linewidth=1)
                if index < 4:
                    ax.axhline(1 / 12, color="gray", linestyle="--", label="balanced chance")
                    ax.set_ylim(0, 1.02)
                else:
                    ax.set_yscale("symlog", linthresh=1e-5)
                ax.set(title=title, xlabel="Epoch (zero-based)")
                if ax.lines:
                    ax.legend(fontsize=7)
                ax.grid(alpha=0.2)
            fig.suptitle("Recording-disjoint validation probes: decodability, not disentanglement guarantees")
            fig.tight_layout()
            temporary = self.root / "audio_diagnostics.tmp.png"
            fig.savefig(temporary, dpi=140)
            temporary.replace(self.root / "audio_diagnostics.png")
        finally:
            plt.close(fig)
