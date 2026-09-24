"""Balanced P(code | content) heatmaps, separate from legacy row normalization."""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch


def content_style_counts(content, style, n_content, n_style):
    c = content.detach().long().flatten()
    s = style.detach().to(c.device).long().flatten()
    if c.shape != s.shape or bool(((c < 0) | (c >= n_content) | (s < 0) | (s >= n_style)).any()):
        raise ValueError("Invalid content/style labels")
    return torch.bincount(c*n_style+s, minlength=n_content*n_style).reshape(n_content, n_style)


def validate_balanced_counts(pairs, counts=None):
    pairs = np.asarray(pairs)
    if pairs.ndim != 2 or pairs.size == 0 or np.any(pairs <= 0) or not np.all(pairs == pairs.flat[0]):
        raise ValueError("Every (content, style) must have the same positive sample count")
    if counts is not None and not np.array_equal(np.asarray(counts).sum(0), pairs.sum(1)):
        raise ValueError("Confusion counts and content/style coverage do not match")


def column_probabilities(counts):
    counts = np.asarray(counts, dtype=np.float64)
    if (counts.ndim != 2 or not np.isfinite(counts).all() or np.any(counts < 0)
            or np.any(counts.sum(0) <= 0)):
        raise ValueError("Need finite nonnegative counts with every content represented")
    return counts / counts.sum(0, keepdims=True)


def column_conserving_hundredths(probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or np.any(p < 0) or not np.isfinite(p).all() or not np.allclose(p.sum(0), 1):
        raise ValueError("Expected column-normalized probabilities")
    scaled = p * 100
    units = np.floor(scaled).astype(np.int64)
    for col in range(p.shape[1]):
        remainder = 100 - int(units[:, col].sum())
        order = np.argsort(-(scaled[:, col] - units[:, col]), kind="stable")
        units[order[:remainder], col] += 1
    return units


def text_color(rgb):
    rgb = np.asarray(rgb)[:3]
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb+.055)/1.055)**2.4)
    luminance = float(linear @ [.2126, .7152, .0722])
    return "black" if (luminance+.05)/.05 >= 1.05/(luminance+.05) else "white"


def format_probability(hundredths):
    if int(hundredths) == 0:
        return ""
    return f"{int(hundredths)/100:.2f}".removeprefix("0")


def save_confusion_diagnostics(prefix, counts, pairs, contents, styles, title,
                               metadata=None, code_to_label=None):
    counts = np.asarray(counts)
    pairs = np.asarray(pairs)
    validate_balanced_counts(pairs, counts)
    if counts.shape[1] != len(contents) or pairs.shape != (len(contents), len(styles)):
        raise ValueError("Label names do not match matrix dimensions")
    p = column_probabilities(counts)
    if code_to_label is None:
        codes, labels = linear_sum_assignment(-counts)
        code_to_label = dict(zip(codes.tolist(), labels.tolist()))
    ordered = sorted(code_to_label, key=lambda code: code_to_label[code])
    ordered += [i for i in range(counts.shape[0]) if i not in code_to_label]
    accuracy = sum(float(counts[code, label]) for code, label in code_to_label.items()) / counts.sum()
    hundredths = column_conserving_hundredths(p)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("muted_blue", ["#f6f8fc", "#4979a8"])
    fig, ax = plt.subplots(figsize=(18, max(18, 18*counts.shape[0]/len(contents))))
    ax.imshow(p[ordered], cmap=cmap, vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(len(contents)), contents, fontsize=14)
    ax.set_yticks(range(len(ordered)), [f"q{code:02d}" for code in ordered], fontsize=13)
    ax.set_xlabel("True content label — each column sums to 1", fontsize=15)
    ax.set_ylabel("Code ID (Hungarian row order)", fontsize=15)
    ax.set_title(f"{title}\nOne-to-one mapping rate = {accuracy:.4f} ({accuracy:.2%})\n"
                 f"P(code | content); {int(counts.sum()):,} fragments; "
                 f"{int(pairs.flat[0])} per (letter, style)", fontsize=17, pad=16)
    ax.set_xticks(np.arange(-.5, len(contents), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(ordered), 1), minor=True)
    ax.grid(which="minor", color="#cbd2dc", linewidth=.45)
    ax.tick_params(which="minor", length=0)
    fig.text(.5, .012, "Column-conserving rounding; blank = rounded 0.00 (may be nonzero); CSV/JSON retain exact probabilities.",
             ha="center", fontsize=11)
    fig.tight_layout(rect=(0, .025, 1, 1))
    fig.canvas.draw()
    cell_width = ax.get_window_extent().width / len(contents) * 72 / fig.dpi
    fontsize = cell_width * .85 / 2.4  # four monospace glyphs, width ~0.6em each
    for row, code in enumerate(ordered):
        for col in range(len(contents)):
            label = format_probability(hundredths[code, col])
            if not label:
                continue
            ax.text(col, row, label, ha="center", va="center",
                    fontsize=fontsize, fontfamily="DejaVu Sans Mono", fontweight="normal",
                    color=text_color(cmap(p[code, col])))
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "png"):
        fig.savefig(str(prefix) + "." + suffix, dpi=200)
    plt.close(fig)
    with Path(str(prefix)+".csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["code_id", "hungarian_label", *contents])
        for code in ordered:
            label = code_to_label.get(code)
            writer.writerow([code, contents[label] if label is not None else "unmatched", *p[code]])
    payload = {**(metadata or {}), "normalization": "P(code|content)",
               "content_labels": list(contents), "styles": list(styles),
               "content_style_counts": pairs.tolist(), "raw_counts": counts.tolist(),
               "probabilities": p.tolist(), "row_code_order": ordered,
               "code_to_label": {str(k): int(v) for k, v in code_to_label.items()},
               "display_hundredths": hundredths.tolist(), "one_to_one_accuracy": float(accuracy),
               "fragment_count": int(counts.sum()), "annotation_fontsize": fontsize,
               "annotation_version": 2, "zero_annotations": "hidden after conserving rounding"}
    Path(str(prefix)+".json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload
