"""Render a content-by-style reconstruction grid with Hungarian code labels."""

import argparse
import csv
import datetime
import json
from importlib import import_module
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import yaml

from model.factory import get_model
from run_codebook_health import build_loader
from utils.codebook_metrics import (
    batch_confusion_counts,
    compute_assignment_metrics,
    file_sha256,
)
from utils.evaluation_paths import resolve_evaluation_paths


def compute_hungarian_mapping(confusion_counts):
    """Return code-to-label and label-to-code mappings for maximum count."""
    counts = np.asarray(confusion_counts, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError(f"Expected a 2D confusion matrix, got {counts.shape}.")
    row_ind, col_ind = linear_sum_assignment(-counts)
    code_to_label = {int(code): int(label) for code, label in zip(row_ind, col_ind)}
    label_to_code = {label: code for code, label in code_to_label.items()}
    accuracy = float(counts[row_ind, col_ind].sum() / counts.sum())
    return code_to_label, label_to_code, accuracy


def select_one_sample_per_style(dataset, style_names):
    """Select the lexicographically first PNG for every configured style."""
    if not hasattr(dataset, "png_paths"):
        raise TypeError("Style-grid selection currently requires a PNG dataset.")

    selected = {}
    indexed_paths = sorted(
        enumerate(dataset.png_paths), key=lambda item: Path(item[1]).name
    )
    style_set = set(style_names)
    for dataset_index, path in indexed_paths:
        try:
            _, style = Path(path).stem.rsplit("_", 1)
        except ValueError:
            continue
        if style in style_set and style not in selected:
            selected[style] = (dataset_index, Path(path).resolve())
        if len(selected) == len(style_names):
            break

    missing = [style for style in style_names if style not in selected]
    if missing:
        raise ValueError(f"No test sample found for styles: {missing}")
    return [selected[style] for style in style_names]


def image_for_plot(tensor):
    image = tensor.detach().float().cpu().clamp(-1, 1)
    image = image.mul(0.5).add(0.5).permute(1, 2, 0).numpy()
    return np.clip(image, 0.0, 1.0)


def write_mapping_csv(path, counts, code_to_label, content_names):
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=(
                "code_index",
                "mapped_label_index",
                "mapped_label",
                "matched_count",
                "code_usage",
                "conditional_accuracy",
            ),
        )
        writer.writeheader()
        for code_index in sorted(code_to_label):
            label_index = code_to_label[code_index]
            usage = int(counts[code_index].sum())
            matched = int(counts[code_index, label_index])
            writer.writerow(
                {
                    "code_index": code_index,
                    "mapped_label_index": label_index,
                    "mapped_label": content_names[label_index],
                    "matched_count": matched,
                    "code_usage": usage,
                    "conditional_accuracy": matched / usage if usage else 0.0,
                }
            )


def render_grid(
    output_path,
    originals,
    reconstructions,
    content_indices,
    vq_indices,
    content_names,
    style_names,
    code_to_label,
    checkpoint_name,
    hungarian_accuracy,
    mapping_scope="full test",
):
    n_contents = len(content_names)
    n_styles = len(style_names)
    fig, axes = plt.subplots(
        n_contents,
        n_styles * 2,
        figsize=(n_styles * 2.0, n_contents * 1.15),
        squeeze=False,
    )
    cell_records = []

    for style_index, style_name in enumerate(style_names):
        positions = {
            int(label): position
            for position, label in enumerate(content_indices[style_index].tolist())
        }
        missing = [index for index in range(n_contents) if index not in positions]
        if missing:
            raise ValueError(
                f"Selected {style_name} sample lacks content indices: {missing}"
            )

        for label_index, true_label in enumerate(content_names):
            position = positions[label_index]
            code_index = int(vq_indices[style_index, position])
            mapped_index = code_to_label.get(code_index)
            mapped_label = (
                content_names[mapped_index] if mapped_index is not None else "unmatched"
            )
            correct = mapped_index == label_index
            original = originals[style_index, position]
            reconstruction = reconstructions[style_index, position]
            reconstruction_mse = float(
                torch.nn.functional.mse_loss(reconstruction, original).item()
            )

            original_axis = axes[label_index, style_index * 2]
            recon_axis = axes[label_index, style_index * 2 + 1]
            original_axis.imshow(image_for_plot(original))
            recon_axis.imshow(image_for_plot(reconstruction))
            original_axis.axis("off")
            recon_axis.axis("off")
            recon_axis.text(
                0.02,
                0.98,
                f"q{code_index:02d}->{mapped_label}",
                transform=recon_axis.transAxes,
                ha="left",
                va="top",
                fontsize=5.5,
                color="lime" if correct else "#ff4040",
                bbox={"facecolor": "black", "alpha": 0.72, "pad": 1.0},
            )
            if style_index == 0:
                original_axis.text(
                    -0.12,
                    0.5,
                    true_label,
                    transform=original_axis.transAxes,
                    ha="right",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )
            if label_index == 0:
                original_axis.set_title(f"{style_name}\noriginal", fontsize=8)
                recon_axis.set_title(f"{style_name}\nrecon", fontsize=8)

            cell_records.append(
                {
                    "true_label_index": label_index,
                    "true_label": true_label,
                    "style_index": style_index,
                    "style": style_name,
                    "fragment_position": position,
                    "code_index": code_index,
                    "hungarian_mapped_label_index": mapped_index,
                    "hungarian_mapped_label": mapped_label,
                    "mapping_correct": correct,
                    "reconstruction_mse": reconstruction_mse,
                }
            )

    fig.suptitle(
        f"{checkpoint_name} | Hungarian mapping from {mapping_scope} "
        f"(accuracy={hungarian_accuracy:.4f})\n"
        "Recon annotation: quantized code -> mapped label; green=correct, red=mismatch",
        fontsize=12,
        y=0.999,
    )
    fig.subplots_adjust(
        left=0.035, right=0.997, bottom=0.005, top=0.975, wspace=0.02, hspace=0.06
    )
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)
    return cell_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run")
    parser.add_argument("--active_checkpoint")
    parser.add_argument("--config")
    parser.add_argument("--output_dir")
    args = parser.parse_args()

    resolved = resolve_evaluation_paths(
        run=args.run,
        active_checkpoint=args.active_checkpoint,
        config=args.config,
    )
    if resolved["active_checkpoint"] is None:
        parser.error("No checkpoint resolved.")
    if not torch.cuda.is_available():
        raise RuntimeError("Reconstruction-grid evaluation requires a GPU allocation.")

    config_path = Path(resolved["config"])
    checkpoint_path = Path(resolved["active_checkpoint"])
    run_dir = Path(resolved["run_dir"] or checkpoint_path.parent)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "vis"
    output_dir.mkdir(parents=True, exist_ok=True)
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    device = torch.device("cuda")
    loader, content_names, sampling = build_loader(config)
    dataloader_module = import_module("dataloader." + config["dataloader"])
    style_names = list(dataloader_module.S_LIST)
    model = get_model(config["dataloader"], config["model_config"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    counts = torch.zeros(
        (config["model_config"]["n_atoms"], len(content_names)),
        dtype=torch.int64,
        device=device,
    )
    sample_count = 0
    with torch.inference_mode():
        for batch_data, content_idx, _ in loader:
            batch_data = batch_data.to(device, non_blocking=True)
            outputs = model(batch_data, freeze_codebook=True)
            counts += batch_confusion_counts(
                outputs[3],
                content_idx,
                config["model_config"]["n_atoms"],
                len(content_names),
            )
            sample_count += int(batch_data.shape[0])

    counts_np = counts.cpu().numpy()
    metrics = compute_assignment_metrics(counts_np)
    code_to_label, label_to_code, hungarian_accuracy = compute_hungarian_mapping(
        counts_np
    )

    selected = select_one_sample_per_style(loader.dataset, style_names)
    selected_items = [loader.dataset[index] for index, _ in selected]
    originals = torch.stack([item[0] for item in selected_items]).to(device)
    content_indices = torch.stack([item[1] for item in selected_items])
    selected_style_indices = torch.stack([item[2] for item in selected_items])
    with torch.inference_mode():
        selected_outputs = model(originals, freeze_codebook=True)
    reconstructions = selected_outputs[0].detach().cpu()
    vq_indices = selected_outputs[3].detach().cpu()
    originals = originals.detach().cpu()

    checkpoint_stem = checkpoint_path.stem
    prefix = f"reconstruction_grid__{checkpoint_stem}__test-full-map__8styles"
    png_path = output_dir / f"{prefix}.png"
    json_path = output_dir / f"{prefix}.json"
    mapping_csv_path = output_dir / f"hungarian_mapping__{checkpoint_stem}__test-full.csv"
    write_mapping_csv(mapping_csv_path, counts_np, code_to_label, content_names)
    cell_records = render_grid(
        png_path,
        originals,
        reconstructions,
        content_indices,
        vq_indices,
        content_names,
        style_names,
        code_to_label,
        checkpoint_path.name,
        hungarian_accuracy,
    )

    selected_accuracy = float(
        np.mean([record["mapping_correct"] for record in cell_records])
    )
    summary = {
        "evaluated_at": datetime.datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "config": str(config_path.resolve()),
        "data_split": "test",
        "mapping_scope": sampling,
        "mapping_sample_count": sample_count,
        "mapping_fragment_count": int(counts_np.sum()),
        "hungarian_one_to_one_accuracy": hungarian_accuracy,
        "assignment_metrics": metrics,
        "code_to_label": {
            str(code): {
                "label_index": label,
                "label": content_names[label],
            }
            for code, label in sorted(code_to_label.items())
        },
        "label_to_code": {
            content_names[label]: code for label, code in sorted(label_to_code.items())
        },
        "grid": {
            "rows": len(content_names),
            "columns": len(style_names) * 2,
            "style_order": style_names,
            "column_pattern": ["original", "reconstruction"],
            "selected_sample_paths": {
                style: str(path) for style, (_, path) in zip(style_names, selected)
            },
            "selected_style_indices": selected_style_indices[:, 0].tolist(),
            "mapped_label_accuracy": selected_accuracy,
            "mean_reconstruction_mse": float(
                np.mean([record["reconstruction_mse"] for record in cell_records])
            ),
            "cells": cell_records,
        },
        "outputs": {
            "png": str(png_path.resolve()),
            "mapping_csv": str(mapping_csv_path.resolve()),
        },
    }
    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    print("Checkpoint:", checkpoint_path.resolve())
    print("Hungarian mapping scope: full test")
    print("Hungarian one-to-one accuracy:", hungarian_accuracy)
    print("Selected-grid mapped-label accuracy:", selected_accuracy)
    print("Selected-grid mean reconstruction MSE:", summary["grid"]["mean_reconstruction_mse"])
    print("Saved grid:", png_path.resolve())
    print("Saved mapping:", mapping_csv_path.resolve())
    print("Saved metadata:", json_path.resolve())


if __name__ == "__main__":
    main()
