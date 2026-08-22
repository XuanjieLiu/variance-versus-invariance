import argparse
import csv
import datetime
import json
import os
from importlib import import_module

import numpy as np
import torch
import yaml

from model.factory import get_model
from model.v3_loss import V3Loss
from utils.codebook_metrics import (
    batch_confusion_counts,
    compute_alias_geometry_metrics,
    compute_assignment_metrics,
    file_sha256,
)
from utils.subset_sampling import SUBSET_STRATEGIES, make_subset_loader
from model.v3_loss import V3_RATIO_KEYS


def select_best_checkpoint(run_dir):
    history_path = os.path.join(run_dir, "loss_epoch_history.csv")
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"Missing epoch loss history: {history_path}")

    candidates = []
    with open(history_path, newline="") as history_file:
        for row in csv.DictReader(history_file):
            if row["partition"] != "val" or not row["total_loss"]:
                continue
            epoch = int(row["epoch"])
            candidates.append((float(row["total_loss"]), epoch))

    for val_loss, epoch in sorted(candidates):
        checkpoint = os.path.join(run_dir, f"cp_epoch{epoch}.pt")
        if os.path.exists(checkpoint):
            return checkpoint, epoch, val_loss
    raise FileNotFoundError(
        f"No retained checkpoint matches validation history in {run_dir}"
    )


def select_best_macro_checkpoint(run_dir):
    metadata_path = os.path.join(run_dir, "best_macro_atom_purity.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing macro-purity metadata: {metadata_path}")
    with open(metadata_path) as metadata_file:
        metadata = json.load(metadata_file)
    checkpoint = os.path.join(run_dir, metadata["checkpoint"])
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(checkpoint)
    return (
        checkpoint,
        int(metadata["epoch"]),
        float(metadata["macro_atom_purity"]),
        float(metadata["validation_total_loss"]),
    )


def compute_codebook_metrics(confusion_counts, codebook):
    counts = np.asarray(confusion_counts, dtype=np.float64)
    metrics = compute_assignment_metrics(counts)

    codebook = torch.as_tensor(codebook, dtype=torch.float32)
    if codebook.ndim == 3 and codebook.shape[0] == 1:
        codebook = codebook.squeeze(0)
    if codebook.ndim != 2:
        raise ValueError(f"Expected a 2D codebook, got shape {tuple(codebook.shape)}")
    pairwise_distances = torch.pdist(codebook)
    min_atom_distance = (
        float(pairwise_distances.min().item())
        if pairwise_distances.numel()
        else float("nan")
    )

    alias_metrics = compute_alias_geometry_metrics(counts, codebook)
    return {**metrics, **alias_metrics, "min_atom_distance": min_atom_distance}


def resolve_data_path(config):
    for name in ("test", "test.hdf5", "val", "val.hdf5"):
        path = os.path.join(config["data_dir"], name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No test or validation split under {config['data_dir']}")


def build_loader(
    config, subset_size=None, subset_seed=0, subset_strategy="random"
):
    dataloader_module = import_module("dataloader." + config["dataloader"])
    data_path = resolve_data_path(config)
    loader = dataloader_module.get_dataloader(
        data_path,
        batch_size=config["batch_size"],
        num_workers=config.get("num_workers", 0),
        n_fragments=config["model_config"]["n_fragments"],
        fragment_len=config["model_config"]["fragment_len"],
        shuffle=False,
    )
    if subset_size is None:
        metadata = {
            "subset_size": None,
            "subset_seed": None,
            "subset_strategy": "full",
            "style_counts": {},
        }
        return loader, dataloader_module.C_LIST, metadata

    subset_loader, metadata = make_subset_loader(
        loader,
        subset_size,
        seed=subset_seed,
        strategy=subset_strategy,
    )
    return subset_loader, dataloader_module.C_LIST, metadata


def evaluate(
    config,
    checkpoint,
    subset_size=None,
    subset_seed=0,
    subset_strategy="random",
):
    if not torch.cuda.is_available():
        raise RuntimeError("Codebook health evaluation requires a GPU allocation.")
    device = torch.device("cuda")

    loader, content_labels, subset_metadata = build_loader(
        config, subset_size, subset_seed, subset_strategy
    )
    model = get_model(config["dataloader"], config["model_config"]).to(device)
    checkpoint_state = torch.load(
        checkpoint, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint_state["model"])
    model.eval()
    loss_fn = V3Loss(config["loss_config"])

    confusion_counts = torch.zeros(
        (config["model_config"]["n_atoms"], len(content_labels)),
        dtype=torch.int64,
        device=device,
    )
    loss_sums = {}
    v3_ratio_sums = {}
    sample_count = 0
    first_batch = None

    with torch.inference_mode():
        for batch_data, content_idx, _ in loader:
            batch_data = batch_data.to(device, non_blocking=True)
            if first_batch is None:
                first_batch = batch_data.detach().clone()
            outputs = model(batch_data, freeze_codebook=True)
            losses = loss_fn.compute_loss(
                outputs[0],
                outputs[1],
                outputs[2],
                outputs[4],
                outputs[5],
                batch_data,
            )
            batch_size = batch_data.shape[0]
            sample_count += batch_size
            for name, value in losses.items():
                target = v3_ratio_sums if name in V3_RATIO_KEYS else loss_sums
                target[name] = target.get(name, 0.0) + value.item() * batch_size

            confusion_counts += batch_confusion_counts(
                outputs[3],
                content_idx,
                config["model_config"]["n_atoms"],
                len(content_labels),
            )

    mean_losses = {name: value / sample_count for name, value in loss_sums.items()}
    mean_v3_ratios = {
        name: value / sample_count for name, value in v3_ratio_sums.items()
    }
    codebook = model.vq.codebook.detach().cpu()
    confusion_counts_cpu = confusion_counts.cpu().numpy()
    codebook_metrics = compute_codebook_metrics(confusion_counts_cpu, codebook)

    with torch.inference_mode():
        model.eval()
        eval_outputs = model(first_batch, freeze_codebook=True)
        eval_recon = torch.nn.functional.mse_loss(eval_outputs[0], first_batch).item()
        model.train()
        train_outputs = model(first_batch, freeze_codebook=True)
        train_recon = torch.nn.functional.mse_loss(train_outputs[0], first_batch).item()

    return {
        "sample_count": sample_count,
        "fragment_count": int(confusion_counts_cpu.sum()),
        "sampling": subset_metadata,
        "losses": mean_losses,
        "v3_ratios": mean_v3_ratios,
        "codebook": codebook_metrics,
        "fixed_batch": {
            "eval_recon_loss": eval_recon,
            "train_recon_loss": train_recon,
            "eval_to_train_recon_ratio": eval_recon / max(train_recon, 1e-12),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument(
        "--selection-metric",
        choices=("val_loss", "macro_atom_purity"),
        default="val_loss",
    )
    parser.add_argument("--test-subset-size", type=int)
    parser.add_argument("--test-subset-seed", type=int, default=0)
    parser.add_argument(
        "--test-subset-strategy",
        choices=SUBSET_STRATEGIES,
        default="random",
    )
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)

    selected_epoch = None
    selection_val_loss = None
    selection_macro_atom_purity = None
    if args.checkpoint:
        checkpoint = args.checkpoint
    elif args.run_dir:
        if args.selection_metric == "macro_atom_purity":
            (
                checkpoint,
                selected_epoch,
                selection_macro_atom_purity,
                selection_val_loss,
            ) = select_best_macro_checkpoint(args.run_dir)
        else:
            checkpoint, selected_epoch, selection_val_loss = select_best_checkpoint(
                args.run_dir
            )
    else:
        parser.error("Provide either --checkpoint or --run-dir.")

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(checkpoint)

    results = evaluate(
        config,
        checkpoint,
        subset_size=args.test_subset_size,
        subset_seed=args.test_subset_seed,
        subset_strategy=args.test_subset_strategy,
    )
    summary = {
        "evaluated_at": datetime.datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "config": os.path.abspath(args.config),
        "checkpoint": os.path.abspath(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "selected_epoch": selected_epoch,
        "selection_metric": args.selection_metric,
        "selection_val_loss": selection_val_loss,
        "selection_macro_atom_purity": selection_macro_atom_purity,
        "test_subset_size": args.test_subset_size,
        "test_subset_seed": args.test_subset_seed,
        "test_subset_strategy": (
            args.test_subset_strategy if args.test_subset_size is not None else "full"
        ),
        **results,
    }

    output_path = args.output
    if output_path is None:
        output_dir = args.run_dir or os.path.dirname(checkpoint)
        output_path = os.path.join(output_dir, "codebook_health.json")
    with open(output_path, "w") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
