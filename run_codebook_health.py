import argparse
import csv
import datetime
import hashlib
import json
import os
from importlib import import_module

import numpy as np
import torch
import yaml
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Subset

from model.factory import get_model
from model.v3_loss import V3Loss


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


def compute_codebook_metrics(confusion_counts, codebook):
    counts = np.asarray(confusion_counts, dtype=np.float64)
    usage = counts.sum(axis=1)
    total = usage.sum()
    active_mask = usage > 0

    if total == 0:
        raise ValueError("No code assignments were collected.")

    usage_prob = usage[active_mask] / total
    usage_perplexity = float(np.exp(-np.sum(usage_prob * np.log(usage_prob))))
    purity = float(counts.max(axis=1).sum() / total)
    row_ind, col_ind = linear_sum_assignment(-counts)
    one_to_one_accuracy = float(counts[row_ind, col_ind].sum() / total)
    dominant_label_coverage = int(
        np.unique(np.argmax(counts[active_mask], axis=1)).size
    )

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

    return {
        "codebook_size": int(counts.shape[0]),
        "content_label_count": int(counts.shape[1]),
        "active_codes": int(active_mask.sum()),
        "usage_perplexity": usage_perplexity,
        "codebook_purity": purity,
        "one_to_one_accuracy": one_to_one_accuracy,
        "dominant_label_coverage": dominant_label_coverage,
        "min_atom_distance": min_atom_distance,
    }


def resolve_data_path(config):
    for name in ("test", "test.hdf5", "val", "val.hdf5"):
        path = os.path.join(config["data_dir"], name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No test or validation split under {config['data_dir']}")


def build_loader(config, subset_size=None, subset_seed=0):
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
        return loader, dataloader_module.C_LIST

    generator = torch.Generator().manual_seed(subset_seed)
    count = min(int(subset_size), len(loader.dataset))
    indices = torch.randperm(len(loader.dataset), generator=generator)[:count]
    subset = Subset(loader.dataset, sorted(indices.tolist()))
    subset_loader = DataLoader(
        subset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 0),
    )
    return subset_loader, dataloader_module.C_LIST


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(config, checkpoint, subset_size=None, subset_seed=0):
    if not torch.cuda.is_available():
        raise RuntimeError("Codebook health evaluation requires a GPU allocation.")
    device = torch.device("cuda")

    loader, content_labels = build_loader(config, subset_size, subset_seed)
    model = get_model(config["dataloader"], config["model_config"]).to(device)
    checkpoint_state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint_state["model"])
    model.eval()
    loss_fn = V3Loss(config["loss_config"])

    confusion_counts = np.zeros(
        (config["model_config"]["n_atoms"], len(content_labels)), dtype=np.int64
    )
    loss_sums = {}
    sample_count = 0
    first_batch = None

    with torch.no_grad():
        for batch_data, content_idx, _ in loader:
            batch_data = batch_data.to(device)
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
                loss_sums[name] = loss_sums.get(name, 0.0) + value.item() * batch_size

            indices = outputs[3].detach().cpu().numpy().reshape(-1)
            labels = content_idx.numpy().reshape(-1)
            np.add.at(confusion_counts, (indices, labels), 1)

    mean_losses = {name: value / sample_count for name, value in loss_sums.items()}
    codebook = model.vq.codebook.detach().cpu()
    codebook_metrics = compute_codebook_metrics(confusion_counts, codebook)

    with torch.no_grad():
        model.eval()
        eval_outputs = model(first_batch, freeze_codebook=True)
        eval_recon = torch.nn.functional.mse_loss(eval_outputs[0], first_batch).item()
        model.train()
        train_outputs = model(first_batch, freeze_codebook=True)
        train_recon = torch.nn.functional.mse_loss(train_outputs[0], first_batch).item()

    return {
        "sample_count": sample_count,
        "fragment_count": int(confusion_counts.sum()),
        "losses": mean_losses,
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
    parser.add_argument("--test-subset-size", type=int)
    parser.add_argument("--test-subset-seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)

    selected_epoch = None
    selection_val_loss = None
    if args.checkpoint:
        checkpoint = args.checkpoint
    elif args.run_dir:
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
    )
    summary = {
        "evaluated_at": datetime.datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "config": os.path.abspath(args.config),
        "checkpoint": os.path.abspath(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "selected_epoch": selected_epoch,
        "selection_val_loss": selection_val_loss,
        "test_subset_size": args.test_subset_size,
        "test_subset_seed": args.test_subset_seed,
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
