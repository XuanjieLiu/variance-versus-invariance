"""Carry completed histories and the actual macro-best into a new resume run."""

import csv
import json
from pathlib import Path
import shutil

from utils.codebook_metrics import file_sha256


HISTORIES = (
    "loss_history.csv", "loss_epoch_history.csv", "codebook_epoch_history.csv",
    "v3_ratio_epoch_history.csv", "disentanglement_probe_history.csv",
    "objective_schedule_epoch_history.csv",
    "foreground_color_epoch_history.csv", "training_codebook_usage.csv",
    "bn_diagnostic_epoch_history.csv", "optimization_scale_history.csv",
    "optimization_scale_epoch_history.csv", "per_style_codebook_epoch_history.csv",
)


def copy_completed_history(source, destination, completed_epoch):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing resume history: {destination}")
    count = 0
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with source.open(newline="") as incoming, temporary.open("w", newline="") as outgoing:
        reader = csv.DictReader(incoming)
        if not reader.fieldnames or "epoch" not in reader.fieldnames:
            raise ValueError(f"History has no epoch column: {source}")
        writer = csv.DictWriter(outgoing, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if int(row["epoch"]) <= completed_epoch:
                writer.writerow(row)
                count += 1
    temporary.replace(destination)
    return count


def inherit_resume_artifacts(source_checkpoint, destination_dir, checkpoint_state):
    source_checkpoint = Path(source_checkpoint).resolve()
    source, destination = source_checkpoint.parent, Path(destination_dir).resolve()
    if source == destination:
        raise ValueError("History inheritance requires a new run directory")
    completed_epoch = int(checkpoint_state["epoch"])
    protocol_file = "disentanglement_probe_protocol.json"
    if (source / protocol_file).exists() and (destination / protocol_file).exists():
        old = json.loads((source / protocol_file).read_text())
        new = json.loads((destination / protocol_file).read_text())
        for key in ("fit_pages", "score_pages", "grid_pages", "protocol"):
            if old[key] != new[key]:
                raise ValueError(f"Resume diagnostic protocol differs in {key}")

    best_path = None
    best_epoch = checkpoint_state.get("best_macro_epoch")
    if checkpoint_state.get("best_macro_atom_purity") is not None:
        if best_epoch is None or int(best_epoch) > completed_epoch:
            raise ValueError("Invalid best epoch in resume state")
        label = "init" if int(best_epoch) < 0 else str(best_epoch)
        filename = f"cp_best_macro_atom_purity_epoch{label}.pt"
        parent_best = source / filename
        if not parent_best.is_file():
            raise FileNotFoundError(f"Resume tracks macro-best but its file is missing: {parent_best}")
        best_path = destination / filename
        if best_path.exists() or (destination / "best_macro_atom_purity.json").exists():
            raise FileExistsError("Refusing to overwrite an existing macro-best in resume directory")
        shutil.copy2(parent_best, best_path)
        metadata = {
            "checkpoint": filename, "epoch": int(best_epoch),
            "macro_atom_purity": checkpoint_state["best_macro_atom_purity"],
            "validation_total_loss": checkpoint_state["best_macro_val_loss"],
            "stage": checkpoint_state.get("best_macro_stage"),
            "inherited_from": str(parent_best),
            "checkpoint_sha256": file_sha256(best_path),
        }
        (destination / "best_macro_atom_purity.json").write_text(json.dumps(metadata, indent=2) + "\n")

    copied_rows = {}
    for name in HISTORIES:
        if (source / name).is_file():
            copied_rows[name] = copy_completed_history(source / name, destination / name, completed_epoch)
    lineage = {
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": file_sha256(source_checkpoint),
        "completed_epoch": completed_epoch, "next_epoch": completed_epoch + 1,
        "source_run": str(source), "inherited_history_rows": copied_rows,
        "inherited_macro_best": str(best_path) if best_path else None,
        "parent_snapshot_paths": [str(path) for path in sorted(source.glob("cp_snapshot_epoch*.pt"))],
        "parent_reconstruction_directory": str(source / "reconstruction_diagnostics"),
        "optimizer_scheduler_scaler_restored": True,
        "rng_restored": False,
        "note": "Legacy checkpoints lack RNG state. Epoch-boundary resume, not bitwise data-order continuation.",
    }
    (destination / "resume_lineage.json").write_text(json.dumps(lineage, indent=2) + "\n")
    return str(best_path) if best_path else None
