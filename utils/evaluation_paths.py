import json
from pathlib import Path


def _require_file(path, description):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    return path


def resolve_run_directory(run, logs_dir="logs"):
    if run is None:
        return None
    candidate = Path(run).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    elif candidate.exists():
        resolved = candidate.resolve()
    else:
        resolved = (Path(logs_dir).expanduser() / candidate).resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {resolved}")
    return resolved


def checkpoint_from_metadata(run_dir, alias):
    metadata_name = {
        "current": "current_checkpoint.json",
        "best": "best_macro_atom_purity.json",
        "best_val": "best_validation_loss.json",
    }[alias]
    metadata_path = _require_file(
        Path(run_dir) / metadata_name, f"{alias} checkpoint metadata"
    )
    with metadata_path.open() as metadata_file:
        metadata = json.load(metadata_file)
    checkpoint_name = metadata.get("checkpoint")
    if not checkpoint_name:
        raise ValueError(
            f"Missing 'checkpoint' in {alias} checkpoint metadata: {metadata_path}"
        )
    return _require_file(
        Path(run_dir) / checkpoint_name, f"{alias} checkpoint"
    )


def resolve_evaluation_paths(
    run=None,
    active_checkpoint=None,
    config=None,
    logs_dir="logs",
):
    """Resolve config and checkpoint without allowing cross-run mismatches."""
    run_dir = resolve_run_directory(run, logs_dir=logs_dir)
    checkpoint_path = None

    if active_checkpoint in (None, "current") and run_dir is not None:
        checkpoint_path = checkpoint_from_metadata(run_dir, "current")
    elif active_checkpoint in ("best", "best_val"):
        if run_dir is None:
            raise ValueError(f"--active_checkpoint {active_checkpoint} requires --run.")
        checkpoint_path = checkpoint_from_metadata(run_dir, active_checkpoint)
    elif active_checkpoint is not None:
        checkpoint_candidate = Path(active_checkpoint).expanduser()
        if run_dir is not None and checkpoint_candidate.parent == Path("."):
            checkpoint_candidate = run_dir / checkpoint_candidate
        checkpoint_path = _require_file(
            checkpoint_candidate, "Evaluation checkpoint"
        )
        inferred_run_dir = checkpoint_path.parent.resolve()
        if run_dir is not None and inferred_run_dir != run_dir.resolve():
            raise ValueError(
                "Checkpoint is outside the requested run directory: "
                f"checkpoint={checkpoint_path}, run={run_dir}"
            )
        if run_dir is None:
            run_dir = inferred_run_dir

    if config is not None:
        config_path = _require_file(config, "Evaluation config")
    elif run_dir is not None:
        config_path = _require_file(run_dir / "config.yaml", "Run config")
    else:
        raise ValueError(
            "Cannot resolve config: provide --config, --run, or a checkpoint "
            "inside a run directory."
        )

    return {
        "run_dir": str(run_dir) if run_dir is not None else None,
        "config": str(config_path),
        "active_checkpoint": (
            str(checkpoint_path) if checkpoint_path is not None else None
        ),
    }
