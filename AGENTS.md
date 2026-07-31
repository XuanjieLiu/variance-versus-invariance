# Variance-versus-invariance Agent Instructions

These instructions apply to every agent working in this repository.

## Compute safety

- `lo-*` hosts are login nodes. Never run Python, tests, training, evaluation, or
  recomputation directly on a login node.
- Run smoke tests and short checks through `srun` on `ws-ia` with one GPU.
- Submit formal training through `scripts/submit_slurm.sh`.
- Interactive `salloc` sessions must be released with `exit`. `sbatch` resources
  are released automatically when the job reaches a terminal state.
- Do not modify or cancel Slurm jobs belonging to another project.

## Formal experiment lifecycle

- `experiment_log/SETTINGS.md`, `ACTIVE_RUNS.md`, and `ARCHIVE.md` are the
  tracked source of truth for formal experiments.
- Define a setting, its metrics, and its decision gate before launching runs.
- Smoke runs are never added to the experiment ledger.
- Before submitting a formal run, add it to `ACTIVE_RUNS.md`. After submission,
  add the Slurm job ID and current status.
- On acceptance, failure, timeout, or cancellation, remove the run from
  `ACTIVE_RUNS.md` and add one concise row to `ARCHIVE.md` with its terminal
  state, best checkpoint or failure point, core metrics, and conclusion.
- A Run ID must never appear in both active and archive tables.
- Keep checkpoints, plots, generated summaries, and Slurm output under ignored
  artifact directories. Do not add them to Git.
- Preserve unrelated user changes. Do not commit or push unless explicitly
  requested.
