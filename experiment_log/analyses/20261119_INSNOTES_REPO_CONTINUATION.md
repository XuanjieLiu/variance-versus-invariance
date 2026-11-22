# InsNotes REPO 10000epoch continuation, 2026-11-19

After [2000epoch acceptance](20261119_INSNOTES_REPO_ACCEPTANCE.md), the user
authorized matching the repository10000epoch budget. This is a continuation,
not a restart, hyperparameter intervention, or OOD fine-tuning experiment.
Protocol: [IBP-AUDIO-S2-R1](../SETTINGS.md#ibp-audio-s2-r1--source-budget-continuation-to10000-epochs).

| Arm | Run ID | SLURM | Node | Run directory under logs/ |
|---|---|---|---|---|
| Fragment z_s | VVI-IBP-AUDIO-MEL-REPO-K12-S0-R1 | 309229 | ws-l2-007 | `20261119-1654__VVI-IBP-INSNOTES-MEL-REPO-K12-S0-R1` |
| Mean z_s | VVI-IBP-AUDIO-MEL-REPO-MEAN-K12-S0-R1 | 309230 | ws-l2-008 | `20261119-1654__VVI-IBP-INSNOTES-MEL-REPO-MEAN-K12-S0-R1` |

Both submitted with `scripts/submit_slurm.sh`, ws-ia,1RTX5000Ada,32GB,4CPU,
24h and `/home/xuanjie.liu/.venvs/v3_audio/bin/python`. No Python ran on login.

## Budget and state

- Load each parent `20261114-1706__...` current1999, not its selected best.
- `epochs: 8000` means2000..9999:240000 new updates,300000 cumulative.
- Model, BN, EMA, AdamW moments, scheduler and scaler are restored unchanged.
  LR resumes .00013261955589475294 and bottoms out at.00002 at epoch3873.
- Parent directories are preserved. Completed histories and actual retained
  macro-best/loss-best files are inherited; best scores are not reset.
- Old audio diagnostic rows are inherited after checking manifest, partitions,
  fixed sample grid and probe settings. Parent WAVs are linked in lineage,
  not duplicated. New WAVs every200epochs and final9999; final best export.
- No RNG was saved in the old checkpoints. Seed0 restarts reproducible resumed
  sampling, not the exact uninterrupted training sequence. This limitation is
  explicitly recorded in each `resume_lineage.json`.

The mel data/generator, all losses, optimizer hyperparameters, architecture,
normalization, eval schedule and mean/fragment difference remain unchanged.
Source semantic gate and test isolation also remain unchanged.

## Provenance

Configs under `configs/insnotes/`:

| Config | SHA256 |
|---|---|
| `cfg_vvi_insnotes_mel_repo_k12_seed0_r1_10000e.yaml` | `aa8c377b4d88aeb5c90b9cf026562510c3b8a799913ec3f177f81525380c1d00` |
| `cfg_vvi_insnotes_mel_repo_mean_k12_seed0_r1_10000e.yaml` | `1c645c4a7b5be12d563dc71fb58401374b6290eb40995db14b9aba4514a8aa0a` |

Parent current1999 SHA256:

- Fragment: `034d40a881db27a216635447d396e725d2d6d3e8c0ba70854d6d1ef9b7259a96`.
- Mean: `de439d34f83807a894bc9591fbd1c7899b014f97352d7b2da47918cefd252c7b`.

Each formal run archives its submitted config/source and parent lineage.
Environment retains OMP/MKL4, OPENBLAS/NUMBA1 and the audio environment from S2.

## Preflight

Job309227 passed166unit tests and one full resumed epoch/30updates per arm.
Model, optimizer, scheduler and scaler exactly match the parent at restore.
Both show finite losses and nonzero encoder/decoder gradients; current2000
and all three retained checkpoint roles exist. Prior best epochs1880/1920
and their audio exports are preserved. Audio curves include epoch0..2000,
and parent checkpoint SHA256 is unchanged after the smoke. Temporary artifacts
are deleted. Evidence: `logs/insnotes_resume_preflight_309227.json` and `.log`.

Engineering changes are limited to inheriting audio scalar history, checking
audio protocol consistency, and linking parent audio artifacts. Training loss
or gradient behavior was not changed. Smoke validation is not scientific
evidence about the long-budget outcome.
