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

- `experiment_log/README.md`, `experiment_log/directions/*.md`,
  `experiment_log/SETTINGS.md`, `ACTIVE_RUNS.md`, and `ARCHIVE.md` are the
  tracked source of truth for research and formal experiments.
- Direction files own cross-run hypotheses and conclusions; SETTINGS owns
  controlled protocols; ACTIVE and ARCHIVE own run lifecycle state.
- Define a setting, its metrics, and its decision gate before launching runs.
- Smoke runs are never added to the experiment ledger and their artifact
  directories must be deleted immediately after validation.
- Before submitting a formal run, add it to `ACTIVE_RUNS.md`. After submission,
  add the Slurm job ID and current status.
- On acceptance, failure, timeout, or cancellation, remove the run from
  `ACTIVE_RUNS.md` and add one concise row to `ARCHIVE.md` with its terminal
  state, best checkpoint or failure point, core metrics, and conclusion.
- A Run ID must never appear in both active and archive tables.
- Keep checkpoints, plots, generated summaries, and Slurm output under ignored
  artifact directories. Do not add them to Git.
- New runs use the `current_and_best_macro` policy: retain exactly the latest
  current checkpoint and the best validation macro-atom-purity checkpoint.
- Explicitly registered periodic snapshots are an exception: user-requested
  VVI-RQ2-S3/S4/S5/S6/S7/S8/S9/S10, VVI-PN-S1 and VVI-HEX-S1 runs keep `cp_snapshot_epoch*.pt` every 25 completed epochs in
  addition to current/best. Never prune these with current/best rotation.
- VVI-PN-S1, VVI-HEX-S1 and VVI-RQ2-S7/S8/S9/S10 additionally retain one best-validation-loss checkpoint:
  at most 11 files at 200 epochs; the historical S6 C128 W50 remains at most10.
  S10 R1 is the user-authorized 570-epoch continuation (absolute200..769):
  retain22 new25-epoch snapshots plus current/macro-best/best-val, at most25
  files per R1 run. Keep parent snapshots in the parent run, without duplication.
  Never use test labels or test losses to choose any retained checkpoint.
- S7 BN interventions are diagnostics only. Freeze VQ and restore every buffer,
  module mode and RNG state, including on errors; never select checkpoints using
  the small-batch intervention metrics. Do not change the training BN architecture.
- S8 is the explicitly authorized exception: matched W50/lr1e-4 arms compare
  decoder BatchNorm versus GroupNorm(8 groups), including decoder shortcuts.
  Keep encoder BN, all objectives, EMA and other training settings unchanged.
  Old configs default to BN; never reinterpret an old checkpoint as a GN model.
  GN's decoder-only BN diagnostic is a declared no-op, not evidence of recovery.
- S9 explicitly removes decoder normalization in both matched W50/lr1e-4 arms;
  one retains encoder BN and one also removes it. Replace by Identity, including
  shortcuts, from scratch. Keep fixed input scaling, V3 equations, VQ/EMA,
  initialization and optimizer unchanged; no unregistered Fixup/scaling changes.
  All-NoNorm BN interventions are declared no-ops. Shared initialization hashes
  exclude normalization affine parameters, which are absent in the removed arm.
  Per-style full-validation counts and sampled optimization scales are diagnostic
  only and must not alter checkpoint selection, RNG or optimizer steps.
- Preserve unrelated user changes. Do not commit or push unless explicitly
  requested.

## Dataset-version comparisons

- Never overwrite historical UppercaseLetters PNGs. V2 uses its own directory,
  corrected signed-noise/full-strip renderer, pinned seeds and manifest hash.
- PhoneNumsV2 likewise has a separate immutable manifest and corrected colors;
  PAPER/REPO use the same100,000 pages and differ only in declared training
  parameter bundles. Do not silently substitute legacy PhoneNums data.
- HexDigitsV2 is a separate immutable 100,000-page PhoneNums-rendered extension,
  lowercase `0123456789abcdef`, integer labels0..15,16 fragments/page. Preserve
  PhoneNums font/48px layout and noise distribution, not uppercase fit-to-bbox.
  HEX-S1 restores original encoder/decoder BN and REPO lr1e-3; only fragment
  versus permanent page-mean decoder input differs. V3 still uses raw z_s.
  Group0..9/a..f metrics must use the same global Hungarian assignment; never
  rematch groups or styles to improve apparent accuracy. No addition training
  or native-VQ dimension reduction is authorized in this setting.
- V2 CTRL/W100 retain the original BatchNorm and training objective. Do not add
  a simultaneous normalization or optimizer intervention.
- S10 reuses immutable UppercaseLettersV2 and the HEX-S1 optimizer/objective/BN
  bundle, comparing ordinary fragment versus permanent page-mean decoding.
  Registered default budget is200 natural epochs/130k steps, not Hex's500k;
  do not claim class-count isolation or silently regenerate/resample the data.
  User update2026-09-17 authorizes S10 R1 from current199 to769, reaching500,500
  cumulative updates. Restore optimizer/scheduler/scaler and keep the existing
  epoch-based LR schedule; this is budget extension, NOT a step-matched LR test.
- Foreground masks are estimates from observed inputs, not clean references or
  true glyph masks. Color metrics never affect checkpoint selection or loss.
- Balanced reconstruction-node confusion matrices show P(code | content), with
  actual content/style counts verified and the same Hungarian mapping as the grid.
