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
  VVI-RQ2-S3/S4/S5/S6/S7/S8/S9/S10, VVI-PN-S1, VVI-HEX-S1, VVI-HEX-RANK-S1 and VVI-HEX-DIVEQ-S1 runs keep `cp_snapshot_epoch*.pt` every 25 completed epochs in
  addition to current/best. Never prune these with current/best rotation.
- VVI-PN-S1, VVI-HEX-S1, VVI-HEX-RANK-S1, VVI-HEX-DIVEQ-S1 and VVI-RQ2-S7/S8/S9/S10 additionally retain one best-validation-loss checkpoint:
  at most 11 files at 200 epochs; the historical S6 C128 W50 remains at most10.
  S10 R1 is the user-authorized 570-epoch continuation (absolute200..769):
  retain22 new25-epoch snapshots plus current/macro-best/best-val, at most25
  files per R1 run. Keep parent snapshots in the parent run, without duplication.
  Never use test labels or test losses to choose any retained checkpoint.
- IBP-AUDIO-S2 is the 2026-11-14 authorized mel REPO fragment/MEAN pair:
  2000epochs/60000updates each, same data/seed and original repository
  optimizer/VQ bundle. Retain current + health-gated macro-best + independent
  best-validation-loss (at most3 files, no periodic model snapshots). Audio
  previews are diagnostic, including a final export of the loss-selected model.
- IBP-AUDIO-S2-R1 is the 2026-11-19 authorized continuation of BOTH source
  arms from current1999 to9999:8000 additional epochs/240000updates, total
  10000epochs/300000updates. Preserve optimizer/scheduler/scaler, data and
  all hyperparameters; inherit histories and retained bests into separate R1
  directories. Keep the same at-most3 checkpoint policy. No OOD FT is implied.
- IBI-RD-HEX-S1 is the 2026-11-20 authorized image representation-distillation
  source pair: independent student encoder/decoder/VQ, one frozen RANK2
  epoch59 teacher, LR1e-4 versus3e-4,200epochs/500000updates. Keep current,
  health-gated macro-best and independent minimum-validation-RD-loss (at most3
  checkpoints). V3/rank terms are diagnostics only; targets/calibration use
  source data, never OOD. Do not start OOD FT or a third recon-only control
  without a separate user decision. The preferred research name is now IBI;
  preserve historical IBP run IDs and artifact paths.
- IBI-RD-HEX-S2 is the 2026-11-21 authorized LR1e-4 source replica:50epochs,
  permanent zero-based snapshots0/4/19/49 plus current/macro-best/best-val.
  This exception retains early source states for matched V5 reconstruction-only
  fine-tuning in new style, new content, and both. The earliest three FT jobs
  may start once snapshot0 is atomically saved; no OOD outcome selects ages.
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
- VVI-HEX-RANK-S1 is the 2026-10-08 user-authorized v3_rank pair on the original
  Hex16 permanent-MEAN protocol: target_rank4 versus2, weight.01, otherwise
  unchanged seed0/C512/S512/K16/BN/EMA/200epoch settings. Here rank target is NOT
  n_atoms. Preserve labels as diagnostics only; no addition training or dim change.
- S10 reuses immutable UppercaseLettersV2 and the HEX-S1 optimizer/objective/BN
  bundle, comparing ordinary fragment versus permanent page-mean decoding.
  Registered default budget is200 natural epochs/130k steps, not Hex's500k;
  do not claim class-count isolation or silently regenerate/resample the data.
  User update2026-09-17 authorizes S10 R1 from current199 to769, reaching500,500
  cumulative updates. Restore optimizer/scheduler/scaler and keep the existing
  epoch-based LR schedule; this is budget extension, NOT a step-matched LR test.
- VVI-HEX-DIVEQ-S1 is the 2026-10-10 authorized scratch Hex16 comparison:
  ordinary fragment decoder in BOTH arms, C512/S512 versus C4/S4, native VQ
  dimension equal to content width, K16,200epochs. Use trainable stochastic
  DiVeQ, no EMA/commitment or rank penalty; preserve BN and the Hex REPO
  optimizer/data/other objectives. This is a quantizer-bundle comparison, not
  an isolated EMA toggle. Do not resume the cancelled rank pair or add MEAN.
- Foreground masks are estimates from observed inputs, not clean references or
  true glyph masks. Color metrics never affect checkpoint selection or loss.
- Balanced reconstruction-node confusion matrices show P(code | content), with
  actual content/style counts verified and the same Hungarian mapping as the grid.
