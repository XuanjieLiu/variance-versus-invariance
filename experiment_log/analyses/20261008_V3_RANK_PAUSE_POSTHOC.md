# v3_rank pause and reconstruction diagnostics — 2026-10-08

User requested both training jobs stopped before further investigation. Jobs
222839/222840 were verified as these runs and cancelled at21:36:40+04. Slurm
reports CANCELLED; no other jobs were touched. All checkpoints/configs/history
were preserved. No resubmission is authorized.

| Arm | Last saved current | Completed epochs | Saved optimizer steps | Macro-best | Future resume starts |
|---|---|---:|---:|---|---|
| rank4 | cp_current_epoch69.pt | 70 | 175000 | cp_best_macro_atom_purity_epoch58.pt | epoch70 |
| rank2 | cp_current_epoch70.pt | 71 | 177500 | cp_best_macro_atom_purity_epoch59.pt | epoch71 |

Training continued briefly during Slurm termination; last step CSV entries are
176405/177858 respectively. These unsaved partial-epoch updates are NOT part of
the current checkpoint and must not be used as its recovery step.

## Why reconstruction_diagnostics ends at epoch050

This is scheduled behavior, not a plot failure. The config requests
`reconstruction_every_n_epochs: 25`, plus
`extra_grid_epochs: [0,10,11,12,15,20,50,100]`.
`DisentanglementMonitor.begin` uses `(epoch + 1) % 25 == 0 OR epoch in extras`.
Periodic plots are therefore024,049,074,099,124,149,174,199. Epoch050 is an extra
diagnostic; neither run reached074. All expected nodes through the last complete
epochs exist. Validation, color/probe/rank histories continue after050 each epoch.
No training/plot-code change is needed, and no historical images were overwritten.

## Epoch and step units

HexDigitsV2 has80000 training pages, batch32, one GPU, no gradient accumulation:
one epoch =2500 optimizer steps. Each step processes32 pages ×16 fragments.
The10000 validation pages add313 forward batches but NO optimizer steps.
Epoch numbers are zero-based and the diagnostics run AFTER training that epoch:
end-of-epoch step = (epoch + 1) ×2500. Thus epoch000=2500steps,
049=125000,050=127500,058=147500,059=150000,074=187500;200epochs=500000.
Step-level logging still exists in loss_history.csv; epoch aggregate plots do not
replace the optimizer step counter. UppercaseV2 has20800 train pages, therefore
only650steps/epoch atbatch32: an epoch is dataset/protocol-dependent.

## Posthoc evaluation

GPU-only evaluation job223311 onws-l1-014 evaluates both validation-selected
macro-best checkpoints, not best-val and not the last snapshot. Full held-out
test:10000 pages/160000 fragments,1250 per(content,style). Hungarian mapping uses
the full test; display selects one fixed page per style for a16-row×16-column
original/reconstruction grid. These are TEST examples, not the periodic VAL pages.

Artifacts stay below each run's
`reconstruction_diagnostics/posthoc_best_20261008/`: checkpoint-specific PNG,
mapping CSV, metadata JSON, foreground mask, balanced column-normalized
confusion PNG/SVG/CSV/JSON, and `health__macro-best__test-full.json`.
Evaluation reuses existing tools without altering checkpoints or training config.
Job223311 completed21:40:20+04 with exit0 and released its GPU. This is a
partial-run diagnosis, not acceptance of the originally planned200-epoch budget.

| Macro-best | Full-test macro | Weighted / one-to-one | Recon MSE | FG RGB MAE | FG chroma RMSE | Mean per-page effective rank |
|---|---:|---:|---:|---:|---:|---:|
| rank4 epoch58 | .999975005 | .999975 | .0235013 | .0885320 | .0572664 | 1.016563 |
| rank2 epoch59 | .999987502 | .9999875 | .0431347 | .1258322 | .1152751 | 1.827369 |

Both have16 active codes, perplexity approximately16 and coverage16. Matrix
mapping is fitted to held-out test for posthoc diagnostic alignment, never used
to choose the checkpoint. The16x16 displayed grids use only128 fragments;
the metrics above use the full160000 fragments. Visually, rank4 largely retains
style colors but teal is muted/blue-shifted; rank2 shows clear systematic color
shifts (e.g. black→green, brown→purple) despite correct content mapping. Color
metrics compare observed noisy inputs using input-derived foreground masks.

The rank4 floor has NOT succeeded at this checkpoint, despite near-perfect
purity. Rank2's best is closer to2, and its latest saved validation epoch70
reaches2.00279 (not a new full-test/current evaluation). Neither result establishes
arithmetic usability or persistence through the planned200epochs.

Checkpoint SHA256:

- rank4 epoch58: `177db17bf8c41e55732d52e795d18c504f698d1b6fd5ac984e1645da6d7d41b5`
- rank2 epoch59: `0dd986ed363a266653de7e7e4e73c85d0730161d4039fd68d918f5e1c1a636a9`

All test outputs include checkpoint identity/hash; original configuration and
dataset identity remain in the run/provenance. No Python or evaluation ran on
the login node. No code/config/checkpoint was changed; only research/manual
records and new posthoc diagnostic artifacts were written.

## Native-codebook projections (follow-up request)

GPU job223326 onws-l1-004 generated PCA/t-SNE/Isomap in2D and3D for the same
macro-best58/59 checkpoints, using the existing geometry script unchanged. Labels
come from hash-verified full10000-page test Hungarian assignments; no labels
enter fitting. Raw16x512 native atoms, no whitening or feature normalization;
seed0, t-SNE perplexity4/exact/3000iterations, connected5-neighbor Isomap.

| Macro-best | PC1 variance | PC2 variance | First2 total | Native participation-ratio dimension |
|---|---:|---:|---:|---:|
| rank4 epoch58 | 99.178619% | .820944% | 99.999563% | 1.016563 |
| rank2 epoch59 | 65.747100% | 33.905254% | 99.652354% | 1.827371 |

These are all-atom unweighted native-space statistics, unlike the per-page
statistics above. Rank4 remains nearly collinear (with a small bend), whereas
rank2 has two appreciable linear directions. The existing unregularized Hex MEAN
macro-best171 report has PC1=99.993414%, native effective dimension1.000132; this
is a historical checkpoint comparison, not a matched-time/seed-replication claim.
Neither layout establishes numeric ordering or addition usability. In particular,
3D t-SNE spreads both codebooks and must not be read as evidence of rank3.

Each run now contains six individual PNG/SVG plots, overview PNG/SVG, coordinates,
raw atoms/distances and geometry_report.json under:

- rank4: `codebook_geometry/cp_best_macro_atom_purity_epoch58__native-vq__seed0__p4__k5__v3_rank_20261008/`
- rank2: `codebook_geometry/cp_best_macro_atom_purity_epoch59__native-vq__seed0__p4__k5__v3_rank_20261008/`

Source checkpoint hashes were checked unchanged after plotting. Training remains
stopped; this follow-up creates diagnostics only, not another formal run.
