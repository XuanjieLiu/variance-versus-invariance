# v3_rank: native-VQ effective-rank floor

Added 2026-10-08. User-confirmed method name **v3_rank**; variants **rank2** and
**rank4** name the target participation-ratio dimension, NOT codebook size. This is an opt-in
regularizer, not a new model architecture or an accepted experimental result.
Adding the option alone launches no training. The user-authorized formal pair
is now registered as `VVI-HEX-RANK-S1` (Hex16 permanent MEAN, target4 versus2).

## Configuration

Set `method: v3_rank` (`method: V3` remains compatible). Add the following nested block to the existing `loss_config`,
preserving its other entries (especially `weights`):

```yaml
loss_config:
  rank_regularization:
    enabled: true
    target_rank: 4
    weight: 0.01
    eps: 1.0e-12
```

The same fragment is in `configs/regularization/v3_rank4.yaml`; it is not a
standalone run config and there is no automatic YAML-overlay loader.
`enabled` defaults to false. When enabled, the other defaults are shown above.
The initial weight .01 is a starting value, not an empirically tuned recommendation.
`target_rank` may be a finite real number >=1; weight must be finite and >=0;
epsilon must be finite and >0. Unsupported options raise an error. Do not add
`rank_loss` or `weighted_rank_loss` to `weights`: the new contribution is added
exactly once through the block above. A zero weight enables diagnostics only.

Existing `run_training.py` also accepts a two-level override:

```bash
# Inside a GPU compute allocation only; this command starts training.
python run_training.py --config <existing-config.yaml> --name <new-run-name> \
  --loss_config.rank_regularization '{enabled: true, target_rank: 4, weight: 0.01, eps: 1.0e-12}'
```

Register a separate setting/run before any formal submission; never resume an
old run in place with a silently changed objective. Formal runs still go through
`scripts/submit_slurm.sh`. The registered Hex pair inherits the Hex MEAN snapshot
and best-val policy; the method alone does not change checkpoint retention.

## Exact objective and gradient path

For each page, take the actual native quantized vectors used in that forward,
`Q` of shape `[fragments, native_vq_dim]`, before `project_out`. Center over
fragments, not across pages, then compute:

```
C = centered(Q).T @ centered(Q) / (fragments - 1)
r_eff = trace(C)^2 / (sum(C^2) + eps)
rank_loss = mean_pages(relu(1 - r_eff / target_rank))
total_loss = original_total_loss + weight * rank_loss
```

For wide vectors, use the smaller fragment-by-fragment Gram matrix, which has
the same nonzero eigenvalues. No SVD/eigendecomposition is required. The hinge
is applied to each page BEFORE averaging; pooling pages could hide a rank1
representation per page behind different style-dependent directions.
Calculations use float32 with autocast disabled (float64 inputs stay float64).
No content/style labels, auxiliary head or extra model forward are used.

`CSAE.forward(..., return_native=True)` appends a seventh native-VQ tensor;
default calls still return the historical six values. A temporary pre-hook on
`vq.project_out` captures the quantizer's actual STE tensor and is removed in
`finally`. This also corrects `quantize_with_native` so it does not reconstruct
codes by looking up indices after the EMA update. Looking up after forward can
otherwise retrieve moved or replaced atoms instead of those used by the decoder.

Rank gradients follow STE into encoder/project_in. The EMA buffer remains
non-trainable and follows encoder assignments through the unchanged EMA update.
There is still only one assignment/EMA update per forward. State-dict keys,
decoder inputs, BN behavior, V3 ratios/components and commit loss are unchanged.
The original six-output path is unchanged when disabled.
During gradient-enabled eval (such as downstream adaptation with frozen BN),
the native interface retains its previous STE capability by attaching the
captured hard value to project_in; the library only attaches STE in train mode.
Normal inference mode does not create this extra gradient path.

The theoretical target bound is `min(native_dim, fragments-1, n_atoms-1)`.
Trainer, Tester and health evaluation validate this from config; each rank
calculation also validates the actual tensor's fragment/dimension bound. Do not
lower the target to the *currently* active code count: that would reward collapse.

## Outputs and resume

Training, validation and health evaluation include:

- `rank_loss`: unweighted mean page hinge;
- `weighted_rank_loss`: actual additional contribution to total loss;
- `rank_effective_rank`: mean per-page participation-ratio dimension;
- `rank_target`: configured target;
- `rank_trace`: mean total centered variance, for scale diagnostics;
- `rank_active_fraction`: fraction of pages below target.

The training fields go into `loss_history.csv`, `loss_epoch_history.csv`,
`loss_curves.png`, console logs and existing WandB logging. Plot cadence uses
`plot_every_n_epochs`. Health evaluation includes them in its `losses` JSON.
Training histories retain the existing equal-batch aggregation convention;
health evaluation uses page-weighted aggregation. These are **page-sampled**
statistics, not an unweighted spectrum of every atom in the full codebook.

No new checkpoint tensors or schedule state are necessary. Save/reload the
run's config alongside its checkpoint as usual. Historical CSVs copied into a
new resume run are expanded atomically, keeping old rows and leaving unavailable
rank values blank. Disabled configurations retain the old CSV schema. A changed
objective makes total-loss values across regimes non-comparable; use a separate
run and reset label-free best-loss bookkeeping if deliberately changing it.

## Limitations / experiment gate

This implements ONLY the requested rank floor: no total-variance penalty,
isotropic whitening, cosine VQ, jitter, warmup or additional style invariance.
Epsilon slightly breaks scale invariance near zero. At constant codes the loss
is1 and all statistics/gradients are finite, but missing directions have zero
gradient at exact collapse. Even a nonconstant exact line can be stationary in
missing directions. Prefer preventing collapse early; this is not a guaranteed
rescue of an already collapsed checkpoint. `rank_trace` exposes scale shrinkage.

More geometric dimensions need not carry useful content. Per-page centering
removes constant page offsets, not all style leakage. A future controlled test
must retain macro/weighted purity, usage, reconstruction and content/style probes,
then test the same small addition model. Never select checkpoints using test
labels or effective rank alone; existing macro-best selection remains unchanged
(its total-validation-loss tie-break includes the new term).

No formal efficacy claim follows from unit tests or smoke training.

## Verification (2026-10-08)

GPU Slurm job222824 on `ws-l1-015` exited0 and released its allocation. Python
compilation and31 tests passed:11 new rank formula/gradient/AMP/native-EMA/logging
tests,1 real-Hex training/resume/health smoke, and19 existing native-interface,
V3-monitoring, Hex/PhoneNums protocol and resume-artifact regression tests.
Log: `slurm_logs/20261008_v3_rank_verify2.log`.

The smoke used32 train pages and4 validation pages at batch4, one initial epoch
plus one resumed epoch, then a16-page style-stratified health check. Verified
finite rank fields, exact-once weighted loss inclusion, inherited CSVs, strict
checkpoint compatibility, loss/codebook/V3 CSV+PNG output and at most2 checkpoints
per run. These small debug subsets are not formal acceptance results. The exact
newly created `logs/smoke-v3-rank-*` temporary directory was deleted after tests.
Forward comparisons allow float32 GPU k-means reduction roundoff while checking
assignments and RNG; disabled loss outputs are checked for exact equality.
