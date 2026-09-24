# S8 acceptance — transient BN semantics, early GN collapse

Accepted2026-09-14. No training was restarted or submitted during acceptance.

## Operational outcome and protocol

Both runs completed all200 epochs (0–199), not crashes or timeouts:

| Arm / Slurm job | Training node | Finished (Asia/Dubai) | Exit | Checkpoints |
|---|---|---|---|---|
| BN /191621 | ws-l5-012 |2026-09-14 06:11:24+04|0|11|
| DGN8 /191622 | ws-l5-013 |2026-09-14 06:06:34+04|0|10; no health-eligible macro-best|

Evidence: final epoch199 metrics/current and snapshot files plus job exit traps;
neither job remains in squeue. Slurm accounting database is unavailable, so do
not claim a separately retrieved sacct COMPLETED state. Stderr has only the
existing GradScaler deprecation warning, no model exception.

GPU acceptance Job192130 and follow-up routing Job192132 ran on `ws-l1-005`,
exited0 and released their resources. An initial acceptance invocation192129
failed on an import typo in the new analysis script before any evaluation;
corrected and rerun, unrelated to the two successful training jobs.

All five available checkpoint roles were evaluated on the unchanged full test:
2600 pages,67,600 fragments,325 per(content,style), manifest SHA256
`d455680108040b20d4880aff2bba29b4de0878f97cb0d37fd10ca91b8f1dc5b6`.
Use normal eval, frozen VQ, checkpoint's saved decoder epoch, and run-local
config. Macro-best39 therefore uses **page-mean** decoding, not an accidental
post-warmup evaluation mode. Checkpoint/config/data hashes are in each report.

Artifacts under each run's `acceptance_20260914_s8/` include health JSON,
paired26x16 reconstruction/mask PNG, balanced column-normalized confusion
PNG/SVG/CSV/JSON, Hungarian mapping and trajectory summary. No historical
training plots/configs/checkpoints were overwritten or deleted.

## Full-test results

| Arm / selection / epoch | Macro | Weighted | Hungarian | Active | Perplexity | Coverage | Recon | Foreground RGB MAE | Chroma RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BN macro-best39 |.90738|.79728|.79728|26|23.505|25|.044494|.18771|.15554|
| BN best-val197 |.39122|.19541|.19244|24|11.834|17|.007049|.05240|.03961|
| BN current199 |.37757|.21294|.20112|25|14.089|16|.007959|.05451|.04290|
| DGN8 best-val30 |.28897|.05740|.05740|7|2.490|7|.080767|.23720|.15541|
| DGN8 current199 |.15144|.05652|.05652|8|1.935|8|.072657|.21516|.13780|

BN39 **passes the preregistered aggregate checkpoint gate** (macro.75,
active26, perplexity20, coverage24, recon.30). It is not a full disentanglement
success: colors fail, one style collapses, and the state does not survive
ordinary fragment decoding. Do not rewrite the registered gate retroactively.
BN current and both available GN endpoints fail it. GN has zero eligible
macro-best epochs; its missing macro-best is correct policy behavior.

### Important per-style failure hidden by macro

Full-test per-style counting confirms at BN39:

- All8450 **blue** fragments (every letter,325 pages) select **code19**.
  Per-style perplexity1, global-mapping accuracy1/26=.03846.
- The other seven styles each use25 codes, with shared-global-mapping accuracy
  .90130–.90911. This is a real cross-style semantic structure, not wholly a
  metric artifact, but blue has completely failed.
- Code19 carries1/8 of the data, yet gets only1/26 of macro's code-wise weight.
  Hence macro.907 overstates uniform sample/style success; weighted/Hungarian
  .797 and per-style metrics expose the limitation. Do not discard blue samples.
- Visual inspection: BN39 produces mostly gray-brown glyphs for non-blue styles
  and an almost fixed blue glyph for blue input. BN199 restores many source
  colors/shapes despite weak VQ mapping. DGN8 current produces an almost fixed
  brown glyph for most styles, with blue relatively better reconstructed.

At DGN8 current, code16 receives99.63–99.86% of black/green/red/purple/orange/brown
fragments and87.33% of teal. Its global-mapping accuracy is near1/26 on those six
styles, versus.14379 on blue. Normalization has not rescued healthy semantics.

## Training trajectory and mechanism

### BN: a state was learned, then lost after release

First registered stable8-of-10 window: **epochs35–44**. Qualifying windows start
at35–43; none starts at50 or later. There were113 relaxed macro-health-eligible
epochs, but eligibility is not the stronger stable-phase criterion.

| Epoch | Style input | Val macro | Val Hungarian | Active | Perplexity |
|---:|---|---:|---:|---:|---:|
|39|page mean|.9055|.7953|26|23.48|
|49|page mean|.8610|.7719|26|23.79|
|50|fragment|.8143|.6821|24|20.82|
|55|fragment|.6541|.5698|25|22.29|
|74|fragment|.5075|.3360|22|16.27|
|199|fragment|.3769|.1994|25|14.06|

This is progressive deterioration around/after release, not a one-epoch crash
or never-learning. Compared with S7 BN/W50 lr1e-3 (no stable window, best test
macro.44149), lr1e-4 enables much better early alignment in this seed. It does
not establish durable colored disentanglement or multi-seed reliability.

Frozen routing probe: fixed8 test pages,1/style, cyclic within-page permutations,
normal eval, no updates, state equality checked. These are diagnostics, not
full-test acceptance scores or in-distribution causal effect estimates.

| BN checkpoint | Original recon | After permuting content codes | After permuting fragment style | Replace style with page mean |
|---|---:|---:|---:|---:|
|39|.04599|.12207|.04599|.04599|
|199|.00767|.12138|.21765|.13872|

At39, style permutations have no meaningful effect **by construction** of
page-mean decoding; this does not prove the raw style vector lacks content.
At199, fragment-style permutation strongly changes reconstructed glyphs, while
content-code permutation also hurts: the endpoint uses **both pathways**, not
an established completely ignored VQ pathway. The validation whitened
style-to-content linear probe is.993 at39 and.907 at199; raw values are only
.301/.153, illustrating the scale sensitivity of unwhitened probes. Decodable
style content already exists during warmup, then becomes accessible to the
decoder after release. Association plus endpoint intervention supports renewed
cross-factor entanglement, but a no-release continuation control is still needed
to identify the training-time cause. Color recovery is not semantic recovery.

At BN197/199 the full-test V3 hinge sum is0 despite Hungarian near.20. The
implementation (`model/v3_loss.py`) constrains aggregate pairwise-distance
ratios, not label-conditional information leakage or the worst style's purity.
Thus low/zero V3 does not certify semantic disentanglement. The recorded
validation V3 at199 is.02345, not0; ratios are batch-composition-dependent, so
do not conflate test and validation statistics or average their hinge values.

### GN: early poor solution, not a switch-induced failure

At epoch0, validation active5/perplexity1.59/Hungarian.0438; whole-run maximum
Hungarian is only.08561 (epoch158). No stable window, no healthy macro-best.
The unconstrained macro peak.556 at12 has weighted purity only.0554 and
perplexity1.73; it is not successful learning.

GN removes decoder running statistics as designed; the decoder-only BN probe
is an exact no-op. Yet normal reconstruction remains poor and often resembles
a fixed average glyph. Post-release val recon median/p95 is.07294/.07991 versus
BN.03459/.08397. A smoother loss curve is not a successful model.
Fixed32-page current encoder-BN intervention changes90.0% of codes and raises
perplexity1.88→6.78, but fixed-global-mapping accuracy remains.05649 and recon
.07350→.07356. Encoder-mode sensitivity remains; this is not evidence that
changing encoder normalization will necessarily fix training.

BN mode pathology is reduced, not eliminated at low lr. Its largest val recon
is.94870 at180 (S7 W50 previously reached46.74). The fixed32-page decoder-BN
intervention at180 changes recon.97827→.01418 with exactly unchanged codes.
At the healthy BN39 endpoint all four BN modes have recon approximately.0458;
the learned early semantics are not merely an eval-mode reconstruction artifact.

## Decision and next experiment proposal — not submitted

- Archive both as operationally completed. BN: transient aggregate semantic
  success, failed color/per-style uniformity and persistence. GN8: scientific
  negative at this configuration/seed. Do not blindly resume either current199.
- Keep decoder BN/lr1e-4 as the more promising branch. Do not prioritize wider
  decoder-GN sweeps just because GN has no train/eval running-stat gap.
- Best next controlled pair: branch **both from the same BN snapshot49** with
  optimizer/scheduler/scaler restored and a shared, explicitly seeded continuation
  data/RNG stream. Current checkpoint payloads do not save RNG state, so do not
  claim exact continuation of the original sampler stream. One keeps page-mean permanently;
  one releases the fragment residual gradually (e.g.50 epochs):
  `style_in = page_mean + alpha(epoch) * (z_s - page_mean)`, alpha0 versus0→1.
  Preserve all other settings and equal training budgets; define the final
  alpha=1 observation window before submission. Existing S8 is the abrupt-
  release historical reference, not a same-continuation-RNG control.
- This proposal tests maintenance and release, **not a guaranteed color fix**.
  BN39/49 already lose colors and blue semantics. Track full per-style mapping,
  weighted purity and colors alongside aggregate macro; add these diagnostics
  before another run, without silently changing historical scores/gates.
- If mean-only keeps high purity but fails colors/blue, isolate style-scale/
  shared-style representation next. If both matched branches degrade, revisit
  encoder/VQ/EMA rather than attributing everything to release. No new model,
  optimizer or normalization change is authorized by this acceptance request.

Scripts: `scripts/accept_s8.py`, `scripts/diagnose_s8_routes.py`.
Training runs and source hashes: [S8 protocol](20260913_S8_DECODER_NORMALIZATION.md).
