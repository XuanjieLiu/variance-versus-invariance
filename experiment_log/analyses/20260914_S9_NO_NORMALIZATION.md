# S9 — Decoder-only versus all-backbone normalization removal

Status update2026-09-15: both completed199/exit0 and are archived; no continuation.
Six full-test roles and figures are saved. See [S9 acceptance](20260915_S9_ACCEPTANCE.md).
Submission-state statements below are historical, not current job status.

User approved the no-normalization pair on2026-09-14 after S8 acceptance.
This supersedes the proposed snapshot49 mean/residual continuation for the next
two GPU slots. No S8 checkpoint is resumed or interpreted with a new architecture.

## Registered protocol

Setting **VVI-RQ2-S9 — Normalization Removal Isolation**, Direction RQ2,
hypotheses H19 (decoder Identity), H20 (additional encoder Identity); H6 mode
diagnostic and H18 release reference.

| Run ID | Encoder | Decoder |
|---|---|---|
| VVI-RQ2-V2-C512-K26-W50-LR1E4-DNONORM-S0 | original BN | Identity replaces all28 BN modules |
| VVI-RQ2-V2-C512-K26-W50-LR1E4-ALLNONORM-S0 | Identity replaces all32 declared BN modules | Identity replaces all28 BN modules |

The encoder includes two unused declared BN1d heads; removing just those would
not change forward computation. All actual CNN/shortcut BN layers are removed
in All-NoNorm. Neither variant introduces GN/LN/IN, weight standardization,
Fixup, residual rescaling, gradient clipping or a new output activation.
Build the entire historical model first, then replace norms without RNG draws.
Shared Conv/Linear/VQ parameters match; full parameter/state hashes differ
because removed BN affine parameters/buffers are absent. Record a separate
`non_normalization_parameters_sha256` for this comparison.

Both scratch seed0,200 epochs0–199,C512/S512/nativeVQ512/K26, batch32,float32,
AdamW lr1e-4/wd.1, exponential scheduler. W50 means page mean0–49, ordinary
fragment style from50. Relativity15, recon/four V3 weights1, commit.1,
EMA.98/dead threshold16 unchanged. Preserve fixed RGB[-1,1] input scaling.
Use existing UppercaseLettersV2 manifest SHA256
`d455680108040b20d4880aff2bba29b4de0878f97cb0d37fd10ca91b8f1dc5b6`.
Configs in `configs/uppercase/rq2/v2/normalization/`, suffixes
`dnonorm_seed0.yaml` and `allnonorm_seed0.yaml`.

The pair isolates the extra removal of encoder BN conditional on decoder
Identity. S8 BN is the historical decoder-normalization reference. Analyze both
pre/post50 periods; norm removal does not itself remove fragment-style leakage.
Normalization-free instability would motivate a separately registered stabilized
variant, not prove that normalization is inherently necessary.

## Diagnostics and acceptance

Existing loss/codebook/V3 ratios/leakage/colors/online usage/BN diagnostics remain.
Periodic recon/mask/balanced confusion every25 completed epochs and additional
0/10/11/12/15/20/50/100. Retain current, health-gated macro-best, best-val and8
permanent snapshots (max11). Existing macro-health/acceptance thresholds unchanged.

New opt-in diagnostics in both arms:

- Full validation per-style code counts reuse the existing forward/indices.
  Check325 samples for every(content,style), use one global Hungarian mapping
  for all eight styles. Log macro, weighted purity, fixed-global-mapping accuracy,
  active/perplexity/coverage and largest code fraction. Save per-epoch raw counts
  and mapping JSON, `per_style_codebook_epoch_history.csv`,
  `per_style_codebook_metrics.png`. No per-style rematching in the main accuracy.
- Sample training activation/latent RMS and max magnitudes and encoder/decoder
  gradient L2/max every100 global steps plus the final step of each epoch.
  Gradients are observed after GradScaler.step unscales, before zero_grad.
  Epoch summaries average **sampled** steps, not the entire epoch. Save step/
  epoch optimization CSV, `optimization_scales.png`, protocol and finite flags.
  These detached observations do not change activations/gradients or draw RNG.
- Decoder-only BN mode is a no-op in both runs. All four modes are no-ops in
  All-NoNorm; plot labels explicitly state this, not empirical recovery.

Normal full-test macro-best/best-val/current acceptance uses2600 pages. Original
aggregate semantic gate: macro>=.75, active26, perplexity>=20, coverage>=24,
recon<=.30; stable window8/10 epochs with macro>=.70 and the same usage floors.
Report colors and **worst style** alongside aggregate results; do not call a
blue-collapsed or colorless state full disentanglement success. No new arbitrary
per-style numeric gate replaces the registered one. Norm absence or a smooth
loss curve is not success. Label-based diagnostics never enter training loss.

- Decoder-NoNorm succeeds, All-NoNorm fails: encoder BN is beneficial under this
  protocol; do not claim decoder normalization is required.
- All-NoNorm improves both semantics and colors: supports the full removal in
  this seed; replicate before a general claim.
- Early learning followed by post50 loss: norm removal is insufficient for
  maintenance; return to H18 matched style-release branches.
- Either is numerically unstable: inspect scale/gradient histories and predeclare
  initialization/residual stabilization before another run, not a silent retry.

## Operations

All Python/testing/evaluation/training only through Slurm on ws-* compute.
GPU regression suite and a complete1-epoch training/full-validation smoke in
each arm precede formal submission. Verify shared parameter/sampler/probe hashes,
49→50 switch, strict state reload, all artifacts and state/RNG-neutral diagnostics.
Delete smoke immediately after validation; preserve lightweight summary outside.
Formal resources each: ws-ia,1GPU,32GB,4CPU,12h. Register ACTIVE before sbatch,
then fill job IDs/nodes/config and source hashes. No commit/push.

## GPU preflight accepted (2026-09-14)

Job192257 on ws-l1-005 exited0: all74 regression tests passed (13.098s).
Both arms completed a full650-step training epoch and full2600-page validation
(67,600 fragments). Verified nine CSV/PNG diagnostic families, reconstruction/
balanced matrix/raw counts, finite sampled activation/latent/gradient scales,
49→50 style switch, strict checkpoint reload, checkpoint retention and
model/optimizer/RNG-neutral diagnostics. All-NoNorm standalone512-sample
style-stratified current-checkpoint evaluation also passed. This is software
acceptance, not scientific success; epoch0 purity is not a decision gate.

An earlier temporary smoke had a resume-only logging block misplaced relative
to monitor initialization; it was fixed, only that temporary job was stopped,
then the full final suite above was rerun. No formal run used that version.
Both validated smoke directories were deleted (about1.27GiB total); lightweight
summary remains in `logs/diagnostics/20260914_s9_preflight/summary.json` and
preflight stdout/stderr in `slurm_logs/s9-preflight-192257.*`.

Shared non-normalization initialization SHA256:
`3bf4c7b643ae437913199556ecf1591534390004782dc7f56739b2ecdac8bf0e`.
Shared first-epoch sampler SHA256:
`0f69d3ac14e130a19b5ee2ef1327c41a45d77e6bde2a1ddc50af8549291ecff1`.
Shared fixed32-page BN diagnostic SHA256:
`a478bbec44f89c4f163b41794ffaa0cf59760cb387be17b6e2f36bb21318ce03`.
Config SHA256 Decoder-NoNorm:
`5ee58f49da9ecf83fddf50bf98fa979b41a323e13b6f2d8767f038cfcbb88be0`;
All-NoNorm:
`c3bd1a03ba02d2db2bd3b2f829f940dcf56ffb7260598dde6c1143113e594411`.

Two formal runs registered in ACTIVE before submission, with prefix
`20260914-1125__`:

| Arm | Job | Node | Slurm start (Asia/Dubai) | Time limit |
|---|---|---|---|---|
| Decoder-NoNorm | 192280 | ws-l1-006 | 2026-09-14 11:25:51 | 12h |
| All-NoNorm | 192281 | ws-l1-011 | 2026-09-14 11:26:03 | 12h |

Both RUNNING and actual training steps observed; CUDA device0, NVIDIA RTX5000
Ada32GB, each1GPU/32GB host memory/4CPU. The allocation releases automatically
on exit. No other-project job or concurrency limit was changed.

Both source snapshots are identical SHA256:
`318526081e55742756b8c35d096497d03195b7d44920ab9c152e5f692d817ba6`.
HEAD `e0bb0c6b70b5798c0dddc69c6903f9cdf315568a`, dirty. Each run retains
`reproducibility/source.tar.gz`, source-file hashes, Git status/diff and submitted
config. Subsequent ledger status updates are outside these start-time snapshots.
Compute-node read-only submission acceptance passed: submitted config bytes
equal source; run-local config differs only in timestamped name; initial shared
parameters and full-state hashes match the corresponding smoke; encoder BN
counts32/0 and decoder BN/GN counts0/0. No ACTIVE/ARCHIVE duplication and no
remaining S9 smoke artifacts. Full dataset manifest remains pinned as above.
