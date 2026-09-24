# S8 — Independent decoder normalization control

Status update2026-09-14: both completed199/exit0 and are archived. BN has a
transient high-purity checkpoint but color/per-style/persistence failures; GN8
fails early. See [full acceptance](20260914_S8_ACCEPTANCE.md). The submission
status descriptions below document the original launch, not current activity.

User selected the independent normalization experiment after S7 acceptance.
Use the previously proposed matched **lr1e-4 BN versus decoder-GN** pair, not
the two-learning-rate sweep. No historical checkpoint is resumed. Comparison
within S8 changes only decoder normalization; comparison of S8 BN to S7 W50
also changes learning rate and must be described separately.

## Protocol

Setting: **VVI-RQ2-S8 — Low-LR Decoder Normalization Isolation**.
Direction RQ2; H6 (normalization-mode failure), H17 (decoder-GN intervention),
H15 (lower-lr historical reference only).

| Run ID | Encoder | Decoder | AdamW lr | Decoder style |
|---|---|---|---:|---|
| VVI-RQ2-V2-C512-K26-W50-LR1E4-BN-S0 | original BN | original BN |.0001|mean0–49; fragment50–199|
| VVI-RQ2-V2-C512-K26-W50-LR1E4-DGN8-S0 | original BN | GN,8 groups |.0001|mean0–49; fragment50–199|

Both scratch, seed0,200 epochs, C512/S512/nativeVQ512/K26, batch32, float32;
AdamW wd.1 and original exponential scheduler (.98 per20 epochs). Relativity15,
recon/four V3 weights1, commitment.1, EMA.98, dead-code threshold16. No new loss,
schedule, data generation or encoder change. Existing UppercaseLettersV2 manifest:
`d455680108040b20d4880aff2bba29b4de0878f97cb0d37fd10ca91b8f1dc5b6`.
Configs live in `configs/uppercase/rq2/v2/normalization/`.

Decoder GN replaces every BN in the main and shortcut branches. Eight fixed
groups divide all decoder feature widths (16–256); no silent fallback to another
group count. Preserve each layer's eps and affine parameter initialization.
Build the original model first, then replace normalization without drawing RNG;
all shared named parameters, encoder state and initial VQ state match. Full
state hashes intentionally differ because GN does not retain BN running buffers.
This is a scratch architectural intervention, not a function-preserving warm
conversion of trained BN. Old configs/checkpoints remain BN and load strictly.

GroupNorm uses input statistics in both train and eval modes and does not use
batch running estimates. That property motivates the intervention but does not
guarantee color or semantic success.
[PyTorch GroupNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GroupNorm.html).

## Diagnostics / decision gate

Keep loss, codebook, V3 ratio, detached leakage, foreground color and online
usage curves. Keep complete2600-page validation and four BN-mode interventions
on32 fixed pages (4 per style,8 styles). In the GN arm,
`decoder_batch_stats` is explicitly **no-op: no decoder BN**, while `both` only
switches encoder BN. CSV records encoder/decoder BN counts and actual switched
layer counts. This deliberate no-op must not be called measured recovery.
All diagnostic interventions restore model buffers/modes/RNG and never select
checkpoints. Encoder BN can still cause train/eval differences in either arm.

Every25 completed epochs plus0/10/11/12/15/20/50/100: paired reconstruction,
input-only foreground mask, full balanced column-normalized confusion, shared
Hungarian mapping. Current + macro-best + best-val +8 snapshots: at most11 files.
Macro health floor remains active24/perplexity18/coverage18/weighted.20.

Acceptance uses normal full-test2600 pages at macro-best/best-val/current. Compare
reconstruction/color separately from code semantics. Semantic gate unchanged:
macro>=.75, active26, perplexity>=20, coverage>=24, recon<=.30; stable phase is
8/10 epochs with macro>=.70, active26, perplexity>=20, coverage>=24. Report
weighted purity, leakage, coverage and color limitations, not only macro.

For normalization, compare matched validation trajectories (median/tails of
normal-eval recon, prediction out-of-range fraction), and normal full-test
best-val/current recon plus per-style foreground color. GN's zero decoder-mode
gap is a design property, **not** sufficient evidence that actual recon improved.

- GN improves normal recon/stability but not purity: reconstruction pathology
  partly resolved; investigate encoder/style leakage or VQ objective next.
- GN improves both normal recon and stable purity: supports decoder normalization
  as a useful training intervention at this seed/lr, then replicate a seed.
- Both improve relative to S7, little pair difference: lower lr may explain much
  of the improvement; no claim that GN is necessary.
- Both fail with strong encoder-mode sensitivity: isolate encoder normalization
  next rather than simultaneously changing V3/EMA/codebook width.

## Operations

GPU regression tests and two short full-validation smoke runs precede formal
submission. Verify trainable-parameter and sampling hashes,49→50 switch,
state/RNG-neutral diagnostics, strict GN reload and run-local-config evaluation.
Delete smoke directories after inspection; retain a lightweight preflight summary.
Formal: two ws-ia jobs,1GPU/32GB/4CPU/12h each,200 epochs. Register ACTIVE before
sbatch; then record Job IDs, nodes, code/config/data hashes. No commit/push.

## Preflight — 2026-09-13

- GPU regression suite: **68 tests passed**, Job191614 on `ws-l5-012`.
- Both one-epoch/two-training-batch smoke runs passed on `ws-l5-013`, with
  complete2600-page validation (67,600 fragments per arm), all seven diagnostic
  CSV/PNG sets, balanced reconstruction/confusion artifacts and checkpoint load.
- Verified49→50 style-mode switch, unchanged model/optimizer/RNG under diagnostic
  interventions, strict GN reload, and standalone512-page style-stratified
  evaluation using the run-local config/current alias.
- Both arms have32 encoder BN modules. BN has28 decoder BN modules; GN has
  zero decoder BN and28 GN modules, including shortcuts. GN decoder-only BN
  intervention is an exact no-op under production deterministic settings.
- Visually checked the no-op diagnostic label. Deleted both smoke directories
  after validation (about1.30GiB total); no formal artifacts removed. Lightweight
  evidence remains at `logs/diagnostics/20260913_s8_preflight/summary.json`.

Matched named-parameter SHA256:
`206f3333b3fab56c6bc5c135abfcd47bf04dc4a31a260ee3ccaf801e26b969e9`.
Matched first-epoch sampler SHA256:
`0f69d3ac14e130a19b5ee2ef1327c41a45d77e6bde2a1ddc50af8549291ecff1`.
Matched fixed diagnostic page SHA256:
`a478bbec44f89c4f163b41794ffaa0cf59760cb387be17b6e2f36bb21318ce03`.

| Arm | Submitted config SHA256 |
|---|---|
| BN | `32376a2ca22e7a4449961f7f65d3d7fe5db9a1b697decab7103b516d8728084a` |
| DGN8 | `b20665bc5cd64b216a91ea29314b4f153050c43c2308c8b5989824adefe911ad` |

Shell syntax and `git diff --check` passed. Code is intentionally dirty; formal
jobs preserve the source archive, per-file hashes, patch, Git state and config
under each run's `reproducibility/`. No commit or push.

## Formal submission

Both registered as prepared before submission; now running:

| Arm | Run directory under `logs/` | Slurm job | Start (+04) | Node |
|---|---|---|---|---|
| BN | `20260913-2309__VVI-RQ2-V2-C512-K26-W50-LR1E4-BN-S0` |191621|2026-09-13 23:10:12|ws-l5-012|
| DGN8 | `20260913-2309__VVI-RQ2-V2-C512-K26-W50-LR1E4-DGN8-S0` |191622|2026-09-13 23:10:18|ws-l5-013|

Slurm confirmed each allocation: ws-ia,1GPU,4CPU,32GB,12h; both GPUs are
RTX5000 Ada32GB, CUDA visible device0. Logs show actual epoch0 training updates
on both nodes. End-of-allocation limits are2026-09-14 11:10+04, not guaranteed
training finish times; sbatch releases resources automatically.

Git HEAD `e0bb0c6b70b5798c0dddc69c6903f9cdf315568a`, dirty. Both archived source
tarballs have SHA256
`0a3c5ed637d97c53815c18f268e36c2ca00b8ad29a8f375dc655f55c6978f4e7`.
Submitted YAML backups match source bytes; run-local configs match after only
resolving the timestamped name. Formal initial parameter hashes match each other
and preflight; full BN/GN state hashes intentionally differ due to BN buffers.
Both use the pinned manifest and expected32-encoder/28-decoder normalization
module counts. ACTIVE/ARCHIVE are disjoint; no smoke directories remain.
