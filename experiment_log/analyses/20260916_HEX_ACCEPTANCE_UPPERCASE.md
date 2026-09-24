# Hex acceptance and uppercase replication — 2026-09-16

## Full-test acceptance

HEX-S1 REPO193731 and MEAN193733 completed epoch199/500,000 steps with exit0,
at2026-09-16T01:45:03+04 and01:52:32+04. GPU acceptance196431 on ws-l1-005
finished normally. Six checkpoint roles use full10,000-page test/160,000 fragments,
exactly1250 per(content,style). All11 checkpoints in each source run are retained.

| Arm / role | Epoch | Macro | Weighted | One-to-one | Perplexity | Recon | FG chroma RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| REPO macro-best |192|1.000000|1.000000|1.000000|16.000|.026380|.062881|
| REPO best-val |122|.502787|.472719|.451875|15.431|.005892|.033225|
| REPO current |199|.999732|.999731|.999731|16.000|.069236|.150108|
| MEAN macro-best |171|1.000000|1.000000|1.000000|16.000|.023599|.055699|
| MEAN best-val |171|1.000000|1.000000|1.000000|16.000|.023599|.055699|
| MEAN current |199|.999938|.999938|.999938|16.000|.025210|.061654|

All roles have16 active codes; coverage16 except REPO best-val122, which has12.
Both best/current pairs pass the registered Hex semantic gate, so these are not
only transient best-checkpoint successes. Both current states align all8 styles:
minimum per-style accuracy=.99900(REPO)/.99975(MEAN), using one global mapping.
Current digits/a-f accuracies: REPO=.99997/.999333; MEAN=.99990/1.00000.
No label rematching by group/style and no test-based checkpoint selection.

All six role-specific health reports, hashes, paired16x16 reconstruction grids,
masks,16x16 balanced confusion SVG/PNG/CSV/JSON and acceptance indexes are in
each source run's `acceptance_20260916_hex/`. MEAN best and best-val are separate
retained files with the same epoch171 model; both roles are explicitly reported.

Success is semantic, not complete disentanglement. MEAN current grid preserves
eight colors well. REPO current shows color substitutions (e.g. black→brown,
red→purple), and its chroma error is2.43x MEAN's. Fixed-batch eval/train recon
ratio=2.511(REPO)/1.192(MEAN). Whitened raw z_s→content validation readouts remain
.78662/.59058 (chance.0625), despite code→style readout=.125 (chance.125).
Thus high VQ purity does not prove raw style contains no content or that ordinary
fragment decoding is equivalent to page-context decoding.

## Temporal evidence: compare steps, not just epochs

Registered stable windows require8/10 epochs with macro and Hungarian>=.70,
all16 codes active, perplexity>=13 and coverage>=15.

- REPO first qualifying window158–167, approximately397,500–420,000 steps.
  At epoch51/130,000 steps, its validation Hungarian was only.22124; at149 it
  was.47516, then171=.99260. The final success emerges well after the old
  uppercase run's entire130k-step budget. Early epoch0/1=.804/.874 did not persist.
- MEAN first window3–12, approximately10,000–32,500 steps. At130,000 steps it
  already achieved.99368. Permanent mean supports much earlier semantics in this
  matched seed, but not uninterrupted monotonic improvement.
- Last50 mean Hungarian=.85335/.95359; validation recon exceeded1 in7/6 epochs.
  Neither run is completely free of BN/optimization transients.

RQ2-H21 is supported in this seed: the REPO bundle works on Hex16. RQ2-H22 is
supported for earlier alignment and better current color/BN behavior, not for a
necessary condition (ordinary REPO also eventually succeeds).

## Why old uppercase W0 is not a class-count-only comparison

GPU audit196435 on ws-l1-013 compared saved config/source snapshots and actual
inputs, not just current config names. Evidence:
`logs/diagnostics/20260916_hex_uppercase_audit/comparison.json` and source diffs.

| Property | Uppercase S7 W0 | Hex REPO |
|---|---|---|
| Alphabet / fragments / atoms |A-Z /26/26|0-9a-f /16/16|
| Training pages |20,800|80,000|
| Steps/epoch;200-epoch total |650;130,000|2500;500,000|
| Render font size |25px, fit-to-bbox with2px margin|48px, original NumberOffice placement|
| Estimated foreground fraction |.14795|.29888|
| Fragments per training batch |832|512|
| Fragment presentations total |108.16million|256million|

Foreground fractions use fixed256 validation pages,32 per style, observed-input
mask estimation, not clean glyph masks or an unbiased causal estimator. Both
loaders produce RGB and match manual [-1,1] normalization/slicing. No grayscale
conversion was found. Font file hash is identical; rendered scale/layout is not.

C512/S512/nativeVQ512, PhoneNums ResNet/BN, batch32/seed0/float32, AdamW lr.001
and wd.1, all V3/recon weights, EMA.98/dead16 and relativity15 match. Archived
backbone module, V3 loss and optimizer/scheduler utility sources are byte-identical.
Autoencoder additions only configure normalization; explicit `batch` is a no-op
and tests verify the same initialized tensors. Trainer changes add detached
diagnostics/provenance, not a new loss or optimizer. Health gates scale with K
and affect retention only, not training; they cannot explain current-state gaps.

Most credible explanations, in order of testability:

1. **Update budget / timing:** strong direct evidence above. Same200 epochs means
   3.846x different optimizer exposure. The epoch-based smooth scheduler also
   differs per optimizer step: lr(e)=.001*.98**(e/20); at about130k steps Hex has
   factor~.950 while uppercase has~.818. More steps alone is not yet proven to
   rescue uppercase, but must be separated from class count. Budget alone also
   cannot explain everything: permanent MEAN already aligned Hex within~30k steps.
2. **Rendering/objective balance:** uppercase glyphs occupy about half the pixel
   area. All-image MSE can place less aggregate weight on glyph errors, while
   V3 hinge weights stay fixed. This is a plausible optimization-basin difference,
   not evidence that the correct remedy is blindly increasing recon weight.
3. **More distinctions and different statistics:** K/N26 versus16 changes VQ
   geometry, within-page pair statistics, batch-BN sample count and k-means/EMA
   dynamics. Mean pairwise distances are averaged, so there is no simple automatic
   26/16 multiplier on V3. Uniform expected per-code usage remains32 in both,
   hence dead16 has the same nominal ratio, though nonuniform tails can differ.
4. **Early trajectory sensitivity:** trainable parameter hashes actually match
   across K16 and K26 (`206f3333...969e9`), but codebook buffer shapes and first-batch
   k-means inputs differ. Same initial backbone is not the same optimization
   trajectory across datasets. The exact Hex smoke and formal runs themselves
   diverged numerically across GPU nodes despite matching initialization/order
   hashes; this remains a reproducibility caveat, not proof of a hidden bug.

Old uppercase current199 has Hungarian.11337, active21/perplexity4.25, recon.13933,
and fixed-batch eval/train ratio5.093. It is actual assignment/usage collapse plus
normalization sensitivity, not a plotting error. No single causal explanation is
established by these cross-dataset runs.

## S10 requested reruns (registered default)

`VVI-RQ2-S10 — Hex-Protocol Uppercase Replication`: original UppercaseLettersV2
manifest unchanged; C512/S512/nativeVQ512/K26, original BN, exact HEX-S1 loss,
optimizer/scheduler/EMA and diagnostics. Only decoder mode differs:

- `VVI-RQ2-V2-C512-K26-HEXREPO-S0`: ordinary fragment decoding throughout.
- `VVI-RQ2-V2-C512-K26-HEXREPO-MEAN-S0`: permanent page mean, no release; raw z_s V3.

Default200 natural epochs0–199/130,000 steps, scratchseed0/batch32/float32.
This deliberately retains the existing data and budget while repeating W0 with
current verified code and introducing a permanent-mean comparison. It does NOT
isolate class count from Hex and does NOT test the500k-step explanation. A user
choice was requested whether to extend to~770 epochs and step-align scheduler;
no implicit data regeneration or oversampling is authorized by this default.

Configs: `configs/uppercase/rq2/v2/cfg_vvi_rq2_v2_c512_k26_hexrepo[_mean]_seed0.yaml`.
Macro health gate remains uppercase active>=24/ppl>=18/coverage>=18/weighted>=.20.
Full2600-page test gate macro>=.75,active26,ppl>=20,coverage>=24,recon<=.30;
report weighted/Hungarian/per-style/color/probes, and persistence separately.
Stable8/10 window uses uppercase macro>=.70/active26/ppl>=20/coverage>=24;
also report Hungarian, do not select checkpoints using it. Retain current,
macro-best,best-val+8 every25 snapshots (max11). Two ws-ia/1GPU/32GB/4CPU/24h jobs
after complete650-step train/full2600-page validation smoke. No old-run continuation,
no normalization/loss changes, no commit/push.

### Implementation/preflight

No training-library or model/loss implementation change was needed for S10.
New files are two configs, acceptance/audit/smoke scripts and protocol tests;
existing historical data/checkpoints/configs are preserved.

GPU Job196453 on ws-l1-008 passed12 focused tests (2 S10 protocol,6 normalization
and detached-monitor regressions,4 BN/usage diagnostics), then both complete
650-step/20,800-page train epochs with full2600-page/67,600-fragment validation.
Nine CSV/PNG diagnostic families, strict checkpoint reload,512-page CLI/health
evaluation,26x8 actual counts325, numerical/plot outputs, and model/optimizer/BN/
VQ/RNG neutrality passed. Sampled scale count is7 (100,200,...600,and last650).
Both had3 saved checkpoints (current,best-val,snapshot0); macro-best is correctly
absent when epoch0 misses its health gate. Low first-epoch semantic scores are
not an infrastructure failure or evidence to change the registered hypothesis.

GPU Job196476 reran the portable2-test protocol suite and confirmed both arms'
initial state, sampler order and BN-page fingerprints match each other and old
S7 W0 exactly. Reports: `logs/diagnostics/20260916_s10_preflight/`.
Both reconstruction/matrix grids were visually checked. Each652MiB smoke
directory was then deleted, about1.27GiB total; neither is a formal ledger entry.

- Initial full-state SHA256: `11d6c4f25a85967b86447086f069f4f3e1a515b392ef392037619fd5c04d37a7`.
- First-epoch sampler: `0f69d3ac14e130a19b5ee2ef1327c41a45d77e6bde2a1ddc50af8549291ecff1`.
- BN pages: `a478bbec44f89c4f163b41794ffaa0cf59760cb387be17b6e2f36bb21318ce03`.
- REPO config: `64bd95d37f344c2e2dc30ccea6c650ed7f05f681062e3e766eb1978e5d3bfe36`.
- MEAN config: `d7a01a2ae3b5afc02ed975c1062eb4241f67624f0facda7c89316ae0996621b7`.
- Data manifest: `d455680108040b20d4880aff2bba29b4de0878f97cb0d37fd10ca91b8f1dc5b6`.

Pending a user choice about extended exposure, the registered default remains
200 epochs; do not describe these as500k-step-matched experiments. Submit only
after registering ACTIVE and preserve code/config/data provenance per run.

### Formal startup

No budget response arrived before submission; both use the explicitly stated
200-epoch default, without oversampling or altered LR schedules. Registered at
21:57 before sbatch; started2026-09-16 21:58 Asia/Dubai:

| Run name | Job | Node | Slurm start |
|---|---:|---|---|
| 20260916-2157__VVI-RQ2-V2-C512-K26-HEXREPO-S0 |196481|ws-l1-009|21:58:19|
| 20260916-2157__VVI-RQ2-V2-C512-K26-HEXREPO-MEAN-S0 |196482|ws-l1-014|21:58:25|

Both RUNNING, ws-ia/1RTX5000 Ada GPU/32GiB/4CPU/24h, CUDA visible. Startup
verification on ws-l1-009 confirms exact submitted/effective config equivalence
(except timestamped name), model/BN-page/manifest hashes, and all submitted
Python/YAML/shell source hashes. Both source.tar.gz checksums equal
`35455a881761a4c0d5568cac71d4cddc1c8b6a6679a2172dcd65701fde8e0308`.
Git HEAD `e0bb0c6b70b5798c0dddc69c6903f9cdf315568a`, dirty, archived per run;
no commit/push. ACTIVE/ARCHIVE are disjoint and smoke directories absent.
See `logs/diagnostics/20260916_s10_preflight/submission_verification.json`.
At22:01 both progressed in training (ordinary epoch0 step538, MEAN epoch1 step230).
No early scientific outcome is inferred from this startup check.
