# Experiment settings

| Setting | Direction | Hypothesis | State |
|---|---|---|---|
| VVI-H0 | RQ1/RQ2 | historical references | completed |
| VVI-S1 | RQ1-A | RQ1-H1 | completed |
| VVI-S2 | RQ1-B | RQ1-H2 | completed |
| VVI-S3 | RQ2; RQ3 source | RQ2-H1 | completed |
| VVI-RQ2-S1 | RQ2 | RQ2-H2 | completed; not supported |
| VVI-RQ3-S1 | RQ3 | RQ3-H1, RQ3-H3 | completed; successful |
| VVI-RQ2-S2 | RQ2 | RQ2-H3 | planned; matched branch pair |
| VVI-RQ2-S3 | RQ2; RQ1-A control | RQ2-H8; RQ2-H9 | R1 CTRL/W100 completed/accepted2026-09-11; MEAN cancelled |
| VVI-RQ2-S5 | RQ2 | RQ2-H9; RQ2-H10 | completed; both fail stable content gate |
| VVI-RQ2-S6 | RQ2 | RQ2-H11 | timeout at198; last complete197; accepted negative |
| VVI-PN-S1 | RQ2; paper replication control | RQ2-H12 | completed; REPO partial reproduction, PAPER semantic failure |
| VVI-RQ2-S7 | RQ2; C512 dimensional control | RQ2-H13; RQ2-H14; RQ2-H6 diagnostic | completed199/accepted; both fail content gate |
| VVI-RQ2-S8 | RQ2; independent decoder normalization | RQ2-H6; RQ2-H17; H15 historical reference | completed199/accepted2026-09-14; BN transient semantics, GN early collapse |
| VVI-RQ2-S9 | RQ2; normalization removal | RQ2-H19; RQ2-H20; H6/H18 reference | completed199/accepted2026-09-15; better recon, mixed semantic codes |
| VVI-HEX-S1 | RQ2; hexadecimal progressive extension | RQ2-H21; RQ2-H22 | completed199/accepted2026-09-16; both semantic gates pass, MEAN earlier/better color |
| VVI-RQ2-S10 | RQ2; uppercase Hex-protocol replication | RQ2-H23; RQ2-H24; RQ2-H25a | originals completed199/130k; R1 running197821/197822 to769/500500 |
| VVI-RQ2-S4 | RQ2 | RQ2-H1; RQ2-H7 | cancelled before start; deferred for BN/data diagnosis |
| VVI-RQ3-S2 | RQ3 | RQ3-H2 | planned; cold-start control |
| VVI-RQ13-S1 | RQ1-B; RQ3 | RQ1-H5; RQ3-H4 | completed; both VVI gates passed, addition probes failed |
| VVI-RQ1A-S1 | RQ1-A; RQ2 observation; RQ3 pilot | RQ1-H7; RQ1-H8; RQ1-H9 | completed; seed-0 K26 threshold bracketed in `(64,128]` |
| VVI-RQ13-S2 | RQ1-A; RQ3 | RQ1-H10; RQ1-H11; RQ3-H6 | completed; C32 failed, C128 transient only |
| VVI-RQ13-S3 | RQ1-B; RQ3 | RQ1-H12; RQ3-H7 | planned; K52-WARM to K128/D128 generalized expansion |
| VVI-RQ4-S1 | RQ4 bridge | V5 RQ4-H1; RQ4-H2; RQ4-H3 | transferred to V5; Teacher A gate passed |

## VVI-RQ2-S10 — Hex-Protocol Uppercase Replication

Requested2026-09-16: copy HEX-S1 BN/C512/S512/nativeVQ512/loss/AdamW.001/EMA.98
protocol onto immutable UppercaseLettersV2,K26/N26. Ordinary fragment versus
permanent page mean; no release. Default200 natural epochs/130k steps, not500k;
budget extension is a separate requested user choice, not silently applied.
Seed0/batch32, max11 checkpoints/current+macro+best-val+25-epoch snapshots,
uppercase health/acceptance gates unchanged, full2600-page validation/test.
See [acceptance, comparison and registered protocol](analyses/20260916_HEX_ACCEPTANCE_UPPERCASE.md).

**User-authorized continuation2026-09-17.** Both originals completed199/exit0,
but current validation macro/Hungarian is only.26572/.21571(fragment) and
.47832/.39036(MEAN). These are validation monitoring results, not new full-test
acceptance. Resume each current199, not its macro-best, for570 incremental epochs
(absolute200..769), giving770×650=500,500 cumulative optimizer updates.
Run IDs append `-R1`; configs append `_resume1.yaml`. Preserve all model/data/
loss/EMA/BN/decoder parameters, optimizer moments, scheduler and scaler; no LR
reset or page resampling. Original epoch-based decay remains: start epoch200
lr=.001×.98^10≈.000817073; near epoch769 lr≈.0004599, versus Hex's≈.0008179
at its last epoch199. This tests longer exposure on the existing trajectory,
not equal LR at equal optimizer step. Old checkpoints lack RNG state: matched
seeded epoch-boundary resume, not bitwise uninterrupted continuation.

Keep original health/content gates and full2600-page validation. Inherit all
completed histories plus actual macro-best/best-val weights. Do not replicate
parent snapshots. R1 retains22 snapshots at224,249,...,749, plus current and two
best roles (max25 physical checkpoints per run); diagnostic grids remain every25
completed epochs. Each ws-ia1GPU/32GB/4CPU/24h (partition maximum). If timed out,
archive that terminal attempt and explicitly resume its latest completed epoch
to the same cumulative target; never call a timeout a completed770-epoch result.
At completion full-test macro-best/best-val/current and compare trajectories
by cumulative step, including stable8/10 windows, per-style accuracy, colors,
usage and BN sensitivity. No retrospective change of semantic acceptance gates.
See [continuation and provenance](analyses/20260917_S10_CONTINUATION.md).

## VVI-HEX-S1 — PhoneNums-to-Hex16 Extension

Accepted2026-09-16: both best/current pass semantic gates; MEAN first stable
window3–12 versus REPO158–167 and better current colors. Six full-test roles
saved; no continuation. Detailed comparison in the S10 note above.

Two matched scratch200-epoch runs: `VVI-HEX-K16-REPO-S0` and
`VVI-HEX-K16-REPO-MEAN-S0`. Independent100k-page HexDigitsV2, lowercase0–9/a–f,
16 fragments/K16, C512/S512/nativeVQ512, original BN and PhoneNums REPO bundle
(lr.001/wd.1, r15, EMA.98/dead16). Only decoder fragment versus permanent page-mean
differs; raw z_s still enters V3.500k optimizer steps each; max11 checkpoints.
Full10k-test gate macro/weighted/Hungarian>=.75, active16/perplexity>=13/coverage>=15,
recon<=.30; stable8/10 window requires macro/Hungarian>=.70 and the same usage floors.
Health-gated macro selection remains distinct from acceptance. Color and grouped
0–9/a–f accuracy reported separately, using the global mapping; no group rematching.
Each1GPU/32GB/4CPU/24h after full-epoch GPU smoke. See
[data/protocol/decision details](analyses/20260915_HEX16_PROTOCOL.md).

## VVI-RQ2-S9 — Normalization Removal Isolation

Approved scratch pair: `VVI-RQ2-V2-C512-K26-W50-LR1E4-DNONORM-S0` (encoder BN,
decoder Identity) and `VVI-RQ2-V2-C512-K26-W50-LR1E4-ALLNONORM-S0` (both Identity).
C512/S512/nativeVQ512/K26, W50, lr1e-4, seed0/200 epochs, existing corrected V2,
original initialization/loss/EMA/optimizer unchanged. Preserve fixed input scaling.
No Fixup, residual rescaling, clipping or GN/LN substitution; all shortcuts included.
Both add detached sampled optimization scales and shared-mapping per-style full-
validation usage/purity. Existing aggregate checkpoint gates unchanged; report
color and worst-style failure separately. Retain current/macro-best/best-val and
25-epoch snapshots, max11. Each ws-ia/1GPU/32GB/4CPU/12h after GPU smoke.
See [protocol and decision gate](analyses/20260914_S9_NO_NORMALIZATION.md).

## VVI-RQ2-S8 — Low-LR Decoder Normalization Isolation

Accepted2026-09-14: BN best39 passes the original aggregate gate (test macro.90738,
1:1.79728) but blue style collapses and colors/persistence fail. GN has no eligible
macro-best, current test1:1.05652. Both finish199/exit0; no continuation submitted.
See [full acceptance and proposed branch control](analyses/20260914_S8_ACCEPTANCE.md).

Matched scratch200-epoch C512/S512/nativeVQ512/K26, W50, seed0, corrected V2,
batch32, float32, AdamW lr1e-4/wd.1 and unchanged exponential scheduler/loss/EMA.
Both share the existing manifest hash
`d455680108040b20d4880aff2bba29b4de0878f97cb0d37fd10ca91b8f1dc5b6`.
Only pair difference: original decoder BN versus GroupNorm8 replacing all
decoder BN including shortcuts; encoder BN stays unchanged. Run IDs end in
`W50-LR1E4-BN-S0` / `W50-LR1E4-DGN8-S0`. Configs in
`configs/uppercase/rq2/v2/normalization/`.

Shared parameter initialization and data order are verified; full model hashes
differ because GN removes running-stat buffers. No trained-BN checkpoint is
converted to GN. Normalization config is saved in each run. GN's decoder-only
BN probe is explicitly a no-op, not evidence of empirical improvement.

Retain S7 diagnostics, early grid epochs and up to11 checkpoints. Normal full-test
semantic gate and stable8/10 window unchanged; separately compare normal-eval
recon/color and validation tails. Lower decoder-mode gap alone is not success.
Each run: ws-ia/1GPU/32GB/4CPU/12h. Pair isolates normalization at low lr; comparing
BN against historical S7 W50 tests lower lr, not normalization. See the
[full protocol and decision gate](analyses/20260913_S8_DECODER_NORMALIZATION.md).

## VVI-RQ2-S7 — V2 C512 Decoder-Warmup Replication

Direction RQ2; hypotheses H13 (wide content/nativeVQ on corrected letters), H14
(50-epoch warmup at C512), H6 (diagnostic normalization-mode sensitivity).

| Run ID | Decoder style | Epochs | Seed |
|---|---|---:|---:|
| VVI-RQ2-V2-C512-K26-W0-S0 | fragment throughout |200 (0–199)|0|
| VVI-RQ2-V2-C512-K26-W50-S0 | page mean0–49; fragment50–199 |200 (0–199)|0|

Configs: `configs/uppercase/rq2/v2/`, suffixes
`c512_k26_control_seed0.yaml` and `c512_k26_meanwarm50_seed0.yaml`.
Existing UppercaseLettersV2 manifest SHA256
`d455680108040b20d4880aff2bba29b4de0878f97cb0d37fd10ca91b8f1dc5b6`;
no regeneration. C512/S512/K26/nativeVQ512 without a projection, original
ResNet/BatchNorm, float32, batch32, seed0 and scratch initialization. AdamW
lr.001/wd.1, original exponential scheduler, relativity15, recon/four V3 weights1,
commit.1, EMA.98/dead threshold16. These already match C128-W50's bundle; do not
introduce an additional loss/optimizer change. Shared initial model, sampler and
fixed diagnostic page hashes are checked in GPU preflight. Cross-task budgets
differ: letters200 epochs≈130k steps, digits≈500k; one seed is not causal proof.

**New opt-in diagnostics.** Online VQ usage reuses training indices; CSV records
100-step windows (partial window at each epoch end) and full-epoch aggregate
active/perplexity/max-code fraction. It describes a changing model, not a fixed
checkpoint. Every full validation also evaluates32 fixed pages (4/style) in
normal eval, encoder-only BN batch statistics, decoder-only, and both modes.
Use the complete normal-validation Hungarian mapping for all four, not a fresh
subset assignment. Record recon, foreground RGB/chroma, usage, assignment-change
fraction and fixed-mapping accuracy. VQ stays frozen. Restore all buffers,
module modes and Python/NumPy/Torch CPU/CUDA RNG in `finally`; no optimizer,
loss or checkpoint-selection side effects. Only normal full validation ranks
checkpoints. Keep all existing loss/color/codebook/V3/leakage curves.

**Figures / retention.** Every25 completed epochs plus extra0/10/11/12/15/20/50/100
produce paired recon/mask and balanced confusion with one full-val mapping.
Extra figures do not add snapshots. Retain current, health-gated macro-best,
best-val, and8 snapshots at24/49/74/99/124/149/174/199: at most11 checkpoints.
Macro selection floors: active>=24, perplexity>=18, coverage>=18, weighted>=.20.
Best-val is only a reconstruction comparison, not the primary semantic metric.

**Acceptance / decision.** Full2600-page test at macro-best/best-val/current,
normal eval: macro>=.75, active26, perplexity>=20, coverage>=24, recon<=.30.
Report weighted purity, Hungarian, colors and BN mode gaps separately. Stable
window is8/10 epochs meeting macro>=.70, active26, perplexity>=20, coverage>=24.
Earlier W50 onset plus post-release persistence supports warmup; W0 success with
W50 failure indicates risk from the early constraint. Continued degradation or
large BN gaps motivates a separate normalization architecture control next.

**Operations.** Two ws-ia jobs, each1GPU/32GB/4CPU/**12h**; GPU tests and two
full-validation smoke checks before registering/submitting. Smoke directories
deleted after visual checks. Do not resume C128-W50 or cancelled old S4 C512.
See [acceptance and provenance](analyses/20260913_PN_W50_ACCEPTANCE_C512.md).

**S7 outcome, accepted2026-09-13 evening.** Both completed199/exit0. W0 had no
health-eligible macro candidate in200 epochs; current full-test macro=.39856,
weighted=.12641, perplexity4.25. W50 restored all26 codes/perplexity24.21 but
current macro=.37628; macro-best102=.44149. No stable window in either arm.
Fixed-batch decoder-only BN interventions remove reconstruction spikes without
changing assignments; encoder-only mode changes do not restore semantics.
No continuation. [Detailed acceptance and proposed lr-only next pair](analyses/20260913_S7_ACCEPTANCE_NEXT_LR.md).
The normalization control remains recommended; next hyperparameters are not yet
a registered/submitted setting.

## VVI-PN-S1 — Correct-color PhoneNums: Paper vs Published Config

Reference: upstream `bbbc7c9c26bda7170612814a407b76633cd22875` and paper AppendixB.2.
Direction RQ2 / hypothesis RQ2-H12: the successful content/style regime may depend
on the original task and the training-parameter bundle, not color corruption.

| Run | Optimizer / decay | Four V3 weights | Commit | EMA | Dead threshold |
|---|---|---:|---:|---:|---:|
| VVI-PN-K10-PAPER-S0 | Adam /0 | .1 each | .01 | .95 | 3.2 |
| VVI-PN-K10-REPO-S0 | AdamW /.1 | 1 each | .1 | .98 | 16 |

Shared: corrected PhoneNumsV2,100k pages (80k/10k/10k), each style10k/1250/1250,
each page all digits once. C512/S512/nativeVQ512/K10, original PhoneNums ResNet,
BN,10×32×48 RGB fragments, ordinary decoder, r15/recon1, float32, batch32, seed0,
200 epochs from scratch. Same initialization and sampler hashes. Scheduler:
upstream exponential decay factor.98/20 epochs, floor.02, no warmup; Adam betas
.9/.999 and eps1e-8. Configs under `configs/phonenums/v2/`; pin manifest checksum.

Each job requests1GPU/32GB/4CPU/24h. Retain current, best-val, macro-best and8
25-epoch snapshots (max11). Best-val selects the lowest finite complete-validation
total loss, ties keep the earlier epoch; it has no label/health gate. Macro-best
is diagnostic with active>=9, perplexity>=7, coverage>=7 and weighted>=.20.
Evaluate all three retained roles on full10,000-page test; balanced counts1250
per(digit,style). Record colors, mapping and the paper-style10×8 codebook/class-mean
style recombination, alongside10×16 paired reconstruction. Diagnostic labels do
not enter training. Compare macro with Table8 K10 .892, without equating it to
Hungarian or claiming statistical reproduction at one seed. Do not run legacy
quadratic PR retrieval or OOD baselines in this round.

## VVI-RQ2-S6 — Earlier V2 Page-mean Release at Epoch50

RQ2-H11 / `VVI-RQ2-V2-C128-K26-W50-S0`. Copy S5 W100 with the identical dataset,
seed0, C128/S512/K26, BN, loss/EMA/optimizer/scheduler and health gate. From scratch,
200 epochs: page-mean0–49, fragment50–199. Extras at0/50/100 plus every25 completed
epochs; preserve current, macro-best and8 snapshots (max10).1GPU/32GB/4CPU/8h.
Compare with S5 W100 before release, at49→50 and in the later phase. Keep the
existing stable-window/content gate and separate color diagnostics. Earlier
release is an untested intervention, not a demonstrated rescue.

## VVI-RQ2-S4 — Instrumented S3 C512 Reproduction

Requested2026-09-10 after the [S3 posthoc reconstruction check](analyses/20260910_S3_POSTHOC_RECON.md).
Run ID `VVI-RQ2-C512-K26-CTRL-S0`; config
`configs/uppercase/rq2/cfg_vvi_rq2_c512_k26_control_seed0.yaml`.

Status update 2026-09-10T17:16:20+04:00: user cancelled pending job183428;
no GPU allocation, training, or run directory was created. Preserve this protocol
and config as a deferred proposal; no automatic resubmission.

Preserve original S3 training settings: C512/S512/native K26-D512, seed0,
**150 epochs (0–149)** from scratch, fragment decoder, original BatchNorm,
batch32, EMA.98/dead threshold16, AdamW lr.001/weight decay.1, exponential
scheduler, r15, recon/four V3 weights1, commit.1. The original horizon150 is
kept rather than silently changing to the C128 experiment's200. No warmup,
loss curriculum, codebook projection, dataset fix or pretrained initialization.

Engineering-only additions: the same detached leakage probes and full-validation
statistics as RQ2-S3,25-completed-epoch snapshots and26×16 diagnostic grids,
plus grids at epoch0,139,140 around the old transition. Preserve original S3's
unconstrained validation-macro checkpoint ranking (no added health gate); use
current + macro-best + six permanent snapshots (at most8 files). Apply usage
health requirements during scientific acceptance, not as a new training change.

Compare against historical S3 and C128 CTRL on their common0–149 horizon.
Primary question: does high code purity recur, and does color reconstruction
fail at the same time? Use the same phase-window definition as RQ2-S3, and
report incomplete windows at the end rather than claiming long-term stability.
Final acceptance checks full-test best/current purity, usage, reconstruction,
color grids and cross-label probes. A high-purity brown-output endpoint is not
successful style recovery. All computation on GPU, ws-ia1GPU/32GB/4CPU/8h.

## VVI-RQ2-S3 — C128 Style-Bypass Ablation and Warmup

Requested 2026-09-10. Immediate priority over the proposed historical-epoch80
pair, which remains an unrun follow-up. Question: can removing fragment-specific
style input improve content learning, and does the benefit persist after release?

**Matched protocol.** Copy the original C128 run's training protocol from
`20260807-1506__VVI-RQ1A-C128-K26-S0/config.yaml`: scratch epochs0–199, seed0,
UppercaseLetters20,800 train /2,600 validation /2,600 test pages, all26 letters
per page, eight colors. CSAE C128/S512, native K26/D128, original BatchNorm,
EMA .98/dead threshold16, batch32, AdamW lr=.001/weight decay=.1, original
exponential scheduler, relativity15, recon/four V3 weights1, commit .1. No
dataset fix, loss schedule, or checkpoint initialization.

| Run ID | Decoder style input | Purpose |
|---|---|---|
| VVI-RQ2-C128-K26-CTRL-S0 | raw fragment style, epochs0–199 | contemporary original-C128 reproduction |
| VVI-RQ2-C128-K26-MEAN-S0 | page mean broadcast, epochs0–199 | permanent removal of fragment-specific style shortcut |
| VVI-RQ2-C128-K26-W100-S0 | page mean epochs0–99, raw fragment style epochs100–199 | early constraint then release |

Only decoder input is averaged, with gradients preserved. V3 and all probes
receive raw per-fragment style. At epoch100 do not reset optimizer, scheduler,
EMA, scaler or weights. Checkpoint model state stores decoder epoch for correct
standalone evaluation and resume. The MEAN and W100 arms should track before
epoch100; GPU numerical variation can prevent bitwise trajectory equality.

**Probes.** Reuse full-validation forwards each epoch, without extra model
passes or test labels. Fixed512 pages,64 per color; disjoint256 fit/256 score
pages. Never split fragments of one page across fit/score. Native VQ→style uses
Laplace-smoothed code/style counts (accuracy/CE). Raw style→content, style→style
positive control, and pre-VQ content→style use raw and PCA-whitened64 ridge
readouts, ridge=.01/std floor1e-7. Preprocessing uses fit pages only. Labels do
not alter model gradients, switch timing, or checkpoint ranking. Chance is1/8
for color and1/26 for content; probe accuracy is not exact mutual information.
Record full-validation four raw MPDs and per-hinge active-batch fractions too.
Outputs: `disentanglement_probe_protocol.json`, `disentanglement_probe_history.csv`,
`disentanglement_probes.png`, plus existing loss/codebook/V3 histories.

**Snapshots and images.** User-authorized permanent snapshots at completed
epochs25/50/75/100/125/150/175/200 (internal epoch24/49/74/99/124/149/174/199):
`cp_snapshot_epoch<N>.pt`, complete optimizer/scheduler/scaler/model state.
Current and health-gated macro-best still rotate: at most8+2 checkpoint files.
At the same epochs render fixed26×16 original/recon pairs (eight fixed validation
pages, one per color). Annotate code→label using that epoch's full-validation
Hungarian mapping. Save PNG/JSON/mapping CSV in `reconstruction_diagnostics/`.
Also draw epoch0 and100; the epoch99 snapshot/grid is the pre-release reference.
All three runs use identical images. Existing macro-best health gate is unchanged.

**Decision gate.** Compare onset against concurrent CTRL, not only historical
onset~160. Existing stable-phase definition: at least8 of10 consecutive epochs
with macro>=.70, active=26, perplexity>=20, coverage>=24. Final acceptance uses
full test on macro-best and current, macro>=.75, active=26, perplexity>=20,
coverage>=24, recon<=.30. Scalar recon/purity alone do not establish color recovery.

- MEAN improves purity but W100 loses it after release: continuing constraint is
  needed; not evidence of a persistent semantic state.
- W100 retains healthy purity and improves color after release: promising warmup.
- High purity with brown reconstructions: content-only gain; style remains broken.
- No gain: negative for this seed/protocol; test seeds before general conclusions.

The positive style→style control distinguishes low content leakage from complete
style failure. Low geometric variance does not prove low semantic information.
Page sharing requires same-style fragments: MEAN is not a drop-in V5 mixed-style
triplet model; W100 eventually returns to ordinary per-image decoding.

Tests/smokes only on compute GPU; smoke artifacts removed, never entered in the
ledger. Formal requests: ws-ia,1GPU,32GB,4CPU,8h each. No commit/push.

Preflight passed2026-09-10 on ws-l1-011 (RTX5000 Ada):42 unit tests; all three
decoder regimes; warmup epoch99→100; full-validation probe/grid generation;
strict snapshot reload preserves decoder regime and resume state; permanent
snapshots survive current/best rotation. Three initial model hashes matched
`39cf3f48…0a128`. Disposable smoke directories were removed (~1.9GiB).
Each Slurm job saves submitted config/checksum, source hashes/archive, Git SHA,
dirty status and tracked diff under its run's `reproducibility/` directory.

Submitted13:48+04:00: CTRL183134 on ws-l1-001; W100183133 on ws-l1-011;
MEAN183135 queued with `QOSMaxJobsPerUserLimit` (two running jobs per user).
Do not cancel unrelated jobs or bypass scheduler policy to force concurrency.

**User-requested pause2026-09-10,15:10:58+04:00.** Cancelled jobs183133/183134/
183135 to release resources; queued MEAN must not auto-start. All existing model
files, snapshots, diagnostics and logs are retained. Cancellation is operational,
not a scientific failure or full-test acceptance. Wait for explicit continuation.

| Arm | Retained resume checkpoint | Next absolute epoch | Remaining epochs to199 |
|---|---|---:|---:|
| CTRL | `logs/20260910-1348__VVI-RQ2-C128-K26-CTRL-S0/cp_current_epoch37.pt` | 38 | 162 |
| MEAN | none; never started | 0 | 200 |
| W100 | `logs/20260910-1348__VVI-RQ2-C128-K26-W100-S0/cp_current_epoch40.pt` | 41 | 159 |

Resume from current rather than macro-best to preserve latest completed training
progress. Reload optimizer/scheduler/scaler and keep the W100 switch at absolute
epoch100. Interrupted partial epochs will be replayed from their start; this is
not exact mid-batch continuation. File existence/metadata were checked after
cancellation; future checkpoint loading and all training must run on compute GPU.

**Continuation authorized later2026-09-10.** Submit new R1 runs (keep cancelled
submissions in ARCHIVE). CTRL uses162 incremental epochs starting38; W100 uses159
starting41; MEAN uses200 from0 because it never ran. Decoder policy, optimizer,
scheduler, scaler, objective and checkpoint health gates remain unchanged.
`inherit_resume_history: true` copies only histories through the resume epoch
and the actual old macro-best file (W100 epoch39), with provenance in
`resume_lineage.json`. Old snapshots/grids stay in the parent run. Fixed probe
fit/score/grid selections must match the parent protocol. Source checkpoints
lack RNG state, so this is epoch-boundary state recovery, not bitwise continuation
of the old data order; factor this interruption into causal comparisons.

| New Run ID | Config | Epoch range |
|---|---|---|
| VVI-RQ2-C128-K26-CTRL-S0-R1 | `configs/uppercase/rq2/cfg_vvi_rq2_c128_k26_control_seed0_resume1.yaml` | 38–199 |
| VVI-RQ2-C128-K26-MEAN-S0-R1 | `configs/uppercase/rq2/cfg_vvi_rq2_c128_k26_pagemean_seed0_resume1.yaml` | 0–199 |
| VVI-RQ2-C128-K26-W100-S0-R1 | `configs/uppercase/rq2/cfg_vvi_rq2_c128_k26_meanwarm100_seed0_resume1.yaml` | 41–199 |

**Accepted 2026-09-11.** Both R1 runs complete epoch199 and pass the registered
full-test content/reconstruction gate. CTRL best epoch166 has recon `.15130`,
macro `.91455`, Hungarian `.91178`, active26, perplexity25.81 and coverage26;
its first stable window is150-159. W100 best epoch197 has recon `.14932`, macro
`.93249`, Hungarian `.93135`, active26, perplexity25.81 and coverage26; its first
stable window is76-85 and post-release182-191 is10/10 healthy. Current epoch199
remains healthy but below best in both arms. W100 supports earlier semantic
catalysis and persistence at seed0. Both reconstruction series lose source
colors, so neither passes the qualitative style-reconstruction requirement.

Continuation preflight passed on ws-l1-002:45 unit tests and real-checkpoint
two-batch smoke epochs38/41 plus C512 scratch epoch0. Model state, learning rate,
optimizer/scheduler/scaler, inherited best39, completed-only histories and fixed
probe protocol verified. Temporary smoke artifacts were deleted (~1.6GiB).

Submission state at16:35+04:00: CTRL-R1 job183426 running on ws-l1-002;
W100-R1 job183427 running on ws-l1-011; C512 job183428 and MEAN-R1 job183429 queued.

**Latest user update,2026-09-10T17:16:20+04:00.** Cancelled pending jobs183428/
183429 with a PENDING-only filter. Slurm confirmed CANCELLED, runtime0 and no
resource allocation; neither run directory exists. CTRL183426/W100183427 remain
RUNNING and unchanged, with the original epoch199 endpoint and W100 epoch100
switch. Cancelled attempts are archived, not scientific negatives; the permanent
MEAN comparison is still unavailable. After these two finish and are accepted,
prioritize BN-state isolation and a versioned correction of Gaussian-noise
uint8 overflow. Do not modify the live runs' model, config, or PNG data, and do
not submit further formal experiments automatically.

## VVI-RQ4-S1 - AGUSA dependency bridge

VVI supplies the accepted C128/S512/K26 epoch-165 source model. Teacher A passed
the V5 gate; all adaptation settings, hypotheses, controls, and future results
are canonical in the V5 [RQ4 direction](../../v5/research/directions/RQ4_AGUSA.md).

## VVI-H0 — Historical Uppercase References

Historical reference checkpoints used to calibrate the current stage. Metrics in
the archive were recomputed on the full 2,600-sample test split with the same
acceptance evaluator used for VVI-S1.

| Run ID | Content dim | Style dim | Role |
|---|---:|---:|---|
| VVI-H0-C4S4 | 4 | 4 | failed low-dimensional reference |
| VVI-H0-C512S512 | 512 | 512 | high-dimensional reference |

## VVI-S1 — Uppercase Style-Bottleneck Isolation

**Direction / hypothesis.** RQ1-A / RQ1-H1.

**Question.** Which four-dimensional bottleneck causes codebook collapse:
content, style, or only their combination?

**Hypothesis.** If `c=4, s=512` restores balanced code usage and one-to-one
alignment, the simultaneous four-dimensional style bottleneck was a major cause.
If it remains collapsed, the four-dimensional content/VQ geometry is primary.

**Data.** UppercaseLetters: 20,800 train, 2,600 validation, and 2,600 test
samples. Every sample contains all 26 uppercase letters in one of eight colors.

**Model.** The existing ResNet-like content-style autoencoder with a 26-entry EMA
VQ codebook. The two isolation runs use `c=4, s=512` and `c=512, s=4`; the
decoder receives their concatenation. No VQ, loss, or normalization change is
made in this stage.

**Objective.** Reconstruction plus the four V3 hinge terms and VQ commitment
loss. Weights, relativity `r=15`, AdamW, learning-rate schedule, EMA decay `0.98`,
and dead-code threshold `16` match the failed `4/4` run.

**Diagnostics.** Validation loss components; test reconstruction/V3/total loss;
active codes; usage perplexity; codebook purity; Hungarian one-to-one accuracy;
dominant-label coverage; minimum atom distance; and fixed-batch eval/train
reconstruction ratio.

**Run plan.**

| Run ID | Content dim | Style dim | Seed | Role |
|---|---:|---:|---:|---|
| VVI-S1-C4S512-S0 | 4 | 512 | 0 | isolate removal of the style bottleneck |
| VVI-S1-C512S4-S0 | 512 | 4 | 0 | complementary content/style control |
| VVI-S1-C512S4-S0-R1 | 512 | 4 | 0 | resume interrupted S0 from epoch 15 through epoch 149 |

**Decision gate.**

- Rescue: active codes `=26`, perplexity `>=20`, one-to-one accuracy `>=0.60`,
  and test reconstruction `<=0.30`. Next test: `c=512, s=4`.
- Persistent collapse: perplexity `<=12` and one-to-one accuracy `<=0.30`.
  Next stage: high-dimensional internal content with `codebook_dim=4`.
- Normalization failure: codebook passes the rescue gate but the fixed-batch
  eval/train reconstruction ratio is `>2`. Next stage: replace BatchNorm with
  GroupNorm.
- Intermediate outcome: run the `c=512, s=4, seed=0` control before changing EMA
  or loss weights.

The completed `c=4, s=512` run met the persistent-collapse gate. The completed
`c=512, s=4` control restored all 26 active codes and perplexity 25.06, but its
full-test one-to-one accuracy remained 0.262. Thus the content bottleneck causes
usage collapse, while the style bottleneck permits balanced but semantically
misaligned assignments. Neither run should be continued.

## VVI-S2 — Projected Four-Dimensional Codebook

**Direction / hypothesis.** RQ1-B / RQ1-H2.

**Question.** Can the codebook itself remain four-dimensional when the encoder
and decoder retain a 512-dimensional content representation?

**Hypothesis.** The failure of `d_emb_c=4` is caused by forcing the entire
content pathway through four dimensions, not by storing 26 atoms in 4D. Learned
`512→4` and `4→512` projections around VQ should preserve network capacity while
testing the desired low-dimensional atom geometry.

**Data.** The same UppercaseLetters train/validation/test splits as VVI-S1.

**Model.** The existing CSAE with `d_emb_c=512`, `d_emb_s=512`, 26 EMA atoms,
and `vq_codebook_dim=4`. The VQ layer supplies learned input/output projections;
all other architecture, normalization, EMA, and dead-code settings are fixed.

**Objective and diagnostics.** Identical to VVI-S1, including the full-test
codebook-health acceptance protocol.

**Run plan.**

| Run ID | Internal content | Style | Codebook dim | Seed | Role |
|---|---:|---:|---:|---:|---|
| VVI-S2-C512S512-CB4-S0 | 512 | 512 | 4 | 0 | test low-dimensional VQ without a low-capacity content path |
| VVI-S2-C512S512-CB4-S0-R1 | 512 | 512 | 4 | 0 | resume interrupted S0 from epoch 15 through epoch 149 |
| VVI-S2-C512S512-CB4-S0-R2 | 512 | 512 | 4 | 0 | continue R1 from epoch 149 through epoch 299; retain current plus macro-purity best |

**Decision gate.**

- Rescue uses the VVI-S1 thresholds: active codes `=26`, perplexity `>=20`,
  one-to-one accuracy `>=0.60`, and test reconstruction `<=0.30`.
- If rescued, repeat the projected-codebook setting with additional seeds before
  changing EMA or loss weights.
- If usage still collapses, compare its learning trajectory with
  VVI-H0-C512S512 and next isolate the projection/VQ training strategy.

The projected 4D run restored all 26 active codes, perplexity 25.34, and test
reconstruction 0.141, but its one-to-one accuracy was only 0.306. Both V3 and
the fixed-batch normalization diagnostic passed (`V3=0`, eval/train=1.19).
R2 extends the same trajectory to epoch 299 because macro purity remains the
relevant many-codes-per-content criterion; it starts with epoch 149's validation
macro purity `0.383157` as the best floor. The seed-matched full-dimensional S3
control remains independent and runs concurrently.

## VVI-S3 — Seed-Matched Full-Dimensional Baseline

**Direction / hypothesis.** RQ2 / RQ2-H1; accepted K=26 source for RQ3.

**Question.** Can the current seed-0, 150-epoch protocol reproduce the strong
full-dimensional historical reference before the 4D projection is blamed?

**Model and data.** The same UppercaseLetters splits and CSAE/V3 objective as
VVI-S2, with `d_emb_c=512`, `d_emb_s=512`, 26 atoms, and a 512D codebook. There
is no VQ input/output projection. Seed, optimizer, scheduler, EMA, dead-code
threshold, batch size, and validation protocol match VVI-S2.

**Primary diagnostic.** Validation macro atom purity is the oracle checkpoint
criterion. It permits multiple codes to map to one content. Hungarian remains a
diagnostic; active codes, perplexity, coverage, and usage-weighted purity guard
against superficially pure low-frequency or duplicated codes. Four raw V3 ratios
are recorded to choose a future relativity margin.

**Run plan.**

| Run ID | Content | Style | Codebook | Seed | Epochs | Role |
|---|---:|---:|---:|---:|---:|---|
| VVI-S3-C512S512-CB512-S0 | 512 | 512 | 512 | 0 | 150 | seed-matched reproducibility control |

**Decision gate.**

- Reproduced: macro purity `>=0.75`, active codes `=26`, perplexity `>=20`,
  coverage `>=24`, and test reconstruction `<=0.30`; next test CB4 with a V3
  relativity chosen from the measured ratios.
- Macro purity `<0.50` with a flat late trajectory: test additional seeds before
  attributing the failure to codebook dimension.
- Macro purity `0.50-0.75` and still rising: resume the best retained checkpoint
  toward epoch 450.
- High macro purity with degraded usage or coverage is not a successful rescue.

S3 passed the reproduced gate at macro-best epoch 145: full-test macro=.834,
usage-weighted purity=.820, active=26, perplexity=25.18, coverage=25, and
reconstruction=.154. Its abrupt transition near epoch 140 motivates VVI-RQ2-S1.

## VVI-RQ2-S1 — Relativity Curriculum Catalyst

**Direction / hypothesis.** RQ2 / RQ2-H2.

Full-dimensional K=26 CSAE, seed 0, trained from scratch for 200 epochs. All S3
architecture, optimizer, EMA, commit, and reconstruction settings are fixed.
Relativity follows absolute-epoch knots `(0,15), (60,15), (100,25), (140,25),
(170,15), (199,15)` with linear interpolation. Reconstruction and commitment
weights remain 1 and .1.

Stable phase means a 10-epoch window with at least eight epochs satisfying
macro>=.70, active=26, perplexity>=20, and coverage>=24. Onset<=120 is success,
121–139 weak acceleration, and >=140/absent no support. Persistence after r=15
restoration additionally requires full-test macro>=.75 and recon<=.30.

**Outcome.** No stable phase occurred. Best validation macro=.589 at epoch 86
coincided with usage collapse. Full-test best macro=.564, weighted=.142,
active=25, perplexity=5.32, coverage=9, recon=.298. Epoch 199 had recon=230.25
and eval/train reconstruction ratio=1485. This run does not support RQ2-H2 and
is not suitable for continuation or transfer to CB4.

## VVI-RQ3-S1 — Warm 2x Codebook Expansion

**Direction / hypothesis.** RQ3 / RQ3-H1 and RQ3-H3.

Load S3 macro-best epoch 145 and expand K26→K52 by interleaving symmetric
`center ± delta` pairs. Delta norm is .01 times the source median nearest-neighbor
distance with seed 0. Split each EMA cluster mass equally and recompute
`embed_avg`. Reset macro-best tracking, lower the dead-code threshold 16→8,
validate before updates, then train epochs 146–295 with all other settings fixed.

Initialization must retain macro and weighted purity>=.75 and coverage=26. Final
success requires active>=47, perplexity>=40, coverage=26, both purities>=.75,
recon<=.30, dominant-code min/max within 1/3, and CV<=.35.

**Outcome.** Passed every gate. Macro-best epoch 275 full test: macro=.9572,
weighted=.9663, active=52, perplexity=48.06, coverage=26, recon=.1498,
dominant-code min/max=1/3, CV=.310. Current epoch 295 also passed, confirming
that the expanded state is stable rather than a checkpoint-selection spike.

## VVI-RQ2-S2 — Matched Pre-transition Recon Pulse

**Direction / hypothesis.** RQ2 / RQ2-H3.

Restore S3 epoch 130 (SHA256
`7ab7561366d770b7ab20fa961e3a32611993971327a7609a4da199e52ddf9c92`)
from the system trash into a dedicated ignored checkpoint-source directory. At
that epoch validation recon=.1214, macro=.2850, active=26, perplexity=25.55, and
coverage=21. Launch two branches with the same model, optimizer, scheduler,
scaler, seed, data order, and epochs 131–170:

| Run role | Relativity | Recon-weight schedule |
|---|---:|---|
| matched control | 15 | constant 1 |
| RQ2-H3 pulse | 15 | knots `(131,1), (135,.5), (145,.5), (155,1), (170,1)` |

Both runs validate before the first update and reset branch-local macro-best
tracking. The pulse passes only if stable-phase onset precedes the control by at
least five epochs, persists after epoch 155, and full test has macro>=.75,
active=26, perplexity>=20, coverage>=24, recon<=.30, and eval/train recon<=2.
If the control does not reproduce a transition, first add RNG-state persistence
to checkpoints; if both branches show eval/train spikes, isolate BatchNorm before
drawing a catalyst conclusion.

## VVI-RQ3-S2 — K52 Cold-start Control

**Direction / hypothesis.** RQ3 / RQ3-H2.

Train from scratch for 300 epochs with seed 0, `d_emb_c=d_emb_s=512`, K=52,
512D codebook, dead-code threshold 8, and otherwise the S3 objective, optimizer,
scheduler, EMA, batch size, and validation protocol. Use
`current_and_best_macro` and the same K=52 success gate as VVI-RQ3-S1.

Passing shows that warm initialization accelerates but is not required. Failure
with healthy optimization shows that semantic inheritance is currently required.
This mechanism control is independent of the application-driven PCA-warm K104
experiment below.

## VVI-RQ13-S1 — K104 Native Low-dimensional PCA-warm Pair

**Direction / hypotheses.** RQ1-B and RQ3 / RQ1-H5 and RQ3-H4.

**Application question.** Can the discrete representation exposed to a small
addition model be 4D or 8D while the image encoder, style pathway, and decoder
remain 512D, and can a redundant K104 codebook retain semantic aliases?

**Source and transform.** Start from `VVI-RQ3-K52-WARM-S0` macro-best epoch 275,
SHA256 `1bdf4788b68d6f0ffc106ef6bd691b7295a4b0ebd93488c8ba566e490b04ff88`.
Fit unweighted float64 PCA to the 52 learned atoms with deterministic component
signs. Freeze `project_in(x)=(x-mu)V_D` and
`project_out(z)=zV_D^T+mu`, then split each projected atom into a symmetric pair
with 1% median-nearest-neighbour jitter, seed 0. EMA mass is split evenly.
Encoder/decoder weights are copied, but optimizer, scheduler, scaler, epoch, and
macro-best state reset; training begins at epoch 0.

**Common training.** `d_emb_c=d_emb_s=512`, K104, threshold 4, EMA .98,
relativity 15, standard V3 weights, batch 32, seed 0, fresh AdamW at `1e-4`, and
150 epochs. The two controlled runs differ only in native VQ dimension D=4/8.

| Run ID | K | Native VQ D | Role |
|---|---:|---:|---|
| VVI-RQ13-K104-D4-PCAWARM-S0 | 104 | 4 | minimum target interface for the addition model |
| VVI-RQ13-K104-D8-PCAWARM-S0 | 104 | 8 | higher-dimensional feasibility/control arm |

**Initialization gate.** Active>=100, perplexity>=80, coverage=26, macro and
weighted purity>=.90, dominant-code counts 2–7 with CV<=.30, recon<=.20. Both
GPU smokes passed: D4 init macro=.9636, weighted=.9805, active=103,
perplexity=87.43, recon=.1438; D8 init macro=.9620, weighted=.9777, active=103,
perplexity=87.12, recon=.1439.

**Training gate.** Active>=94, perplexity>=80, coverage=26, both purities>=.90,
dominant counts 2–7, CV<=.35, recon<=.30, alias nearest-neighbour margin>=.90,
and native within/between ratio no more than 20% worse than initialization.
Macro-best candidates additionally require active>=94, perplexity>=75,
coverage=26, weighted purity>=.90, dominant counts 2–8, and CV<=.40.

**Downstream decision gate.** Freeze each accepted VVI checkpoint and use native
D-dimensional atoms directly in the same two-hidden-layer, 64-unit addition MLP
for 20k steps over the 231 pairs from 0..20. Training targets are observed
`x_c` codes; number labels are diagnostics only. Success is train number
accuracy>=.95 and test-style number accuracy>=.80. If D4 and D8 pass, choose D4;
if only D8 passes, use D8 and next test D6. VVI pass plus probe failure identifies
alias/arithmetic geometry rather than atom purity as the bottleneck.

**VVI outcome.** Both macro-best checkpoints pass every full-test gate. D4 epoch
28: macro=.99968, weighted=.99973, active=104, perplexity=98.00, coverage=26,
dominant min/max=3/5, CV=.098, recon=.1221, alias margin=.990, ratio=.254. D8
epoch 38: macro=.99916, weighted=.99919, active=104, perplexity=98.61,
coverage=26, dominant min/max=3/5, CV=.069, recon=.1228, alias margin=.981,
ratio=.256. At epoch149 purity remained high but alias margin fell to .874/.817;
therefore use macro-best, not current, for the addition probes.

## VVI-RQ1A-S1 — K26 Content-Width Phase Threshold

**Direction / hypotheses.** RQ1-A with RQ2 phase observation and an RQ3 pilot /
RQ1-H7, RQ1-H8, and RQ1-H9.

**Question.** With the original K26 V3 objective, how far can the entire content
embedding be narrowed while still entering the high-purity phase? This is
separate from RQ1-B: here `d_emb_c` and the native VQ atom dimension shrink
together; there is no `vq_codebook_dim` projection. Style remains 512D.

**Matched baseline.** S3 seed 0: `d_emb_c=d_emb_s=512`, K26, threshold 16,
EMA=.98, relativity 15, all loss weights unchanged, batch 32, AdamW `lr=1e-3`,
and the same scheduler/data. The primary K26 arms change only `d_emb_c`; the
C64/K52 pilot also halves the dead-code threshold to 8. All runs use 200 epochs
so a transition later than S3's onset near epoch 140 remains observable.

| Run ID | `d_emb_c` | Style | K | Native VQ D | Seed | Epochs |
|---|---:|---:|---:|---:|---:|---:|
| VVI-RQ1A-C128-K26-S0 | 128 | 512 | 26 | 128 | 0 | 200 |
| VVI-RQ1A-C32-K26-S0 | 32 | 512 | 26 | 32 | 0 | 200 |
| VVI-RQ1A-C64-K26-S0 | 64 | 512 | 26 | 64 | 0 | 200 |
| VVI-RQ1A-C64-K52-S0 | 64 | 512 | 52 | 64 | 0 | 200 |

**Stable-phase definition.** In a 10-epoch validation window, at least eight
epochs simultaneously have macro>=.70, active=26, perplexity>=20, and
coverage>=24. Record the first qualifying epoch as onset. A run formally passes
only if its macro-best full test also has macro and weighted purity>=.75,
active=26, perplexity>=20, coverage>=24, recon<=.30, and eval/train recon<=2.

**Outcome.** C128 first met the stable-window definition at epoch 160 and its
epoch-165 macro-best passed full test (macro=.861, weighted=.848, active=26,
perplexity=25.49, coverage=26, recon=.157). C64/K26 never transitioned despite
healthy reconstruction and all 26 codes active at epoch 199; its best macro was
.428. C32 also did not transition and developed severe BatchNorm-style
train/eval mismatch. The clean seed-0 threshold is therefore `(64,128]`; C96 is
the next midpoint. C64/K52 did not compensate for width: best macro=.514 and
weighted=.374, followed by active-code/reconstruction collapse.

Macro-best candidates are restricted to active>=24,
perplexity>=18, coverage>=18, and weighted purity>=.20 so rare-code macro spikes
cannot replace a healthy checkpoint.

## VVI-RQ13-S2 — Native Content Width under K128 Redundancy

**Direction / hypotheses.** RQ1-A and RQ3 / RQ1-H10, RQ1-H11, and RQ3-H6.

**Question.** When the native codebook contains roughly five aliases per content,
can C128 retain the semantic phase seen at K26, and can that redundancy rescue
the otherwise unsuccessful C32 representation?

**Controlled protocol.** Both runs start from scratch with seed 0, style 512D,
native VQ (no `vq_codebook_dim` projection), K128, EMA decay .98, relativity 15,
the standard V3/reconstruction/commit weights, batch 32, AdamW `lr=1e-3`, and
200 epochs. They differ only in `d_emb_c`. Dead-code threshold 3 is approximately
half of the uniform expected 6.5 assignments per atom in a 32-sample batch,
matching the relative pressure used for K26/K52/K104.

| Run ID | `d_emb_c` | Style | K | Native VQ D | Seed | Epochs |
|---|---:|---:|---:|---:|---:|---:|
| VVI-RQ13-C32-K128-S0 | 32 | 512 | 128 | 32 | 0 | 200 |
| VVI-RQ13-C128-K128-S0 | 128 | 512 | 128 | 128 | 0 | 200 |

**Checkpoint health floor.** Macro-best candidates require active>=96,
perplexity>=60, coverage>=20, and usage-weighted purity>=.20. This is only a
checkpoint anti-degeneration floor, not the success gate.

**Success gate.** Full-test macro-best must have macro and weighted purity>=.75,
active>=115, perplexity>=96, coverage=26, reconstruction<=.30, eval/train
reconstruction<=2, dominant-code counts 2–8 per content with CV<=.40, and alias
nearest-same margin>=.80. A stable training phase requires at least eight epochs
in a 10-epoch window satisfying the same semantic/usage conditions except the
full-test reconstruction diagnostics.

**Decision gate.** C128 pass plus C32 fail means redundancy is compatible with
C128 but does not lower the native-width threshold. If both pass, redundancy
rescues C32 and the next test is C16/K128. If both fail with healthy usage,
K128 cold-start optimization rather than content width is the leading cause;
compare warm expansion before changing V3. If usage collapses, first tune the
dead-code threshold rather than interpreting macro purity alone.

**Outcome.** C32 failed despite excellent usage: health-gated best full-test
macro=.423, weighted=.403, active=128, perplexity=121.33, and coverage=25.
C128 reached macro=.771 and weighted=.768 with all 128 codes at epoch195, but it
was the only epoch above .75; epoch199 fell to macro=.450. Its full-test alias
margin=.766 also missed the .80 gate. Thus 128D is sufficient for a transient
redundant semantic state, while flat K128 scratch optimization cannot maintain
it. Neither run should be continued from current.

## VVI-RQ2-S5 — V2 Data-Correction Replication

**Direction / hypotheses.** RQ2 / RQ2-H9 (warmup catalysis), RQ2-H10 (data-version
effect on color and validation stability). Starts from scratch, not resume.

| Run ID | Decoder style | Epochs | Seed |
|---|---|---:|---:|
| VVI-RQ2-V2-C128-K26-CTRL-S0 | original fragment style throughout | 200 (0–199) | 0 |
| VVI-RQ2-V2-C128-K26-W100-S0 | page mean0–99, original from100 | 200 (0–199) | 0 |

Configs live in `configs/uppercase/rq2/v2/`. C128/S512/K26/nativeVQ128, original
BatchNorm, batch32, float32, EMA decay.98/dead threshold16, relativity15,
recon/content/style/sample/fragment weights1, commit.1, AdamW lr.001/weight decay.1,
and the complete existing scheduler match S3 CTRL/W100. Initial model state and
manifest are hashed; identical seed and manifest order give matched shuffles.

**Data.** Separate UppercaseLettersV2: 26,000 RGB pages,26 unique letters/page;
per style train2600/val325/test325. Generation/split seeds0; signed float noise
then clip/cast and complete last blur strip. Manifest pins every PNG, source/font
hash and dependencies. SHA256:
`d455680108040b20d4880aff2bba29b4de0878f97cb0d37fd10ca91b8f1dc5b6`.
Old PNGs and legacy rendering stay unchanged. This comparison is data-version
replication, not isolation of one fix. No clean references or true masks.

**Diagnostics.** Every full validation reuses its existing forwards for foreground
RGB MAE and RGB-chroma RMSE, valid/invalid fragment counts, mask fraction and
prediction-out-of-range fraction, overall and all8 styles. No prediction clipping.
Input-only masks: channel median on outer2px, interior max-channel delta>.15,
valid if at least16 pixels; invalid-only groups produce NaN, not zero. Report
mask/noise limitations. Retain existing detached leakage/raw-MPD probes.

Every25 completed epochs plus epoch0/100, save paired recon/mask and balanced
confusion SVG/PNG/CSV/JSON. Full val/test counts are325 per(letter,style), verified
from actual batches. Hungarian on raw counts, rows=real code IDs, columns=A–Z,
P(code|content); rate comes from raw counts. Display all values to2 decimals using
column-conserving rounding; unrounded data retained. No extra training forward.

**Retention / gates.** Current + health-gated macro-best +8 periodic snapshots
(epoch24/49/74/99/124/149/174/199), at most10 checkpoints. The unchanged best floor
is active>=24, perplexity>=18, coverage>=18, weighted purity>=.20. Content success:
full-test macro/weighted>=.75, active26, perplexity>=20, coverage>=24, recon<=.30,
eval/train recon<=2. Stable onset: at least8/10 consecutive validation epochs with
macro>=.70, active26, perplexity>=20, coverage>=24. Color is reported separately,
without an uncalibrated hard threshold. Evaluate both macro-best and current on
full test, including colors/confusion/reconstruction. Continued val spikes with
smooth train on V2 motivate an independent BN comparison, not another data change.

**Operations.** Two ws-ia jobs, each1GPU/32GB/4CPU/8h after GPU preflight. Do not
restart cancelled permanent-MEAN or C512. [Preparation note](analyses/20260911_V2_REPLICATION.md).

## v3_rank — Native-VQ Effective-Rank Floor (method option, 2026-10-08)

**Status:** implemented;31 GPU tests including train/resume/evaluation smoke
passed. User-authorized formal pair registered below; efficacy remains untested.
**Direction / hypothesis:** RQ1-B / RQ1-H13. Variants `rank2` and
`rank4` mean target participation-ratio dimension2/4, not encoder width or K.
Per-page native quantized codes use `mean(relu(1-r_eff/target))`, added once
with a separate weight (initial default .01). EMA, V3 and model defaults stay
unchanged. Disabled by default; do not retroactively relabel historical runs.

Before a formal test, choose a matched successful Hex protocol and hold its
dataset/decoder/BN/optimizer/budget fixed. Measure geometry alongside semantic
purity, usage, reconstruction and style leakage; arithmetic remains a separate
downstream gate. Exact collapse may be stationary; do not assume rescue from
an old near-line checkpoint. See [method/config documentation](../manuals/v3_rank.md).

## VVI-HEX-RANK-S1 — v3_rank on Hex16 Permanent MEAN

**Direction / hypothesis:** RQ1-B / RQ1-H13. User-authorized2026-10-08.
Reference: `20260915-1121__VVI-HEX-K16-REPO-MEAN-S0`, whose macro-best171 had
full-test one-to-one1.0 but native PC1 fraction .999934 and effective rank1.000132.
The two new configs match `configs/hex/v2/cfg_vvi_hex_k16_repo_mean_seed0.yaml`
exactly except name, `method: v3_rank`, and the rank-regularization block.

| Run ID | Native codebook | Target rank | Rank weight | Decoder | Budget |
|---|---|---:|---:|---|---|
| VVI-HEX-K16-MEAN-RANK4-S0 | K16 / D512 | 4 | .01 | permanent page mean | 200 epochs, scratch seed0 |
| VVI-HEX-K16-MEAN-RANK2-S0 | K16 / D512 | 2 | .01 | permanent page mean | 200 epochs, scratch seed0 |

Immutable HexDigitsV2 manifest SHA256:
`120228b3129feddde5c9d64b0cff06a293bceb163d385ae5ce03bf2c8947254c`.
80k/10k/10k train/val/test pages,16 fragments per page,8 balanced styles.
C512/S512/native512, original encoder/decoder BN, batch32, float32, AdamW
lr.001/weight decay.1, exponential scheduler .98 each20epochs, EMA.98,
dead threshold16, relativity15, recon/four V3 weights1 and commit.1. No resume,
warmup, rank schedule, dimension reduction, extra variance floor or arithmetic.
Each run has2500 steps/epoch and500k total updates. Rank epsilon1e-12.

**Diagnostics / retention:** inherit all Hex MEAN probes, foreground colors,
per-style/content-group reports, BN interventions, online usage and optimization
scales. Add six rank fields to existing loss CSV/PNG/logs. Full-validation matrices
at original recon nodes remain balanced1250 per(content,style). Keep current,
health-gated macro-best, best-val, and8 periodic25-completed-epoch snapshots
(max11 files). Macro-best gate remains active>=15,ppl>=11.2,coverage>=12,
weighted purity>=.20; no geometry-based checkpoint selection.

**Preflight:** GPU formula/protocol tests, then four real train batches and full
10k-page validation for each arm. Verify matched initialization/full sampler
hashes, diagnostic state/RNG neutrality, strict checkpoint/evaluation loading,
all existing diagnostics and new rank fields. Delete smoke runs immediately.
Formal jobs: ws-ia,1GPU,32GB,4CPU,24h each; ledger before submission.

**Acceptance / decision:** full10k-page test on macro-best, best-val and current;
report rank loss and page-level rank alongside the unweighted full-codebook
spectrum, semantic metrics, colors, style leakage and recon. Do not confuse
sampled page rank with all-atom rank. Inherit the Hex semantic gate:
macro/weighted/one-to-one>=.75,active16,ppl>=13,coverage>=15,recon<=.30;
stable window>=8/10 epochs with macro/one-to-one>=.70 and the same usage floors.
For claiming rank preservation, additionally target page and full-codebook
effective rank>=.9*target at current and in late validation, without a material
purity loss versus the near-perfect historical baseline (report .95 as the
stricter semantic-retention diagnostic). These geometry thresholds are this
experiment's engineering criteria, not literature guarantees. Rank increase
with purity/style leakage deterioration is a trade-off, not success. If rank
stays near1, inspect objective/gradient scale before changing unrelated knobs.
Addition usability is a later separate evaluation, not authorized training here.

## VVI-RQ13-S3 — Warm K52 to K128 with a 128D VQ Interface

**Direction / hypotheses.** RQ1-B and RQ3 / RQ1-H12 and RQ3-H7.

**Recommended intervention.** Preserve the 512D encoder/decoder pathway but set
`vq_codebook_dim=128`, because the application only requires the discrete VQ
interface to be compact. Start from K52-WARM macro-best epoch275 (full-test
macro=.957, weighted=.966), project its atoms label-free into D128, then expand
52 parents to 128 children with a deterministic zero-mean local simplex split.
Allocate two or three children per source atom by largest-remainder EMA mass,
split each parent's EMA mass among its children, freeze the VQ projection, reset
training state, and use AdamW `lr=1e-4` as in the successful K104 PCA-warm runs.

Validate before updates and retain initialization as macro-best. Initialization
must have macro and weighted purity>=.90, active>=120, perplexity>=100,
coverage=26, dominant child counts 2–9 with CV<=.40, alias margin>=.85, and
reconstruction<=.20. Train 150 epochs with threshold 3 and otherwise unchanged
V3/EMA settings. Do not first change relativity, reconstruction weight, EMA
decay, or dead-code threshold: the failed scratch runs already had saturated
V3 loss, all codes active, high perplexity, and no under-threshold EMA clusters.
