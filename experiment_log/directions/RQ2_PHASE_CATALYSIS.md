# RQ2 — Catalyzing the semantic phase transition

## Research question and success criteria

Can the abrupt move from a low-purity reconstruction basin to a high-purity
semantic basin be induced earlier and remain stable after the catalyst is
removed? A stable phase is a 10-epoch window with at least eight epochs satisfying
macro `>=0.70`, active codes `=26`, perplexity `>=20`, and coverage `>=24`.

## Current conclusion

User update2026-09-17: S10 originals both completed199/130k updates with exit0.
Current complete-validation macro/1:1=.265715/.215710(fragment) and
.478319/.390355(permanent MEAN); active26/ppl23.97/24.46 indicate semantic mixing
despite broad usage. These are interim validation values, not a new full-test
acceptance. Both current199 states are continuing to769/500500 updates as R1;
GPU preflight passed; jobs197821/197822 run onws-l1-002/ws-l1-004 from13:20+04.
Restore optimizer/scheduler/scaler and retain the
original epoch-based LR; this does NOT match Hex LR by cumulative step. Old
checkpoints lack RNG state. Separate H25a (longer exposure on existing trajectory)
from H25b (fresh step-LR-matched protocol, untested). See
[continuation](../analyses/20260917_S10_CONTINUATION.md).

Latest acceptance2026-09-16: **HEX-S1 both pass full10k-test semantic gates**.
Macro-best both reach1.0; current Hungarian=.999731(REPO)/.999938(MEAN), all16
active/covered,perplexity~16. Permanent MEAN first stable window3–12 versus
REPO158–167 and preserves colors better (current chroma.06165 versus.15011).
However raw style-content readouts remain high; this is not proof of complete
disentanglement. REPO at130k steps was only.221; old uppercase stopped at130k,
while Hex received500k. Uppercase also uses25px glyphs versus48px, with about
half the estimated foreground occupancy. Class count is not isolated.
Both Hex runs archived; S10 originals196481/196482 repeated uppercase fragment/permanent-mean
with unchanged V2 data and verified Hex bundle, now completed and extended above. See
[acceptance/data-code audit/new protocol](../analyses/20260916_HEX_ACCEPTANCE_UPPERCASE.md).

Latest acceptance2026-09-15: **S9 NoNorm completed and is archived**. Both finish199
with exit0; six full-test roles/figures saved. Current Decoder-NoNorm/All-NoNorm
1:1=.49175/.29590 and recon=.00449/.00366, active26/perplexity~25. No registered
stable window; semantic mixing persists despite better color/reconstruction and
no numerical crash. Whitened z_s→content=.974/.928 is high. Removal of encoder BN
as well reduces final alignment in this seed. See [S9 acceptance](../analyses/20260915_S9_ACCEPTANCE.md).

Prior user priority **HEX-S1**: PhoneNums-style lowercase0–9/a–f on100k pages,
original REPO BN/lr.001, ordinary fragment versus permanent page-mean decoding.
Data generation/audit and both full-epoch preflights passed; the matched pair is
completed as193731/193733 and accepted above. This addresses a smaller
extension than26 uppercase characters without confounding the new data with
NoNorm or low-dimensional VQ. See [HEX16 protocol](../analyses/20260915_HEX16_PROTOCOL.md).

Previous acceptance2026-09-14: **S8 decoder normalization comparison is complete**.
Both finish199/exit0. Low-lr BN learns a real early aggregate semantic phase:
first stable window35–44, macro-best39 test macro.90738/Hungarian.79728,
active26/perplexity23.51/coverage25. However all8450 blue test fragments map to
code19; seven other styles are90.1–90.9% aligned, and reconstructions lose most
source colors. The original aggregate checkpoint gate passes but per-style
uniformity/color and persistence do not. After fragment release at50, macro
declines to.37757 by199 while reconstruction/colors improve.

Decoder GN8 collapses from0, with zero health-eligible epochs and current test
Hungarian.05652/perplexity1.94. No decoder running-stat gap is guaranteed by its
architecture, not evidence of model recovery. At BN199, frozen within-page
style permutation raises recon.00767→.21765, and content permutation also raises
it to.12138: both paths matter, consistent with renewed entanglement, not total
VQ non-use. Training-time release causality still needs a matched continuation
control. Keep BN/lr1e-4 as the promising branch; no blind current199 continuation.
Both runs are archived, no active jobs. Proposed next pair is shared BN snapshot49,
permanent page mean versus gradual fragment-residual release; no submission.
See [full acceptance and caveats](../analyses/20260914_S8_ACCEPTANCE.md).

Earlier acceptance2026-09-13 evening: S7 C512 W0/W50 both finish199/exit0 but fail.
W0 has zero macro-health-eligible validation epochs; current full-test macro=
.39856, weighted=.12641, perplexity4.25. W50 recovers usage but macro-best102
is only.44149 (perplexity23.65), current=.37628 (perplexity24.21). No stable
window in either arm. Decoder-only BN interventions remove large recon spikes
without changing codes; changing encoder BN does not produce high semantic
accuracy under the fixed mapping. Width512 alone is not a solution on V2.
Both are archived, no continuation. The proposed two-lr sweep was superseded by
the subsequently completed S8 BN/GN8 comparison above.
See [S7 acceptance](../analyses/20260913_S7_ACCEPTANCE_NEXT_LR.md).

Earlier acceptance2026-09-13: PhoneNums REPO is a partial reproduction (best-val
macro/1:1/recon=.6934/.6973/.01580; current=.7591/.7334/.04148), not stable.892.
PAPER has excellent reconstruction but poor VQ semantics. V2 C128-W50 collapses
at10→11, before the50 switch, and times out during198 (last complete197).
Decoder-only batch-stat evaluation reduces its late recon235.63→.0589 with
exactly unchanged assignments: semantic collapse and BN reconstruction failure
coexist. This locates endpoint sensitivity, not the original collapse trigger.
All three are archived and will not be continued. Next: S7 C512 W0/W50, same
data/objective/training BN, with online usage and state-neutral BN diagnostics.
See [eight full-test reports and intervention evidence](../analyses/20260913_PN_W50_ACCEPTANCE_C512.md).

Prior acceptance2026-09-12: corrected-data V2 CTRL/W100 both finish199 normally
but neither produces a registered stable window. CTRL current has macro=.58985
yet weighted=.21630, coverage=11 and eval/train recon=35.26; macro alone hides
usage/normalization failure. W100 current partly restores color (foreground
RGB=.11220, chroma=.07954), while macro=.44490 and Hungarian=.39851. Its whitened
style-to-content probe=.9395 establishes decodable leakage, not decoder dependence.
Thus old-data catalysis does not transfer under this corrected data protocol;
this is not evidence that correct color inherently prevents VQ semantics. See
[full-test evidence and mask explanation](../analyses/20260912_V2_ACCEPTANCE_AND_PHONENUMS.md).
This motivated the now-accepted S6 W50 and PN-S1 PAPER/REPO; no continuation of S5.

S3 underwent an abrupt transition around epoch 140: validation macro purity rose
from .270 at epoch 139 to .633 at 140 and .832 at 145. Reconstruction and
commitment cost rose during the transition, while the V3 hinge terms stayed near
their prior zero state. Epochs 141–149 contain eight qualifying epochs out of
nine (epoch 146 is the exception), and epoch 149 remained high, so the evidence
supports persistence in the new regime, but does not establish hysteresis; the
run ended one epoch before a full
10-epoch stability window could be observed.

Full-test S3 macro-best epoch 145 confirms macro=.834, weighted purity=.820,
1:1=.809, active=26, perplexity=25.18, coverage=25, and recon=.154.

The scratch `15→25→15` curriculum did not reproduce this state. Its best
validation macro was .589 at epoch 86, but that checkpoint had active=22 and
perplexity=5.29. Full test was macro=.564, weighted=.142, active=25,
perplexity=5.32, coverage=9, and recon=.298. No epoch reached macro .70, so no
stable-phase onset exists. The current epoch 199 checkpoint was worse and had a
full-test eval/train reconstruction ratio of 1485 (`230.25/.157`).

The run had already diverged from S3 before the relativity ramp began at epoch
60, and it lacked a simultaneous constant-r control. Therefore it is evidence
against this particular curriculum run, not a clean causal proof that higher
relativity can never catalyze a transition. The next test must branch from one
identical pre-transition checkpoint.

2026-09-10 mechanism update: frozen historical epoch80/140 checkpoints show a
switch from continuous-style reconstruction to VQ-content reconstruction.
Shuffling content codes changes the early model's output by only 2.66e-15 MSE,
but changes the later model's output by .06626. Shuffling style embeddings
within a same-color page has the opposite effect (.11246 versus 7.57e-5).
S3/C128 post-transition states reproduce the latter pattern. This supports
content leakage through the style branch before the semantic regime; it does
not establish the temporal trigger of the switch. Color information is weaker
afterward, but low ordinary probe accuracy cannot prove its complete absence.
See the [mechanism analysis and literature](../analyses/20260910_STYLE_BYPASS_PHASE_TRANSITION.md)
for sampling, scale-aware probes, interventions, and causal limitations.

S3-specific posthoc grid check later2026-09-10: both retained epochs145/149
predominantly reconstruct gray-brown glyphs across eight source colors despite
full-test macro=.834/.811 and Hungarian=.809/.771. Thus color failure is not
exclusive to C128/low-dimensional content. See the
[new diagnostic note](../analyses/20260910_S3_POSTHOC_RECON.md). The user requested
an instrumented original-protocol S3 rerun (RQ2-S4) and continuation of RQ2-S3.

Subsequent code/data audit: all26,000 PNG headers are RGB; eight styles from
each split round-trip through the actual loader without pixel error. The
generator's `canvas += np.uint8(noise)` does wrap (245+20 becomes9), whereas
clipping must precede conversion back to uint8. This is a confirmed generation
bug, not proof of the phase trigger. Existing PNG noise is fixed across epochs.
C128 epoch97 has train recon=.135606 and val recon=8940.742, between val=.162718
at96 and .337200 at98. With bounded inputs and an unbounded decoder output,
this signals extreme prediction excursions; BN-state mismatch is suspected,
not isolated. The sharp val spikes and the later train-recon/purity change near
159 must not be treated as the same observation. Peak checkpoint97 is not
retained, so a benign train/eval gap at165 does not resolve the earlier failure.

The completed C128 R1 pair sharpens the routing result. CTRL first satisfies the
registered 8-of-10 stable semantic criterion at epochs150-159. W100, which uses
page-mean style for epochs0-99, first satisfies it at76-85; after returning to
fragment style at100 it again has a10-of-10 healthy window at182-191. On full
test, CTRL best epoch166 has recon/macro/Hungarian `.15130/.91455/.91178`, while
W100 best epoch197 has `.14932/.93249/.93135`. Both use all26 codes with
perplexity about25.81 and coverage26. W100 therefore supports earlier semantic
catalysis and post-release persistence at seed0, with a modest1.96-point best
Hungarian advantage over CTRL.

Neither arm restores color: reconstruction grids become predominantly gray-brown
as semantic assignments improve. The same association occurs in historical C512
and C128 checkpoints. Treat this as a routing-state signature under the current
corrupted dataset, not as proof that code remapping causes color loss or that V3
must trade color for semantics. This motivated the S5 data-version replication,
now completed as summarized above; further scalar-weight tuning on the old PNGs
has low priority. V2 is corrected but still noisy, not clean-target supervision.

## Evidence runs

| Run | Intervention | Evidence | Conclusion |
|---|---|---|---|
| VVI-RQ2-V2-C512-K26-W50-LR1E4-BN-S0 | original BN, lower lr1e-4 | first stable35–44; best39 test macro.90738/1:1.79728; blue entirely code19; current macro.37757 | early aggregate phase but color/per-style/persistence failure; release-associated entanglement |
| VVI-RQ2-V2-C512-K26-W50-LR1E4-DGN8-S0 | matched low-lr decoder GN8 | zero health-eligible epochs; current test macro.15144/1:1.05652/perplexity1.94 | normalization replacement did not rescue semantics; collapse begins before release |
| VVI-S3-C512S512-CB512-S0 | none; r=15 | abrupt jump near 140; high state persists to 149 | single-run support for a semantic phase transition |
| VVI-H0-C512S512 | none; historical protocol | full-test 1:1=.779 at epoch 472 | high-purity state is not unique to S3 |
| VVI-RQ2-R15T25-S0 | `15→25→15` relativity curriculum | no onset; best test macro=.564 with collapsed usage | does not support RQ2-H2 in this run; normalization/trajectory confounded |
| VVI-RQ2-C128-K26-CTRL-S0-R1 | fragment style decoder | first stable window150-159; best test 1:1=.9118 | reproduces delayed semantic phase; color still fails |
| VVI-RQ2-C128-K26-W100-S0-R1 | page mean through99, fragment from100 | first stable window76-85; post-release182-191 is10/10; best test 1:1=.9313 | supports early catalysis and persistence, not color recovery |
| VVI-RQ2-V2-C128-K26-CTRL-S0 | corrected-data version, fragment decoder | best/current test1:1=.56185/.19865; current coverage11 and recon gap35.26 | no stable window, usage and normalization concerns |
| VVI-RQ2-V2-C128-K26-W100-S0 | corrected data, mean0–99 then fragment | best/current test1:1=.61781/.39851; current colored but style→content probe=.9395 | no stable window; color recovery is not disentanglement |
| VVI-RQ2-V2-C128-K26-W50-S0 | mean0–49 then fragment | collapse10→11; current197 test1:1=.0730/perplex2.77/recon235.55 | failed before release; timeout198; usage and decoder BN failures coexist |
| VVI-PN-K10-REPO-S0 | corrected digits, repo training bundle | best-val177 macro=.6934/1:1=.6973/recon=.01580; current199 macro=.7591 | useful partial reproduction, not stable paper score or complete disentanglement |
| VVI-PN-K10-PAPER-S0 | corrected digits, paper training bundle | best-val152 macro=.3097/1:1=.2343/recon=.00335 | cheaper reconstruction is not semantic success |
| VVI-RQ2-V2-C512-K26-W0-S0 | wide corrected-data fragment decoder | no health-eligible macro-best; current test macro=.3986/weighted=.1264/perplexity4.25 | fails despite width512; fixed-code usage collapse plus measured BN-mode recon failure |
| VVI-RQ2-V2-C512-K26-W50-S0 | wide corrected data, mean0–49 then fragment | macro-best102 test=.4415/1:1=.3572/perplexity23.65; no stable window | warmup improves late usage, not stable semantic alignment; BN recon failure remains |

## Hypothesis register

| ID | Hypothesis | State | Evidence / test |
|---|---|---|---|
| RQ2-H1 | The semantic transition accepts higher recon/commit cost and persists. | supported observationally in S3/C128; hysteresis untested | S3 near140; C128 near159 |
| RQ2-H2 | Raising then restoring relativity can advance the transition. | not supported in one confounded run | VVI-RQ2-S1 |
| RQ2-H3 | A temporary reduction in reconstruction weight can cross the basin boundary. | untested, secondary priority | matched available initialization required; S3 epoch130 no longer retained |
| RQ2-H4 | A temporary commitment-weight pulse can reorganize assignments. | untested | commit pulse backlog |
| RQ2-H5 | VQ rotation trick improves encoder/codebook gradient alignment and transition probability. | untested | rotation-trick backlog |
| RQ2-H6 | Train/eval normalization-state drift causes reconstruction spikes that obscure phase metrics. | directly supported for decoder mode failure across S7; original semantic-collapse trigger unproven | W0 ep89 recon277.13→.0179; W50 ep49 46.74→.0752 with decoder-only BN, exactly unchanged assignments |
| RQ2-H7 | The low-purity model bypasses VQ via fragment-specific information in the continuous style branch; the high-purity model uses VQ for shape. | strongly supported at frozen endpoints | historical epoch80/140 within-page channel shuffles; corroborated by S3/C128 post-state |
| RQ2-H8 | Removing fragment-specific style leakage while retaining page-shared style can catalyze code learning. | supported for early catalysis at seed0; permanent constraint untested | W100 reaches stability74 epochs before CTRL; MEAN-R1 never ran |
| RQ2-H9 | A100-epoch page-mean warmup creates a semantic state that persists after ordinary decoding returns. | supported on legacy data; not supported on V2 at seed0 | old W100 healthy after release; V2 never attains a stable window |
| RQ2-H10 | Corrected data generation improves foreground-color retention and/or normalization stability. | mixed: partial W100 color recovery, CTRL normalization/usage still fail | S5 full-test acceptance; data-version contrast, not attribution to one renderer fix |
| RQ2-H11 | Releasing page-mean at50 rather than100 changes the corrected-data trajectory toward stable colored semantics. | not supported for C128 seed0 | S6 collapses at11, well before release; no stable window |
| RQ2-H12 | The paper's PhoneNums training bundle reproduces K10 semantics/colors better than the upstream repo bundle. | not supported for semantics at this budget/seed; PAPER has better recon | PN-S1 primary best-val comparison; cannot isolate individual bundle parameters |
| RQ2-H13 | C512/nativeVQ512 can retain useful colored K26 semantics on V2 with the repo-compatible protocol. | not supported in matched seed0 S7 | both fail; W0 usage collapse, W50 healthy late usage but low purity; not proof of a capacity limit |
| RQ2-H14 | At C512,50-epoch page-mean warmup advances a stable semantic window that persists after release. | not supported for semantic stability; improves late usage versus W0 | neither reaches a stable window; W50 macro-best test=.44149 |
| RQ2-H15 | Lower AdamW lr stabilizes encoder/codebook/decoder trajectories and permits healthy V2 semantics. | partial support for early aggregate semantics, not persistence/color | S8 BN1e-4 has stable35–44/best test macro.907 versus S7 W501e-3 .441/no window; blue collapse remains; single seed |
| RQ2-H16 | Lower dead-code replacement pressure stabilizes code identity. | backlog, untested | W0 current has17 masses exactly reset16, but W50 fails without this late signature; actual replacement-frequency monitor needed |
| RQ2-H17 | Decoder GroupNorm improves normal-eval reconstruction stability and may enable semantic learning. | not supported as semantic/color remedy in S8; smoother poor solution | GN8 has no health-eligible epoch, current1:1.0565/recon.07266; BN learns transient alignment and better recon |
| RQ2-H18 | Releasing fragment-specific style residual erodes a learned VQ semantic state; sustained mean or gradual release preserves it. | release association and endpoint dependence supported; maintenance interventions untested | BN49→55 val macro.861→.654; BN199 style/content permutation both affect outputs; propose matched snapshot49 branches |
| RQ2-H19 | Removing decoder normalization avoids mode/range distortions and improves colored semantic reconstruction. | supported for late recon/color and partial alignment improvement; not stable high purity | S9 current1:1.492 versus S8 BN.201, recon.00449; best34 fails active26 gate; no stable window |
| RQ2-H20 | Removing encoder BN in addition to decoder BN improves stable VQ assignments across styles. | not supported in matched S9 seed0 | All-NoNorm current1:1.296 versus Decoder-NoNorm.492; both broad usage, All-NoNorm stronger style-specific semantic mismatch |
| RQ2-H21 | The useful PhoneNums REPO bundle transfers to16 hexadecimal contents with its renderer and step budget retained. | supported in single seed; archived193731 | Best1:1=1/current.999731; late window158–167, color/BN limitations remain |
| RQ2-H22 | Permanent page-mean decoder input maintains stronger Hex16 semantic alignment than ordinary fragment decoding. | supported for earlier alignment and better current color; archived193733 | Window3–12 versus158–167; both endpoints semantic-successful, so mean is not necessary |
| RQ2-H23 | Uppercase failure reproduces under the verified current Hex training bundle. | supported for poor130k endpoint; extended budget pending | S10 current val1:1=.21571; not a formal new test result |
| RQ2-H24 | Permanent page-mean decoding also improves uppercase semantic alignment. | higher130k endpoint than ordinary, not sufficient for semantic success | S10 current val1:1=.39036 versus.21571; R1 retains permanent mean |
| RQ2-H25 | Optimizer exposure and LR-by-step confound the Hex-versus-uppercase comparison. | confound documented; split interventions below | Hex REPO130k=.221 versus500k=.9997; renderer/sample-count confounds remain |
| RQ2-H25a | Extending current S10 trajectories to~500k updates recovers stable uppercase semantics. | user-authorized2026-09-17; R1 running197821/197822 | Resume199→769 with original epoch-based LR and training state; not a fresh step-LR match |
| RQ2-H25b | A fresh uppercase run with Hex-matched LR by cumulative step improves transfer. | untested backlog, not submitted | Must preregister LR-by-step and separate from simple budget extension |

## Tried, rejected, and backlog

- Tried observationally: constant `r=15`; successful once at seed 0, onset near
  140, but not yet a causal catalyst.
- Rejected for transfer: the tested smooth relativity curriculum did not produce
  a transition and must not be copied to CB4.
- Observed confound: recurrent validation-reconstruction explosions with normal
  train reconstruction, consistent with unstable BatchNorm running statistics.
- Updated backlog order: matched style-shortcut removal, recon pulse, commit
  pulse, then rotation trick. Each
  must share an initialization checkpoint with its constant-objective control.

## Next decision gate

Active priority is S10 R1 to500500 updates; HEX-S1 is completed/archived.
Use unchanged uppercase gates stated below and compare both macro-best and current;
best-only success is transient. S9 and snapshot49 interventions remain deferred.
The following Hex gate remains the historical comparison contract.
Hex full-test uses10,000 pages/160,000 fragments: macro/weighted/Hungarian>=.75,
active16/perplexity>=13/coverage>=15/recon<=.30. A stable10-epoch window needs8
epochs with macro/Hungarian>=.70 and the same usage floors. Best-only passing is
transient; current must retain the semantic gate. Report per-style and0–9/a–f
accuracy under one global mapping, plus colors and leakage. No epoch50 release
exists in either Hex arm. MEAN-only success requires a separate ordinary
single-fragment compatibility check before downstream integration.

S5/S6/PN-S1/S7/S8/S9 are archived. Do not resume these current checkpoints.
Historical S8 maintenance proposal (deferred, not submitted): shared BN snapshot49, restore optimizer/scheduler/
scaler and match newly seeded continuation RNG/data order (old checkpoints lack
saved RNG), then compare permanent page mean with gradual residual release.
Decoder/encoder remain BN, lr1e-4 and all other settings fixed. It tests
maintenance, not guaranteed repair of already-failed blue/color representations.
The Hex configs supersede this proposal for the next two slots.

For historical uppercase comparisons, full-test2600 pages remains the protocol: macro>=.75,
active26, perplexity>=20, coverage>=24, recon<=.30, colors/weighted/probes separate.
Stability remains8/10 epochs with macro>=.70, active26, perplexity>=20, coverage>=24.
For new runs additionally report full per-style code usage and accuracy under
one shared global Hungarian mapping; do not hide blue collapse by rematching or
discarding a style. Keep weighted purity and color errors. Any extra per-style
acceptance floor must be preregistered, not applied retroactively. If mean-only
preserves semantics but colors still fail, isolate the shared-style representation;
if both branches degrade, revisit encoder/VQ/EMA. Small-batch intervention values
never replace formal selection, and GN's decoder-only no-op is not recovery.

Operational pause2026-09-10 at15:10:58+04:00: user requested all three jobs stop
to release resources. Jobs cancelled; CTRL retains current37, W100 current40,
MEAN never started. No scientific acceptance/rejection is inferred. Await user
continuation; preserve the original200-epoch targets and absolute100-epoch switch.

2026-09-10 user update: prioritize **VVI-RQ2-S3**, three matched200-epoch
C128/K26 scratch runs: ordinary decoder, permanent page-mean style decoder,
and100-epoch page-mean warmup then ordinary decoder. See
[SETTINGS](../SETTINGS.md#vvi-rq2-s3--c128-style-bypass-ablation-and-warmup).
Compare simultaneous-control onset, leakage, color recovery and post-release
persistence. Every25 completed epochs retain a checkpoint and fixed26×16 grid.
The historical-epoch80 proposal below is deferred, not an active task. Persistence
alone is not a demonstration of hysteresis.

S3 epoch130 is not present in its run directory; do not assume it is recoverable.
The available historical `uppercase_letters_run2_batch32/cp_epoch80.pt` is a
measured low-purity, style-bypass initialization for a matched 40–60 epoch pilot:

- Control: original decoder inputs, constant `r=15`, recon weight=1.
- RQ2-H8: decoder receives the page-mean style embedding broadcast to each
  fragment; keep raw embeddings for V3 and all loss weights unchanged.

Match RNG/data order and optimizer/scheduler initialization. Inspect legacy
checkpoint state first; missing optimizer state requires identical fresh
optimizers and an explicit warm-start description. Monitor code dependence,
within-page style dependence, raw losses, style/decoder sensitivity, and color
reconstruction every 50 steps around onset. Earlier healthy purity together
with preserved color would support a useful catalyst; purity alone is not
successful content-style disentanglement. This page-sharing intervention applies
to same-style UppercaseLetters pages, not mixed-style V5 triplets. The detailed
protocol is proposed only; no formal run was submitted for this diagnosis.
