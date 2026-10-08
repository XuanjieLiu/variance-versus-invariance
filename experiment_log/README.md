# VVI research dashboard

This directory separates cross-run research conclusions from the operational run
ledger. Direction files synthesize evidence; `SETTINGS.md` defines controlled
experiments; `ACTIVE_RUNS.md` and `ARCHIVE.md` own run lifecycle state.

| Direction | Current conclusion | Confidence | Active setting | Next decision |
|---|---|---|---|---|
| [RQ1: content width and native low-D VQ](directions/RQ1_LOW_DIMENSION.md) | Compact native VQ remains the application goal; successful Hex16 codes are nearly collinear. Test a rank floor before assuming high purity implies useful arithmetic geometry. | geometry observed; intervention untested | VVI-HEX-RANK-S1: 222839/222840 | Compare rank4/rank2 against Hex MEAN: retain semantic quality while broadening native-code spectrum. |
| [RQ2: phase catalysis](directions/RQ2_PHASE_CATALYSIS.md) | Hex16 succeeds at500k; S10 uppercase at130k ends with val1:1=.216/.390, active26/ppl~24. User authorizes continuation, not a new data/normalization intervention. | one matched seed; uppercase interim validation only | S10 R1 running197821/197822 | Extend both current199 to769/500500steps, retain epoch-based LR; full-test and late stability after continuation. |
| [RQ3: joint K/D scaling](directions/RQ3_CODEBOOK_SCALING.md) | Warm K52 and PCA-warm K104 succeed, while flat K128 scratch is unstable even at C128. Healthy usage is not enough: semantic parent structure must be inherited or explicitly constrained. | high | none | Generalize the successful PCA-warm transform from K52 to K128 with D128. |
| [RQ4 bridge: arithmetic-guided unseen-style adaptation](directions/RQ4_ARITHMETIC_STYLE_ADAPTATION.md) | V5 Teacher A passed its gate; V5 is canonical for all AGUSA hypotheses and conclusions. | teacher qualified | none in VVI | Follow the [V5 dashboard](../../v5/EXPERIMENT_SUMMARY.md): matched Teacher B, then OOD controls. |

## Metric contract

Update2026-10-08: [v3_rank](../manuals/v3_rank.md) is the user-confirmed name of the
native-VQ effective-rank-floor method (RQ1-H13). A matched Hex16 permanent-MEAN
rank4/rank2 pair is running as222839/222840 (VVI-HEX-RANK-S1). Codebook K16
and nativeD512 stay unchanged; the intervention is only the rank floor.

- Primary selection metric: validation macro atom purity over active codes.
- PhoneNums PAPER/REPO additionally keep best validation total loss; this
  label-free checkpoint is their primary paper-comparison acceptance point.
- Anti-degeneration metrics: usage-weighted purity, active codes, usage
  perplexity, dominant-label coverage, and dominant-code count min/max/CV.
- Hungarian accuracy is diagnostic only when `K != content_count`.
- For `K>26`, atom purity is necessary but not sufficient. Also report native
  VQ alias within/between geometry and whether a small native-space addition
  probe generalizes across styles.
- The K104 probes show that even purity plus the current alias-distance metrics
  are not sufficient for arithmetic: a per-image alias target can remain
  multimodal. Keep VVI acceptance and addition acceptance as separate gates.
- RQ2-S1 shows that unconstrained macro purity can favor rarely used atoms:
  its macro-best had macro=.564 but weighted purity=.142 and perplexity=5.32.
  Before the next formal run, checkpoint selection should rank macro purity only
  among candidates meeting configurable active/perplexity/coverage floors while
  still retaining exactly one best plus current checkpoint.
- Formal conclusions use the full test split (2,600 pages UppercaseLetters,
  10,000 pages PhoneNumsV2/HexDigitsV2); quick subsets and
  smoke runs never enter the archive.

## Lifecycle

1. Register a hypothesis in one direction file and define its setting in
   `SETTINGS.md`.
2. Add a formal run to `ACTIVE_RUNS.md` before submission.
3. On completion, failure, timeout, or cancellation, remove it from ACTIVE and
   add one row to `ARCHIVE.md`.
4. Update only the evidence synthesis and decision gate in the relevant
   direction file; do not duplicate long conclusions in the run tables.

## Prioritized next experiment queue

Latest user decision2026-10-08: **VVI-HEX-RANK-S1**. Two scratch seed0 runs on
Hex16 MEAN,200epochs/500k steps each, `method: v3_rank`, targets4/2,weight.01.
Configs: `configs/hex/v2/cfg_vvi_hex_k16_mean_rank{4,2}_seed0.yaml`. Keep BN,
optimizer/EMA/data and all diagnostics/retention matched to Hex MEAN. See
[setting](SETTINGS.md#vvi-hex-rank-s1--v3_rank-on-hex16-permanent-mean).
Both running from16:24+04 on2026-10-08, rank4 onws-l1-008 and rank2 onws-l1-014.
See [submission and verification](analyses/20261008_V3_RANK_HEX_SUBMISSION.md).

Prior user decision2026-09-17: **S10 R1 budget continuation**. Originals
196481/196482 completed199/exit0 and are archived with validation-only interim
metrics, not declared new full-test acceptance. Both current199 checkpoints are
extended for570 additional epochs to769/500500 cumulative updates. Keep optimizer,
scheduler/scaler, decoder modes and all data/loss/BN settings; no LR reset or
step-based scheduler rewrite. R1 registered before submission; both full-epoch GPU
preflights and15 unit tests passed. Running197821/197822 onws-l1-002/ws-l1-004
from2026-09-17 13:20+04. Preserve25-epoch snapshots (max25 files per new run) and inherit histories.
This tests longer exposure, not a fully matched learning-rate/data comparison.
See [continuation protocol](analyses/20260917_S10_CONTINUATION.md).

Prior user decision2026-09-16: **S10 uppercase Hex-protocol replication**.
HEX-S1 is accepted/archived: both best1:1=1; current=.999731/.999938 with all16
codes healthy. First stable windows158–167(REPO)/3–12(MEAN). MEAN current colors
are substantially better. Old uppercase is not a pure26-versus16 comparison:
130k versus500k updates,25px versus48px glyphs,estimated foreground15% versus30%.
Keep old data untouched; requested ordinary/permanent-mean pair passed full-epoch
preflight and ran196481/196482 on ws-l1-009/ws-l1-014 from21:58, now completed.
Original200 natural epochs were130k updates; the explicitly authorized extension
above does not resample or regenerate pages. See
[full acceptance and comparison](analyses/20260916_HEX_ACCEPTANCE_UPPERCASE.md).

Prior user decision2026-09-15: **HEX-S1 progressive extension**. Independent
HexDigitsV2 contains0–9/a–f at the original PhoneNums rendering scale and100k-page
budget. Match REPO BN/lr.001 and compare ordinary fragment versus permanent
page-mean decoding for200 epochs. No simultaneous VQ reduction or addition training.
Data tests and84 regressions passed;100k-page generation and all-PNG audit passed.
Both complete-epoch GPU smokes passed, including full10k-page validation and
matched initialization/sampler hashes. Temporary smoke directories were removed;
two formal runs registered before submission, with24h resource limits.
REPO193731 and MEAN193733 started2026-09-15 11:21 and both completed199/exit0
early2026-09-16; now archived. Both formal initial model hashes match each other
and the validated smoke, but this does not guarantee bitwise training trajectories.
See [HEX16 protocol](analyses/20260915_HEX16_PROTOCOL.md).

S9 NoNorm192280/192281 completed199/exit0 and was archived on2026-09-15. Both
improve late recon/colors, but neither has a stable semantic window; current1:1
is.49175/.29590 with broad usage, and style-content readouts remain high. No
continuation. Six full-test role reports and figures are saved; see
[S9 acceptance](analyses/20260915_S9_ACCEPTANCE.md).
The S8 snapshot49 proposal below remains deferred, not submitted.

Prior acceptance (2026-09-14 morning): S8 **independent normalization** is accepted.
Both finish normally, but neither provides persistent colored semantics. BN
best39 reaches macro.907 and a35–44 stable window, with complete blue-style
collapse hidden by global macro. GN8 fails from0 despite smoother recon curves.
The then-proposed shared snapshot49 branch pair (permanent mean versus gradual residual
release) remains a maintenance/causal backlog, not a promised color fix or an
active continuation of current199. The user selected S9 above instead.
See [S8 acceptance](analyses/20260914_S8_ACCEPTANCE.md).
S7 remains accepted/archived, neither continued; its five available checkpoint
roles have full-test reports/new figures and W0 had no eligible macro-best.
See [S7 acceptance](analyses/20260913_S7_ACCEPTANCE_NEXT_LR.md).
Older PN/C128 experiments remain archived and old MEAN/S4 C512 remain cancelled.
One seed and unequal cross-task step budgets cannot isolate dimension causality.

Historical sequence: after the15:10 resource pause, the user authorized continuation:
resume CTRL from38 and W100 from41, start the never-run MEAN from0, and add the
original150-epoch C512 S3 reproduction. See [S3 posthoc diagnosis](analyses/20260910_S3_POSTHOC_RECON.md)
and SETTINGS for scope and resume provenance. The later cancellation above
supersedes the authorization to start the two queued arms.

1. **VVI-RQ13-S3 K52-WARM→K128/D128.** Generalize local warm expansion while
   keeping the discrete interface 128D and the encoder/decoder pathway 512D.
2. **VVI-RQ1A C96/K26 midpoint.** Narrow `(64,128]`, then replicate the boundary.
3. **Alias-invariant arithmetic objective.** Explain and remove the multimodal
   per-image alias target before interpreting another small-adder capacity test.
4. **VVI-RQ2 matched branch pair.** Use the retained historical epoch80 and
   compare original decoding with page-mean style decoding to test continuous
   content leakage. See the [2026-09-10 mechanism analysis](analyses/20260910_STYLE_BYPASS_PHASE_TRANSITION.md).
5. **VVI-RQ3-S2 full-width K52 cold start.** The C64/K52 pilot failed but is
   content-width-confounded; retain the C512 control only as a mechanism test.
