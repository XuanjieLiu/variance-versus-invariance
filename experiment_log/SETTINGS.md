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
| VVI-RQ2-S3 | RQ2; RQ1-A control | RQ2-H8; RQ2-H9 | active; CTRL/W100 running, MEAN queued (QOS) |
| VVI-RQ3-S2 | RQ3 | RQ3-H2 | planned; cold-start control |
| VVI-RQ13-S1 | RQ1-B; RQ3 | RQ1-H5; RQ3-H4 | completed; both VVI gates passed, addition probes failed |
| VVI-RQ1A-S1 | RQ1-A; RQ2 observation; RQ3 pilot | RQ1-H7; RQ1-H8; RQ1-H9 | completed; seed-0 K26 threshold bracketed in `(64,128]` |
| VVI-RQ13-S2 | RQ1-A; RQ3 | RQ1-H10; RQ1-H11; RQ3-H6 | completed; C32 failed, C128 transient only |
| VVI-RQ13-S3 | RQ1-B; RQ3 | RQ1-H12; RQ3-H7 | planned; K52-WARM to K128/D128 generalized expansion |
| VVI-RQ4-S1 | RQ4 bridge | V5 RQ4-H1; RQ4-H2; RQ4-H3 | transferred to V5; Teacher A gate passed |

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
