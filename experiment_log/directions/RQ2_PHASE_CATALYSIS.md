# RQ2 — Catalyzing the semantic phase transition

## Research question and success criteria

Can the abrupt move from a low-purity reconstruction basin to a high-purity
semantic basin be induced earlier and remain stable after the catalyst is
removed? A stable phase is a 10-epoch window with at least eight epochs satisfying
macro `>=0.70`, active codes `=26`, perplexity `>=20`, and coverage `>=24`.

## Current conclusion

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

## Evidence runs

| Run | Intervention | Evidence | Conclusion |
|---|---|---|---|
| VVI-S3-C512S512-CB512-S0 | none; r=15 | abrupt jump near 140; high state persists to 149 | single-run support for a semantic phase transition |
| VVI-H0-C512S512 | none; historical protocol | full-test 1:1=.779 at epoch 472 | high-purity state is not unique to S3 |
| VVI-RQ2-R15T25-S0 | `15→25→15` relativity curriculum | no onset; best test macro=.564 with collapsed usage | does not support RQ2-H2 in this run; normalization/trajectory confounded |

## Hypothesis register

| ID | Hypothesis | State | Evidence / test |
|---|---|---|---|
| RQ2-H1 | The semantic transition accepts higher recon/commit cost and persists. | supported observationally in S3/C128; hysteresis untested | S3 near140; C128 near159 |
| RQ2-H2 | Raising then restoring relativity can advance the transition. | not supported in one confounded run | VVI-RQ2-S1 |
| RQ2-H3 | A temporary reduction in reconstruction weight can cross the basin boundary. | untested, secondary priority | matched available initialization required; S3 epoch130 no longer retained |
| RQ2-H4 | A temporary commitment-weight pulse can reorganize assignments. | untested | commit pulse backlog |
| RQ2-H5 | VQ rotation trick improves encoder/codebook gradient alignment and transition probability. | untested | rotation-trick backlog |
| RQ2-H6 | Train/eval normalization-state drift causes reconstruction spikes that obscure phase metrics. | suspected, not isolated | RQ2-S1 current gap=1485; smaller spikes also occur in S3 |
| RQ2-H7 | The low-purity model bypasses VQ via fragment-specific information in the continuous style branch; the high-purity model uses VQ for shape. | strongly supported at frozen endpoints | historical epoch80/140 within-page channel shuffles; corroborated by S3/C128 post-state |
| RQ2-H8 | Removing fragment-specific style leakage while retaining page-shared style can catalyze code learning. | testing; MEAN queued | RQ2-S3 CTRL versus MEAN; raw style for V3, recon weight=1 |
| RQ2-H9 | A100-epoch page-mean warmup creates a semantic state that persists after ordinary decoding returns. | testing | RQ2-S3 W100 versus CTRL/MEAN; absolute epoch100 switch |

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
