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
supports a new attractor with hysteresis; the run ended one epoch before a full
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

## Evidence runs

| Run | Intervention | Evidence | Conclusion |
|---|---|---|---|
| VVI-S3-C512S512-CB512-S0 | none; r=15 | abrupt jump near 140; high state persists to 149 | single-run support for a semantic phase transition |
| VVI-H0-C512S512 | none; historical protocol | full-test 1:1=.779 at epoch 472 | high-purity state is not unique to S3 |
| VVI-RQ2-R15T25-S0 | `15→25→15` relativity curriculum | no onset; best test macro=.564 with collapsed usage | does not support RQ2-H2 in this run; normalization/trajectory confounded |

## Hypothesis register

| ID | Hypothesis | State | Evidence / test |
|---|---|---|---|
| RQ2-H1 | The semantic transition temporarily accepts higher recon/commit cost and exhibits hysteresis. | single-run supported | S3 epochs 139–149 |
| RQ2-H2 | Raising then restoring relativity can advance the transition. | not supported in one confounded run | VVI-RQ2-S1 |
| RQ2-H3 | A temporary reduction in reconstruction weight can cross the basin boundary. | next matched test | branch pair from S3 epoch 130 |
| RQ2-H4 | A temporary commitment-weight pulse can reorganize assignments. | untested | commit pulse backlog |
| RQ2-H5 | VQ rotation trick improves encoder/codebook gradient alignment and transition probability. | untested | rotation-trick backlog |
| RQ2-H6 | Train/eval normalization-state drift causes reconstruction spikes that obscure phase metrics. | suspected, not isolated | RQ2-S1 current gap=1485; smaller spikes also occur in S3 |

## Tried, rejected, and backlog

- Tried observationally: constant `r=15`; successful once at seed 0, onset near
  140, but not yet a causal catalyst.
- Rejected for transfer: the tested smooth relativity curriculum did not produce
  a transition and must not be copied to CB4.
- Observed confound: recurrent validation-reconstruction explosions with normal
  train reconstruction, consistent with unstable BatchNorm running statistics.
- Backlog order: matched recon pulse, commit pulse, then rotation trick. Each
  must share an initialization checkpoint with its constant-objective control.

## Next decision gate

Restore the recoverable S3 epoch-130 checkpoint and run a paired 40-epoch test:

- Control: constant `r=15`, recon weight=1.
- RQ2-H3: recon weight ramps `1→0.5` over epochs 131–135, holds at .5 through
  145, and returns to 1 by epoch 155; relativity and commit stay fixed.

Both branches use the same seed, data order, optimizer/scheduler state, and
validation protocol. A catalyst is supported only if its first stable window
precedes the control by at least five epochs, remains stable after weight
restoration, and full-test macro>=.75/recon<=.30/eval-train ratio<=2. If the
control itself cannot reproduce the transition, add RNG-state checkpointing
before interpreting pulse efficacy.
