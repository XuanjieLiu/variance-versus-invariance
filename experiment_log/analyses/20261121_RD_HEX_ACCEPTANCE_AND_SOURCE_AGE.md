# RD-V3 HEX acceptance and shorter source-age replica

Date:2026-11-21, Asia/Dubai. Acceptance below uses complete source-validation
logs and retained checkpoint metadata. Independent checkpoint re-evaluation is
queued as job312730, NOT yet completed. No source test or OOD metrics selected
the learning rate or the four source ages.

## Completed source pair

Both jobs310475/310476 reached zero-based199,500000updates, wrapper exit0.
Validation has10000pages/160000fragments, balanced across16symbols and8styles.
Best means minimum validation RD objective. Macro-best selects the same epoch
in both runs; these are not additional independent source timepoints.

| LR | Loss-selected epoch | Val RD total | Val reconstruction | Final RD total | Final reconstruction | Best/final Hungarian and code purity | Active / perplexity |
|---|---:|---:|---:|---:|---:|---|---|
|1e-4|186|.01775805|.01754319|.01799693|.01773133|1.0 / 1.0|16 / 16|
|3e-4|182|.01988253|.01964442|.02024560|.02004790|1.0 / 1.0|16 / 16|

Paths: `logs/20261120-0935__VVI-IBI-RD-HEX-RANK2-LR1E4-S0` and
`logs/20261120-0935__VVI-IBI-RD-HEX-RANK2-LR3E4-S0`.
Teacher full-val recon=.04327776, Hungarian=.99998125. Both students pass the
basic discrete/reconstruction readiness thresholds, not a complete IBI gate.

Whitened raw content -> style accuracy is teacher.77661, LR1e-4 best.88086,
LR3e-4 best.77661. LR1e-4 exceeds the original +.05 inspection tolerance;
do not call it fully behavior-matched. Discrete-code -> style accuracy stays
.125 (chance). Whitened raw style -> content is high for teacher (.98462)
and students (.99316/.99561); the style head is not content-free. Foreground
chroma RMSE improves teacher.11563 -> .04281/.04550. Choose LR1e-4 by the
source-val objective and retain these caveats in the exploratory FT comparison.

## Source ages

Only two unique epochs per run remain:186/199 or182/199. Early weights cannot
be recovered from CSVs. User authorized a new same-seed LR1e-4 trajectory,
not resume and not an independent-seed replication.

The old LR1e-4 run already has Hungarian.99999375 at2500updates and1.0 at7500.
Completed1/5epochs have val RD total.03055/.02276. Recon still fluctuates
(completed20 total=.05364); mapping saturation is not complete convergence.
Reduce200epochs to50; retain fixed ages1/5/20/50, corresponding to
2500/12500/50000/125000updates and zero-based files0/4/19/49. Never substitute
a different age after viewing an OOD outcome.

New source: `20261121-1519__VVI-IBI-RD-HEX-RANK2-LR1E4-S0-50E`, job312731,
depends on successful preflight312730. GPU capacity currently prevents start.

## Execution

- `snapshot_epochs` unions with periodic snapshots and does not alter loss or
  best selection. Atomic files survive current/best rotation.
- A separate SLURM dispatcher calls the original VVI training wrapper, watches
  completed snapshots, and submits three V5 FT jobs per age. No source
  pause/resume or consumption of its RNG.
- Preflight tests schedule/rotation, RD gradients,12 exact protocol matches,
  no RD loss in FT, and three10update CUDA paths from an existing RD model.
  Both completed source best/final states are audited with the common full-val
  collector, unchanged model/BN/VQ/teacher state and logged-loss reproduction.
- Audit output: `logs/rd_hex_acceptance_20261121/summary.json`, created only
  when that queued job runs. Test/OOD data are not used for source acceptance.
- [V5 RQ6](../../../v5/research/directions/RQ6_INDUCTIVE_BIAS_PERSISTENCE.md#rd-v3-source-age-transfer-2026-11-21)
  owns the FT protocols and live downstream job receipt.
