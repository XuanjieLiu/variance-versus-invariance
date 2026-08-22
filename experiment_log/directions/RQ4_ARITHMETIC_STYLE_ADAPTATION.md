# RQ4 bridge - Arithmetic-guided unseen-style adaptation

RQ4 is implemented and evaluated in V5. Its canonical hypotheses, experiment
matrix, success gates, and conclusions live in the V5
[AGUSA direction](../../../v5/research/directions/RQ4_AGUSA.md) and
[research dashboard](../../../v5/EXPERIMENT_SUMMARY.md).

VVI remains the source of truth for the C128/S512/K26 source model and its
semantic acceptance evidence. The accepted artifact is macro-best epoch 165
from `20260807-1506__VVI-RQ1A-C128-K26-S0`.

## Bridge Status

- OOD color data generation and frozen-path CUDA validation are complete in V5.
- C128 categorical Teacher A completed 20k steps and passed its gate. Selected
  step 16500 has number accuracy `.7662` and exact-code accuracy `.6190`.
- The next V5 decision is a matched C128 nearest-VQ Teacher B, followed by the
  registered OOD zero-shot and seed-0 adaptation controls.

Do not duplicate future RQ4 conclusions here. Update the V5 ledger and retain
this page only as the VVI-to-V5 dependency bridge.
