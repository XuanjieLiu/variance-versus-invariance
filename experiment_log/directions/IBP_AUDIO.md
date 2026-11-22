# IBI audio bridge (historical IBP run IDs)

The canonical hypotheses, supervised baseline and fine-tuning comparisons live
in [V5 RQ6](../../../v5/research/directions/RQ6_INDUCTIVE_BIAS_PERSISTENCE.md).
Do not duplicate cross-project conclusions here.

VVI owns the InsNotes mel source reproduction (IBP-AUDIO-S1), run/checkpoint
provenance and source acceptance. See [manual](../../manuals/insnotes_mel_ibp.md),
[settings](../SETTINGS.md#ibp-audio-s1--insnotes-mel-source-pretraining),
[active](../ACTIVE_RUNS.md) and [archive](../ARCHIVE.md).

Status2026-11-14: source job280419 completed2000epochs/60k updates, but failed
the source semantic gate. [Acceptance](../analyses/20261114_INSNOTES_MEL_SOURCE.md)
contains full balanced validation, BN interventions, and matched listening
examples. Audit298700 passed159tests and verified zero model-state change.
No accepted source, OOD FT or supervised baseline yet; test remains sealed.

Follow-up accepted2026-11-19: [IBP-AUDIO-S2](../analyses/20261119_INSNOTES_REPO_ACCEPTANCE.md)
REPO fragment/MEAN jobs298789/298790 completed2000epochs each. Loss-selected
MEAN improves reconstruction and mapping but still fails the source gate;
full-val/per-instrument diagnostics and best audio are archived. Neither
source is authorized for OOD FT. Cross-project interpretation remains in RQ6.

The authorized [R1 jobs309229/309230](../analyses/20261119_INSNOTES_REPO_CONTINUATION.md)
completed9999/300k updates, accepted2026-11-20. [Acceptance](../analyses/20261120_INSNOTES_10000E_ACCEPTANCE.md)
finds lower reconstruction but worse1:1: loss-best.332083/.791667, final
.387778/.784722. Global pitch permutations remain; neither qualifies for OOD.
Test remains sealed. IBI is the preferred research name; preserve IBP artifact
IDs. The separate [RD-V3 image source pair](../../manuals/representation_distillation.md)
uses healthy HEX RANK2, not these audio checkpoints.
