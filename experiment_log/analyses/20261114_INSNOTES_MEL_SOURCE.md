# InsNotes mel source acceptance, 2026-11-14

## Outcome and scope

`20261106-1118__VVI-IBP-INSNOTES-MEL-K12-S0` (job280419) completed normally
on2026-11-06: zero-based epoch1999, 2000 natural epochs, 60000 updates, exit0.
The source semantic gate FAILED. Do not start IBP target fine-tuning with this
checkpoint or interpret this as a negative result about persistence after
successful acquisition. Cross-project decisions live in
[V5 RQ6](../../../v5/research/directions/RQ6_INDUCTIVE_BIAS_PERSISTENCE.md).

Acceptance uses ALL14400 validation notes, exactly100 per pitch x instrument,
120 whole recordings. Test remains sealed because source validation failed.
This is full-validation source-gate acceptance, not claimed full-test success.

## Final and historical results

| Metric | Final checkpoint | Historical peak, if relevant |
|---|---:|---|
| Hungarian one-to-one | .4463889 | .7157639 at epoch890/update26730 |
| Usage-weighted code purity | .4618056 | .7290278 at epoch1060/update31830 |
| Macro atom purity | .6495808 | .8603317 at epoch890 |
| Active codes | 11/12 | activation alone did not ensure semantics |
| Usage perplexity | 7.20968 | source gate >=10 |
| Majority-label coverage | 9/12 | source gate also requires semantic accuracy >=.95 |
| Train epoch-average reconstruction MSE | .3535705 | sampled training groups |
| Full-validation reconstruction MSE | 1.7616249 | current frozen model |
| Train / validation style loss | .9342409 / .9387574 | did not approach zero |

The final validation V3 loss1.792722 comprises content0, style.938757,
sample.103289, fragment.750676. Training content/sample/fragment losses are0.
Validation commitment0 reflects the VQ library's eval behavior, not evidence
that continuous encoder features exactly coincide with codebook atoms.

No evaluation met all macro-best retention floors (12active, ppl>=10,
coverage>=10). Only `cp_current_epoch1999.pt` exists. Historical peaks are CSV
observations, NOT recoverable best checkpoints. Proposed future protocol change:
retain a label-free best-validation-loss checkpoint independent of source gate;
register the checkpoint-policy exception before another formal launch.

## Representation diagnostics

Existing epoch1999 recording-disjoint probes fit5 and score5 validation
recordings per instrument (7200notes each), with PCA32/ridge.01 fit only on the
fit half. Chance for either balanced factor is1/12=.08333.

- Pre-VQ content -> pitch: raw.32222, whitened.92875.
- Raw style -> instrument: raw.08681, whitened.10681.
- Raw style -> pitch leakage: whitened.37736.
- Content/style first-PC fractions: .992757 / .9999943.
- Mean batch style-between/style-within ratio .91864 versus required15.

Pitch information remains linearly decodable after scale-aware preprocessing,
but is poorly exposed by nearest-code assignments. Style has very weak linear
instrument decodability and still carries pitch information. These are probes,
not proofs that nonlinear information is absent. Dominant PC1 indicates severe
anisotropy, not by itself a proof of complete collapse.

## Frozen checkpoint interventions

GPU acceptance job298700 onws-l5-007 reran complete validation for each row,
with identical data order and batch32. No optimizer was created; parameters
require no gradients, VQ updates are disabled, BN buffers and module modes/RNG
are restored after each intervention. Full model fingerprint is unchanged.

| BN statistics used | Reconstruction MSE | Hungarian one-to-one | Codes changed vs ordinary eval |
|---|---:|---:|---:|
| running statistics, ordinary eval | 1.761625 | .446389 | 0 |
| encoder batch statistics only | 1.482904 | .441458 | .209097 |
| decoder batch statistics only | 1.392763 | .446389 | 0 |
| both batch statistics | .332303 | .441458 | .209097 |

**BN explains much of the reconstruction train/eval gap, not the semantic
failure.** Batch-stat evaluation is a reversible diagnostic, not a repaired or
accepted checkpoint, and is batch-composition dependent.

An eval-only same-group mean-style substitution raises MSE to7.167039 without
changing code assignments. This shows dependence on fragment-specific style
variation. Such variation can include nuisance dynamics as well as pitch;
this intervention does not alone prove what is encoded. It does NOT test a
model trained with mean-style decoding from the beginning.

## Listening and validation artifacts

Under the ignored run's `acceptance_20261114/`:

- `acceptance.json`: checkpoint/manifest hashes, histories, complete confusion
  counts, BN interventions, repeated-eval equality and zero-state-change checks.
- `audio.html`: offline listening table; no external assets or server required.
- `audio_manifest.json`: selection, split, inversion parameters and shared gains.
- `audio/`: original / input-mel roundtrip / final-model reconstruction for
  Soprano Sax, Pipe Organ and Trumpet, each playing C/E/G.

Samples are preselected first validation recordings for style IDs0/1/4, not
selected by quality. Re-rendering uses the identical seeded FluidSynth recipe;
full recording feature hashes match the immutable dataset. Audio reconstruction
uses `exp(logmel)-1e-5`, magnitude power1, Slaney128bands, FFT1024/hop512 and
seeded64-iteration Griffin-Lim. The roundtrip control separates missing
phase/filterbank detail from model error. All three tracks of one instrument
share attenuation; no independent loudness normalization is applied.

All159 unit tests passed. Repeated frozen evaluation was identical and every
parameter/buffer remained unchanged. An initial interactive acceptance attempt
298699 was killed by signal9 before evaluation; batch retry298700 completed
exit0. No training jobs were launched during acceptance.

## Next experiments, proposed only

1. Reproduce the original repository optimizer/VQ bundle with the same mel data:
   AdamW lr1e-3, WD.1, commit.1, EMA.98, dead threshold16. Current run used the
   paper-text bundle (Adam, WD0, commit.01, EMA.95, threshold3.2). This is an
   explicitly declared bundle comparison, not an isolated weight-decay claim.
2. Matched ordinary-fragment versus same-sample mean-style SOURCE training.
   A shared style input cannot individually encode the12 different pitches;
   test whether content usage and instrument readout recover. Keep raw z_s in
   the original V3 losses. For later IBP constraint-removal FT, also remove mean
   pooling, otherwise an explicit organization constraint remains in place.
3. Diagnose normalization independently: training-only BN-stat recalibration
   on a COPY of the final model, followed by unchanged validation. If unstable,
   register a matched decoder-GN or lower-LR arm; do not silently switch inference
   to batch stats or claim this alone fixes the poor symbols.

Prefer these controlled changes over blind continuation, supervised replacement
or widening the codebook immediately. Keep mel inputs as requested. Mel128 also
reduces the existing seven frequency-downsampling stages to one final frequency
cell (original512 bins -> four); revising that backbone is a subsequent declared
architecture ablation, not a hidden reproduction fix.

References: [V3 Appendix B.2](https://arxiv.org/html/2407.03824v3#A2.SS2)
for paper optimizer/objective values;
[librosa mel inversion](https://librosa.org/doc/0.10.2/generated/librosa.feature.inverse.mel_to_audio.html)
and [Griffin-Lim](https://librosa.org/doc/0.10.2/generated/librosa.griffinlim.html).
