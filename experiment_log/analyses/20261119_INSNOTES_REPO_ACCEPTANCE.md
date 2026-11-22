# InsNotes mel REPO / REPO-MEAN acceptance, 2026-11-19

## Scope and outcome

Both scratch runs completed zero-based epoch1999 / 60000 updates, exit0:

- Fragment: `20261114-1706__VVI-IBP-INSNOTES-MEL-REPO-K12-S0`, job298789,
  completed2026-11-14 19:17:26+04.
- Mean: `20261114-1706__VVI-IBP-INSNOTES-MEL-REPO-MEAN-K12-S0`, job298790,
  completed2026-11-14 19:15:01+04.

Mean style substantially improves loss-selected/endpoint content mapping and
spectral reconstruction. Neither reaches the registered .95 source semantic
gate. No OOD FT, continuation or supervised baseline was launched. This is
source acquisition evidence, not an IBP persistence result. Cross-project
interpretation lives in [V5 RQ6](../../../v5/research/directions/RQ6_INDUCTIVE_BIAS_PERSISTENCE.md).

Config diff contains only name and `decoder_style_mode`; initial model SHA256
is identical (`f40dee859df33eb9dbfb7d3c74c67cdda464fe59c8b515ac91d5c67b3f19c944`).
MEAN pools the12 same-instrument style vectors for the decoder only. The four
V3 terms still consume raw individual style vectors. See [submission](20261114_INSNOTES_REPO_SUBMISSION.md).

All acceptance metrics use the SAME complete validation set:120 recordings,
14400 notes,100 per pitch x instrument. Test remains sealed. No test or
listening quality is used to select a checkpoint.

## Retained checkpoints

Best for listening means **lowest validation total loss**, not macro-best or
the historical maximum recognition. All three retained roles were reloaded
and their logged validation reconstruction/mapping values reproduced.

| Arm / role | Epoch | Hungarian 1:1 | Weighted purity | Active | Perplexity | Mel reconstruction MSE |
|---|---:|---:|---:|---:|---:|---:|
| Fragment / best validation | 1880 | .379792 | .384236 | 7 | 4.3853 | .676734 |
| Mean / best validation | 1920 | .812500 | .812500 | 12 | 11.9797 | .262874 |
| Fragment / final | 1999 | .440278 | .461389 | 12 | 10.1464 | 1.375043 |
| Mean / final | 1999 | .816944 | .816944 | 12 | 11.9853 | .365027 |
| Fragment / health-gated macro-best | 160 | .859792 | .859792 | 12 | 11.3515 | 3.158096 |
| Mean / health-gated macro-best | 270 | .666667 | .680556 | 12 | 10.1166 | 1.764573 |

Fragment reaches the higher historical1:1 peak (.859792@160); mean peaks at
.819444@1830. Thus mean does NOT improve the absolute historical maximum.
It improves the joint quality of reconstruction and mapping, and late stability:
51 evals from epoch1500 onward range .355625-.708056 for fragment versus
.766389-.819444 for mean. Epoch1830 weights were not retained.

Macro purity is an equal-code average, not a sample-weighted or Hungarian
accuracy. Mean macro-best270 (.853221 macro) is therefore NOT its best symbol
checkpoint. The gate for retaining macro-best is also weaker than the .95
source acceptance gate. Do not equate the two.

For historical reference, the preceding paper-bundle S1 final checkpoint had
1:1=.446389, weighted=.461806 and MSE1.761625. REPO fragment improves final
reconstruction but not final semantic accuracy; REPO-MEAN improves both.

## Where mean's remaining mapping errors occur

Additional GPU audit309091 uses a single GLOBAL validation Hungarian assignment
for the primary per-instrument accuracies. Independently rematching each
instrument is explicitly a diagnostic, not a replacement primary score.

| Instrument | Global-map accuracy | Within-instrument Hungarian diagnostic |
|---|---:|---:|
| Pipe Organ | 0 | 1.000000 |
| Viola | 0 | 1.000000 |
| Muted Trumpet | .916667 | .916667 |
| Choir Aahs | .833333 | .916667 |
| Other eight instruments | 1.000000 each | 1.000000 each |

Pipe Organ and Viola contribute2400/2700 errors (88.89%). Their pitches are
separable but use different code permutations from the other instruments.
For example, the fixed Pipe Organ C uses q4 (global A#), while Soprano Sax C
uses q2 (global C). A decoder receiving instrument style can interpret these
codes differently, so good same-input reconstruction does not establish
instrument-invariant pitch symbols. This is distinct from fragment's broad
code merging: per-instrument Hungarian there is only .3333-.4992.

## Timbre and reconstruction

Loss-selected mean beats fragment in MSE for all12 instruments.

| Best-validation metric, lower is better | Fragment | Mean |
|---|---:|---:|
| All-pitch log-mel MSE | .676734 | .262874 |
| High pitches G through B, log-mel MSE | .811606 | .262980 |
| Mean spectral-shape cosine distance | .064271 | .013317 |
| G through B spectral-shape cosine distance | .069220 | .012651 |

Overall MSE decreases61.16%; high-pitch MSE decreases67.60%. The supplementary
shape error is `1-cos(mean_time(exp(logmel_x)-1e-5), mean_time(exp(logmel_y)-1e-5))`
with nonnegative magnitudes. It is gain-invariant but can reflect pitch errors
as well as timbre; it is NOT instrument recognition or a perceptual score.
These measurements and the mel grids support improved spectral fidelity,
including high pitches, not a claim that timbre is perfect. Listening must
compare original, input-mel roundtrip and reconstruction because phase
recovery also changes perceived timbre.

Recording-disjoint PCA32/whitened ridge probes on the loss-selected models:

| Readout | Fragment | Mean |
|---|---:|---:|
| Raw z_s -> pitch (leakage) | .877222 | .570417 |
| Raw z_s -> instrument | .489444 | .425556 |
| Pre-VQ z_c -> pitch | .984583 | .827917 |
| Quantized code -> instrument | .119861 | .090278 |

These use5 fit and5 score recordings/instrument, chance1/12. "Raw z_s" means
before group averaging; feature preprocessing for this table is whitening.
Mean reduces but does not eliminate raw style pitch leakage, and does not
improve this particular linear instrument readout. Decoder-input pooling is
a separate architectural constraint. Marginal code->instrument chance-level
accuracy does not exclude instrument-dependent code permutations, as observed.

## BN diagnostic, not a repaired model

| Best-validation model / inference | MSE | Hungarian 1:1 |
|---|---:|---:|
| Fragment / normal eval | .676734 | .379792 |
| Fragment / decoder batch stats | .431231 | .379792 |
| Fragment / both batch stats | .385473 | .387014 |
| Mean / normal eval | .262874 | .812500 |
| Mean / decoder batch stats | .255646 | .812500 |
| Mean / both batch stats | .254762 | .812569 |

Mean has a much smaller running-statistics discrepancy at its selected model;
neither BN intervention fixes content mapping. Every buffer/mode is restored.

## Artifacts and checks

Read-only job309087 (ws-l2-007) completed exit0,50 related unit tests passed.
All six checkpoints reproduce full-val metrics. Best repeated evaluation is
identical; parameters, BN/VQ buffers remain unchanged and gradients absent.
Follow-up309091 also completed exit0, adding full per-style mapping counts
without changing either checkpoint or the accepted aggregate confusion.

Artifacts: [acceptance directory](../../logs/insnotes_repo_acceptance_20261119/).

- `acceptance.json`, `summary.csv`: all six roles, histories, probes and hashes.
- `per_style_mapping.json`: global versus per-style diagnostic and raw counts.
- `fragment/best_validation/`, `mean/best_validation/`: confusion/mel grids.
- [Listening comparison](../../logs/insnotes_repo_acceptance_20261119/index.html):
  Soprano Sax, Pipe Organ and Trumpet, original / mel roundtrip / fragment-best /
  mean-best. Each plays C4/E4/G4/B4 from the same fixed validation recording.
  All four tracks share one attenuation per instrument. Offline relative WAVs.
- Existing per-run `audio_examples/best_validation/` remains unchanged; its
  checkpoint hash was verified against the checkpoint loaded for acceptance.

## Decision

Keep MEAN as the more promising source recipe, but do not accept it for IBP
new-style comparisons yet. Next isolate the instrument-dependent permutation
and test content/style swaps with synthetic references. Do not silently remap
per instrument to declare success, or extend2000epochs simply because MSE
can still fall. A successful source must expose shared pitch identities and
retain useful timbre control, not only reconstruct each input.
