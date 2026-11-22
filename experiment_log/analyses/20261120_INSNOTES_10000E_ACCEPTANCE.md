# InsNotes REPO 10000-epoch acceptance

Accepted2026-11-20 (Asia/Dubai). Jobs309229/309230 both reached absolute9999,
300000 cumulative updates, retained3checkpoints and exited0. Fragment finished
02:01:22+04; MEAN01:49:46+04. This is IBP-AUDIO-S2-R1, an8000epoch
continuation, not a fresh10000epoch run. Source paths and unchanged settings:
[submission](20261119_INSNOTES_REPO_CONTINUATION.md).

Read-only GPU audit310460 reloads all6retained roles, reproduces logged
reconstruction/mapping, verifies unchanged parameters/BN/VQ, repeats best
inference deterministically and exports a matched listening comparison.
All numbers below use balanced source VAL:120recordings,14400notes,
100per pitch x instrument. Test and OOD remain sealed.

## Results

| Arm / role | Epoch | Hungarian1:1 | Weighted purity | Active | Perplexity | Log-mel MSE |
|---|---:|---:|---:|---:|---:|---:|
| Fragment best-val | 8600 | .332083 | .362708 | 12 | 11.1657 | .230115 |
| MEAN best-val | 9000 | .791667 | .791667 | 12 | 11.9930 | .185824 |
| Fragment final | 9999 | .387778 | .408472 | 12 | 11.1608 | .361484 |
| MEAN final | 9999 | .784722 | .784722 | 12 | 11.9930 | .191358 |
| Fragment historical macro-best | 160 | .859792 | .859792 | 12 | 11.3515 | 3.158096 |
| MEAN historical macro-best | 270 | .666667 | .680556 | 12 | 10.1166 | 1.764573 |

Against the2000epoch acceptance, best-val MSE improves .676734->.230115 for
fragment and .262874->.185824 for MEAN. But their selected1:1 degrades
.379792->.332083 and .812500->.791667. Final1:1 also falls
.440278->.387778 and .816944->.784722. Neither historical1:1 peak is
improved: fragment.859792@160,MEAN.819444@1830. Lower loss and more training
do not resolve acquisition of shared pitch symbols.

## Remaining failure mode

MEAN best9000 still has Pipe Organ and Viola global-map accuracy0, despite
within-instrument Hungarian1.0. Muted Trumpet now has global.75 but
within-instrument1.0; Choir global.75 and local.916667. The other8styles are
globally perfect. Thus2400of3000errors (80%) remain the two complete code
permutations, not missing active atoms. Independently rematching instruments
must not replace the registered global metric.

MEAN's source code->instrument probe is near chance (.09028 versus.08333),
yet conditional permutations remain. Raw style still carries pitch information:
whitened pitch probe.61014. Both final V3 four-term hinge losses are zero;
satisfying the ratios is insufficient to ensure shared symbols or factor
independence. This is not evidence for/against persistence after healthy acquisition.

MEAN best spectral-shape cosine error.005222 versus fragment.012390;
high-pitch G-B MSE.167739 versus.224832. These are spectral reconstruction
proxies, not measured timbre classification or subjective listening judgments.

## Decision and artifacts

Both fail the >=.95 source semantic gate. No more automatic continuation and
no audio OOD FT. Preserve MEAN as the better source recipe, not a qualified IBI
teacher. Next audio work should address cross-instrument identity/controlled
swaps, not infer success from low reconstruction alone.

For the separate user-authorized RD-V3 experiment, use the existing healthy
image HEX RANK2 epoch59 teacher, not either audio source. Canonical IBI
interpretation lives in [V5 RQ6](../../../v5/research/directions/RQ6_INDUCTIVE_BIAS_PERSISTENCE.md).

- Machine-readable audit: `logs/insnotes_repo_10k_acceptance_20261120/acceptance.json`.
- Summary: `logs/insnotes_repo_10k_acceptance_20261120/summary.csv`.
- Offline listening page: `logs/insnotes_repo_10k_acceptance_20261120/index.html`.
- Both original runs retain fixed original/roundtrip/best audio under
  `audio_examples/best_validation/`. Matching clips use one common gain,
  fixed64-step Griffin-Lim and C4/E4/G4/B4 notes; no learned vocoder.

SLURM accounting was unavailable at acceptance (database refused connection).
Completion is verified by each wrapper's exit0 record, final9999 artifacts,
empty queue and successful fresh checkpoint audit, not an invented sacct status.
