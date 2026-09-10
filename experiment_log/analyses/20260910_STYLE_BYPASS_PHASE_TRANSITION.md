# 2026-09-10 — Content phase transition and a change in reconstruction routing

## Conclusion and confidence

The best supported mechanism is a switch from reconstructing through the
continuous style representation to reconstructing letter identity through the
quantized content code. The user's proposed connection between reconstruction
degradation and better content recognition is plausible and now has direct
frozen-checkpoint intervention evidence. The temporal cause of the switch remains
unidentified: loss of the style shortcut might precede codebook improvement, or
both may follow a common change in the encoder/decoder/VQ dynamics.

High confidence: the two retained historical states use different reconstruction
paths. Moderate confidence: the same mechanism explains S3 and C128 transitions.
Not established: deliberately destabilizing the decoder causes an earlier useful
transition; weight decay is the trigger; a thermodynamic phase transition or
hysteresis has been demonstrated.

This note reports diagnostics, not newly trained formal experiments. Checkpoint
parameters and datasets were unchanged. Python, PDF extraction, numerical
analysis, and evaluation ran through `srun --partition=ws-ia` on a GPU compute
node. Evaluation code and raw results are under
[`logs/diagnostics/20260910_phase_mechanism/`](../../logs/diagnostics/20260910_phase_mechanism/).

## Observed timing: training loss must be distinguished from validation spikes

The following are existing full-validation code metrics and epoch-average
training reconstruction losses, not metrics from a newly sampled test subset.

| Run | Epoch | Train recon | Validation Hungarian | Train V3 hinge sum |
|---|---:|---:|---:|---:|
| S3 C512/K26 | 139 | .12127 | .24700 | .000827 |
| S3 C512/K26 | 140 | .14641 | .58256 | .004181 |
| S3 C512/K26 | 141 | .15551 | .70417 | .000813 |
| C128/K26 | 158 | .12792 | .28006 | .011701 |
| C128/K26 | 159 | .16526 | .66948 | .004148 |
| C128/K26 | 160 | .15611 | .70392 | .000694 |

Both show a persistent rise in training reconstruction cost accompanying better
assignments. The V3 terms are small overall, but are not identically zero at every
step; brief increases occur around the transition. Code metrics were collected
at the end of each epoch, so these histories cannot establish within-epoch
causal ordering. The original step losses are summarized in 50-step bins in
[`trajectory_summary.json`](../../logs/diagnostics/20260910_phase_mechanism/trajectory_summary.json).

Separately, validation reconstruction had extreme spikes before the transition:
S3 epoch137 was 60.07 while train recon was .1247; C128 epoch157 was 33.78 while
train was .1230. These evaluation-specific instabilities should not be called
the same event as the later persistent train-reconstruction increase. A small
train/eval gap at a retained post-transition checkpoint does not rule out earlier
normalization instability as a possible trigger.

See the [aligned trajectory plot](../../logs/diagnostics/20260910_phase_mechanism/transition_windows.png).
Training and validation reconstruction use separately labeled linear/log axes.

## Direct channel interventions on retained checkpoints

`uppercase_letters_run2_batch32` retains low-purity epoch80 and high-purity
epoch140 checkpoints. Their exact transition point is not localized by these
two snapshots; they are not S3's missing epoch139/140 checkpoints.

Assignment metrics below use the identical fixed 512-sample test subset, 64
pages per style, seed0. Reconstruction interventions use the first 128 pages of
that subset, balanced across styles. No optimizer or codebook updates occur.
Each intervention cyclically permutes fragments **inside each page**, preserving
the page's color and overall distributions while changing letter correspondence.

| Frozen model / intervention | Historical epoch80 | Historical epoch140 |
|---|---:|---:|
| Test Hungarian, 512 pages | .12508 | .72964 |
| Original reconstruction MSE, 128 pages | .12795 | .15484 |
| MSE after shuffling content codes within page | .12795 | .21941 |
| MSE after shuffling style embeddings within page | .23613 | .15489 |
| MSE after replacing style with its page mean | .20303 | .15486 |
| Output change from content-code shuffle | 2.66e-15 | .06626 |
| Output change from style-embedding shuffle | .11246 | 7.57e-5 |

Epoch80 reconstructs letters while being effectively insensitive to which VQ
code is supplied. Its continuous style pathway carries fragment-specific
information needed for letter reconstruction. At epoch140, reconstruction is
sensitive to the content code and almost insensitive to within-page style
residuals. This is strong evidence for a change in functional routing, not proof
of which training event initiated it.

The retained S3 epoch145 and C128 epoch165 states reproduce the latter pattern:
content shuffle changes the output by .06270/.07286 MSE, whereas within-page
style shuffle changes it by only 1.06e-4/2.50e-5.

The [before/after image](../../logs/diagnostics/20260910_phase_mechanism/historical_before_after.png)
shows the same A/B/E examples in eight styles. The earlier model produces varied
colors, although they are often inaccurate; the later model predominantly
produces brown glyphs. Raw interventions are in
[`channel_ablation.json`](../../logs/diagnostics/20260910_phase_mechanism/channel_ablation.json);
the six-checkpoint comparison is in
[`checkpoint_comparison.json`](../../logs/diagnostics/20260910_phase_mechanism/checkpoint_comparison.json).

## Why a low V3 loss does not exclude this shortcut

The following is an illustrative mechanism derived from the implemented loss,
not a theorem that the measured model has exactly this form. Suppose

\[
z^s_{ij}=a_i u+\varepsilon h(x_{ij}),\qquad u^T h(x_{ij})=0.
\]

`a_i` is a page-shared statistic such as corruption strength. `h` can encode
letter identity and color in low-amplitude directions. The within-page style
MPD can be O(epsilon), while between-page MPD remains substantial. A decoder
whose gain in the `h` directions is O(1/epsilon) can still recover that
information. The V3 ratios can therefore look favorable without preventing
content leakage into style. Low geometric variance is not the same constraint
as low information or low decoder sensitivity.

The measured style spaces are consistent with this risk: over 512 test pages,
the first principal component of the page-mean style embeddings explains more
than 99.99999% of their variance at historical epoch80. Its absolute correlation
with image horizontal-difference energy is .967. The latter is a proxy for
high-frequency corruption, also affected by glyph/color/blur; it does not alone
identify the Gaussian generator as the causal source.

An important correction to the previous diagnosis: a weak ordinary linear
probe is insufficient to conclude that style information is absent. We fitted
PCA/whitening and ridge probes on a fixed 512-page **train** subset and evaluated
on the separate 512-page test subset. With 64 whitened components, color
accuracy is .752 at historical epoch80, .219 at epoch140, and .211 at C128
epoch165. The unwhitened values are .154/.125/.127. Tiny directions can carry
decodable information. Post-transition color information is substantially
weaker in these probes, but is not proved to be zero. Component counts
4/8/16/32/64 are all saved; these exploratory probes do not select training
checkpoints or enter formal acceptance gates.

The four V3 hinge losses become locally flat when their ratios exceed the
threshold. Both reconstruction-routing modes can occupy low-hinge-loss regions.
Reconstruction, commitment, minibatch fluctuations, EMA updates, and AdamW can
continue changing the state. In particular, an epoch-average loss near zero
does not imply every minibatch had zero gradient. Validation commitment values
returned as zero by the eval-mode VQ layer are also not evidence of zero true
quantization error; the diagnostics compute raw encoder-to-atom MSE explicitly.

## A mechanism for simultaneous recognition improvement and reconstruction loss

Before the switch, decoder sensitivity to the content code is nearly zero in
the intervention. Consequently its reconstruction gradient through the VQ
straight-through route can be weak, leaving code learning mostly to the other
objectives and EMA dynamics. If the continuous shortcut loses effectiveness,
using the code to identify the letter becomes more useful for reducing recon
error. Better shape codes and greater decoder reliance on those codes can then
reinforce each other. Incomplete recovery of color through the style branch
leaves a higher reconstruction floor even as content recognition improves.

For squared error, the optimal decoder for fixed representations is their
conditional mean, `E[X | Q, Z_s]`. If available or effectively used style
information fails to distinguish colors, that mean averages colors. This
explains a brown output tendency; the actual trained network is not assumed to
equal the exact conditional expectation or to be at a global optimum.

This is an **inference** connecting the measured endpoints. Alternatives still
include an encoder change occurring first, a decoder gain change occurring
first, or VQ/normalization instability affecting both. The historical and S3
records do not prove that decoder failure is necessary or sufficient. Nor do
they show that the new state has a lower total objective: the measured train
loss rises. SGD/AdamW plus EMA is not a monotone descent process on the logged
scalar, and decoupled weight decay is not included in that scalar.

## Relevant theory and its limits

- [V3, ICLR 2025](https://arxiv.org/html/2407.03824v3) supplies the actual
  statistical constraints. Appendix C.2/Figure17 demonstrates simultaneous digit
  and color reconstruction on PhoneNums, so sacrificing color is not a
  necessary property of V3. UppercaseLetters and this trajectory differ from
  that experiment.
- [Variational Lossy Autoencoder, ICLR 2017](https://arxiv.org/abs/1611.02731)
  studies how alternative decoding routes affect what latent variables learn.
  This motivates the information-preference/bypass interpretation. Its
  variational/autoregressive setting differs from this deterministic style
  branch, so this is a mechanism analogy, not literal KL posterior collapse.
- [CycleGAN, a Master of Steganography](https://arxiv.org/abs/1712.02950)
  demonstrates reconstruction-relevant information in low-amplitude signals.
  It motivates checking decoder sensitivity rather than inferring independence
  from small latent distances. It does not establish that this V3 model uses
  the same mechanism as CycleGAN.
- [Understanding disentangling in beta-VAE](https://arxiv.org/abs/1804.03599)
  explains reconstruction/disentanglement tradeoffs through information
  capacity. Here the finite code supports at most log2(K) bits per atom index;
  the actual continuous style channel has no comparable explicit rate bound.
  Changing embedding dimension at fixed K is not itself changing log2(K).
- [Straightening Out the Straight-Through Estimator, ICML 2023](https://proceedings.mlr.press/v202/huh23a.html)
  identifies encoder/codebook distribution mismatch and STE optimization
  instability. Discrete assignment boundaries and lagging EMA adaptation are
  plausible sources of sharp changes, but the paper does not predict improved
  label alignment after such a change.
- [Explaining grokking through circuit efficiency](https://arxiv.org/abs/2309.02390)
  motivates competition between solutions and delayed changes in which one
  dominates. Our event is a latent-routing/semantic change with rising train
  reconstruction, not sufficient evidence of classical grokking. Weight decay
  and a rigorously defined bifurcation remain hypotheses.
- [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782)
  motivates measuring competing gradient directions. Conflict is not proved
  by one loss rising as another metric improves, especially when the V3 hinges
  are inactive; it should be measured before proposing gradient surgery.

## Next causal test, proposed but not submitted

Prioritize a matched branch pair from the available historical `cp_epoch80.pt`:

1. Control: original decoder input `D(Q, Z_s)` and unchanged loss weights.
2. Intervention: decoder input `D(Q, mean_fragments(Z_s))`, keeping the raw
   embeddings for the existing V3 loss. This removes fragment-specific style
   leakage while retaining page-shared information. Keep recon weight at 1.

Use the same initialization and RNG/data order; inspect optimizer/scheduler
fields first. If the old checkpoint lacks them, initialize the same fresh
optimizer in both arms and describe them as matched warm starts rather than an
exact continuation. A 40–60 epoch pilot can test whether code dependence and
purity develop earlier. Style sharing is justified for these same-color pages,
not for V5 triplets whose three images can have different styles.

Monitor a fixed diagnostic batch every 50 steps around onset: code Hungarian/
macro/usage, raw recon and quantization error, content-shuffle and
within-page-style-shuffle output changes, style residual scale, decoder input
gradient norms, foreground color error, and train/eval gap. Dense *metrics* need
not add checkpoints beyond current and best. Use validation labels only for
diagnostics, not intervention triggers.

Evidence supporting the mechanism would be: removing the shortcut increases
code dependence first, then raises purity earlier than the matched control,
while page color can still be reconstructed. Improved purity without color
recovery supports shortcut removal as a content catalyst but is not successful
content-style disentanglement. No purity improvement would show that shortcut
removal alone is insufficient. A second phase can fit a decoder against frozen
post-transition embeddings to separate decoder recoverability from missing
encoder information.

The earlier generic recon-weight pulse remains a secondary test. Reducing all
reconstruction gradients does not specifically remove the style shortcut and
may also remove the signal needed to teach the content code. S3 epoch130 is no
longer present in its run directory; the prior plan must not assume it can be
restored. Persistence at fixed settings should be called persistence, not
hysteresis, until a reversible intervention tests dependence on history.
