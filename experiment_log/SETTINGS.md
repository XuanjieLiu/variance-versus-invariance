# Experiment settings

## VVI-H0 — Historical Uppercase References

Historical reference checkpoints used to calibrate the current stage. Metrics in
the archive were recomputed on the full 2,600-sample test split with the same
acceptance evaluator used for VVI-S1.

| Run ID | Content dim | Style dim | Role |
|---|---:|---:|---|
| VVI-H0-C4S4 | 4 | 4 | failed low-dimensional reference |
| VVI-H0-C512S512 | 512 | 512 | high-dimensional reference |

## VVI-S1 — Uppercase Style-Bottleneck Isolation

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

The completed `c=4, s=512` run met the persistent-collapse gate, so it will not
be continued. The complementary `c=512, s=4` control is being run alongside the
VVI-S2 intervention to finish the dimension attribution without delaying the
primary fix.

## VVI-S2 — Projected Four-Dimensional Codebook

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

**Decision gate.**

- Rescue uses the VVI-S1 thresholds: active codes `=26`, perplexity `>=20`,
  one-to-one accuracy `>=0.60`, and test reconstruction `<=0.30`.
- If rescued, repeat the projected-codebook setting with additional seeds before
  changing EMA or loss weights.
- If usage still collapses, compare its learning trajectory with
  VVI-H0-C512S512 and next isolate the projection/VQ training strategy.
