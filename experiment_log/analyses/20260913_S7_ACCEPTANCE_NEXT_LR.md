# S7 C512 W0/W50 acceptance and next hyperparameters

Subsequent user decision: prioritize the independent normalization alternative,
not the two-lr sweep. S8 uses matched W50/lr1e-4 decoder BN versus GN8; see
[S8 protocol and submission record](20260913_S8_DECODER_NORMALIZATION.md).
The recommendations below preserve the acceptance-time reasoning.

Accepted2026-09-13, after both jobs finished normally at epoch199. W0 Job189975
ended18:09:45+04; W50 Job189976 ended18:08:21+04; both exit0, neither timed out.
All calculations ran on ws-l4-003. Full-test acceptance used Job191516;
the subsequent retained-snapshot EMA audit also ran through a short GPU srun.
No training, loss, normalization or checkpoint-selection code was changed.

## Full-test results

Each report uses all2600 test pages /67,600 fragments. Saved under each run's
`acceptance_20260913_s7/`: health JSON with config/checkpoint/data hashes, paired
reconstruction, input foreground mask, balanced column-normalized confusion
PNG/SVG/CSV/JSON, mapping, trajectory summary and acceptance index.

| Run / selection | Epoch | Macro | Weighted | Hungarian | Active | Perplexity | Coverage | Recon | Foreground RGB MAE / chroma RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| W0 best-val |28|.44485|.14567|.14562|7|3.5128|5|.052910|.20560 / .16872|
| W0 current |199|.39856|.12641|.11337|21|4.2523|12|.139333|.31494 / .17374|
| W50 macro-best |102|.44149|.38003|.35725|26|23.6465|20|.109048|.25896 / .15977|
| W50 best-val |16|.44196|.27596|.27256|21|12.1305|19|.073514|.22953 / .15786|
| W50 current |199|.37628|.34065|.31768|26|24.2137|20|.148539|.30307 / .17706|

W0 has **no macro-best checkpoint**: zero validation epochs satisfy all four
registered health floors. Do not substitute its unconstrained maximum as a
successful checkpoint: epoch191 macro=.79926 accompanies weighted=.13219,
perplexity1.7602 and coverage5. Retention worked as designed: W0 has10 files,
W50 has11 (current/best-val/8 snapshots, plus an eligible macro-best only for W50).
W50 has51 health-eligible epochs, but its best eligible macro is only.44333 on
validation. Neither run has a single registered8-of-10 stable window. Neither
passes the normal-eval content gate; do not continue current just for more epochs.

Best-val is lowest **total validation loss**, not necessarily minimum recon.
The full-test health evaluator's commit field is zero because the VQ library
returns no training commitment penalty in eval; this is not zero quantization
error. `ema_audit.json` separately records raw encoder-to-code MSE.

## What failed

### 1. These are not two equivalent semantic failures

W0 already has only6 active codes/perplexity3.43 after epoch0 and stays unhealthy.
Its last50 mean validation Hungarian is.13625. At current, the style ratio
sample/fragment is only.8941 (target15), yielding style loss.9404; training V3
is still.9625. Whitened validation style→content accuracy is.91391 while
style→style is.15625 (8-style chance.125). This is consistent with the continuous
branch retaining shape while poorly separating page style, not successful V3.

W50 is also unhealthy very early, before release50. It later recovers broad
usage, but not clean content: best102 has all26 codes/perplexity23.65 and macro
only.4415. At199 the four full-test mean V3 ratios are78.89/73.31/95.37/59.75,
all above15, yet macro=.3763. Average ratios cannot establish that every batch
hinge is inactive; full-test V3=.02838 and training V3≈.00010. Last50 validation
hinge-active batch fractions are content.0193/style.0751/sample.0878/fragment.0349.
Multiplying weights on already inactive hinges produces no gradient there.
Raising relativity is a different intervention and can reactivate some hinges,
but does not by itself guarantee class-aligned codes.

W50 current detached validation probes: code→style=.33053 (chance.125),
whitened style→content=.86388 (chance1/26), style→style=.51397. Both channels
still carry the wrong factor. These are decodability tests, not a complete causal
proof that the decoder uses that information. The current recon grid visibly
retains many blue shapes but fails most other styles, consistent with the
per-style color metrics, not a successful colored reconstruction result.

### 2. Decoder BN is a measured, separate reconstruction failure

All entries below are the existing fixed32 validation pages,4/style, with VQ
frozen and the same full-validation Hungarian mapping. They are diagnostic, not
test acceptance. Decoder-only interventions leave assignments exactly unchanged
at all200 epochs of each run.

| Run / epoch | Normal recon | Encoder-only BN recon | Decoder-only BN recon | Normal / decoder-only fixed-map accuracy |
|---|---:|---:|---:|---|
| W0 /89 |277.1315|281.7030|.0179|.0793 / .0793|
| W0 /199 |.1400|.1270|.0296|.1274 / .1274|
| W50 /49 |46.7415|10.6337|.0752|.0793 / .0793|
| W50 /199 |.1501|.1492|.0146|.3305 / .3305|

The last50 median W50 recon is.15192 normally versus.00971 with decoder-only
batch statistics. Switching encoder BN changes47.54% of indices at the median
late epoch, but does not restore high semantics: median fixed-map accuracy goes
.26142→.22055. This latter comparison intentionally keeps the mapping fixed;
it is not the newly optimized Hungarian accuracy of the intervention model.
BN mode mismatch is therefore real, but simply forcing train-mode evaluation
does not solve semantic learning and is not an acceptable reporting fix.
See [PyTorch BatchNorm semantics](https://docs.pytorch.org/docs/2.12/generated/torch.nn.modules.batchnorm.BatchNorm2d.html)
for batch statistics versus moving estimates; no training architecture change
has been applied here.

### 3. Slowdown is a more defensible first hyperparameter test than larger losses

The actual exponential scheduler is almost flat: lr starts.001 and remains
about.000818 at epoch199 (`.001 × .98^(199/20)`). The cosine-related config fields
are unused under `scheduler: exponential_decay`; do not mistake them for late
annealing. Consequently the model never gets a genuinely small learning rate
to settle its encoder/codebook/decoder geometry.

Online train usage is not evidence of stable code assignments. W0 epoch0
online perplexity25.68 coexists with frozen-validation3.43; at199 it is22.24
versus4.23. W50 likewise starts25.43 online versus1.63 frozen. Changing model
and code identities can distribute assignments across the epoch while each
fixed checkpoint is poor. This does not prove replacements are the sole cause.

Read-only snapshot audit (`scripts/diagnose_s7_ema.py`) finds17/26 W0 epoch199
EMA masses exactly at the reset value16, consistent with extensive recent
replacement; W50 epoch199 has none, despite its low purity. On one fixed balanced
batch, the next EMA update would expire16–18 W0 codes at threshold16, but0 W50
codes. These are hypothetical one-step counts using the original EMA state, not
a replay, measured training reset frequency, or evidence that a lower threshold
will succeed. Threshold16 is half the uniform32 assignments/code/batch; it is
potentially aggressive, not intrinsically incorrect. Do not increase it to
force a prettier usage curve. Instrument actual replacement counts before a
dedicated threshold experiment.

The broader VQ literature identifies encoder/codebook distribution mismatch and
STE optimization as instability mechanisms; it does not establish the cause in
this run or guarantee a particular learning rate.
[Huh et al., ICML2023](https://proceedings.mlr.press/v202/huh23a.html).

## Proposed next pair — not submitted

For the user's requested **hyperparameter-only** round, use W50 as the common
base because it recovers broad usage, then isolate learning rate. Do not repeat
the W0/W50 split while changing several objective/EMA values simultaneously.

| Candidate | Base | AdamW learning rate | Everything else |
|---|---|---:|---|
| W50-LR3E4 | S7 C512/S512/K26 W50 |.0003|unchanged|
| W50-LR1E4 | S7 C512/S512/K26 W50 |.0001|unchanged|

Both scratch, seed0,200 epochs, same V2 manifest/init/order/batch32, BN, r15,
recon/four V3 weights1, commitment.1, wd.1, EMA.98, threshold16, scheduler factor
.98 per20 epochs, W50 release, diagnostics and11-checkpoint retention. The
learning-rate × decoupled-weight-decay interaction remains part of changing
AdamW lr; this is not a pure parameter-displacement causal theorem.

Hypothesis: slower encoder/decoder drift relative to EMA and BN tracking allows
stable semantic organization. This is a test, not an observed improvement.
At20/50 inspect frozen usage, train/val gaps and code changes; at100/200 inspect
semantic slope and the original stable window. Final gate unchanged: normal
full-test macro>=.75, active26, perplexity>=20, coverage>=24, recon<=.30; report
weighted/Hungarian/color/probes separately. A better-looking train-mode recon
does not qualify. Do not declare low lr failed merely for a later transition;
consider extension only when healthy semantics are genuinely still rising.

**Normalization remains a high-priority separate control.** S7's pre-registered
decision gate points to it, and current evidence strengthens that recommendation.
If the user is willing to spend one slot on an architecture intervention rather
than two lr values, prefer a matched low-lr BN versus decoder-GroupNorm pair;
that needs a separate implementation/registration, and may not fix encoder
semantics. The pure-lr pair is offered to respect the present hyperparameter
request, not because BN has been ruled out. No config, formal run or BN fix has
been created/submitted in this acceptance turn.
