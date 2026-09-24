# V2 acceptance and corrected-color PhoneNums replication — 2026-09-12

## Completed UppercaseLettersV2 pair

Both jobs completed epoch199 with exit0 on2026-09-11. Neither has a qualifying
stable semantic window or passes the original content acceptance gate. Do not
continue either trajectory. Full2,600-page test reports and new-format figures
are saved under each run's `acceptance_20260912/`; historical periodic plots and
all10 checkpoints per run are untouched.

| Run / checkpoint | Macro | Weighted | One-to-one | Active | Perplexity | Coverage | Recon | RGB MAE | Chroma RMSE | Fixed eval/train recon |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CTRL best7 | .67536 | .56266 | .56185 | 24 | 19.77 | 21 | .05631 | .20518 | .16342 | 1.477 |
| CTRL current199 | .58985 | .21630 | .19865 | 26 | 10.41 | 11 | .65290 | .40754 | .39677 | 35.260 |
| W100 best3 | .74091 | .63222 | .61781 | 26 | 21.79 | 24 | .06018 | .21051 | .15819 | 1.093 |
| W100 current199 | .44490 | .42787 | .39851 | 26 | 24.94 | 18 | .02439 | .11220 | .07954 | 6.047 |

CTRL is a learning failure, not a process crash. Current macro hides heavily
unequal usage;13.07% of predicted channel values fall outside[0,1]. Its fixed-batch
mode gap strongly suggests normalization-statistics mismatch, but is not a
selective BN intervention proving the full cause.

W100 improves paired color reconstruction in the late phase, but its VQ mapping
remains poor. At199, detached whitened zS→content accuracy=.9395 (chance1/26),
while code→style=.1594 (chance1/8). This demonstrates decodable content in the
style representation; decoder dependence remains a causal hypothesis, not proven
solely by the probe. The first high macro at3 was transient. At99→100, validation
one-to-one drops .4807→.2464; the run was already unstable before release.

Correct colors are not proven intrinsically harmful. V2 changes images/seeds,
strict split balance and two renderer details; no single-fix causal attribution
is justified. Healthy reconstruction, V3 hinge saturation and active-code count
are individually insufficient to demonstrate semantic disentanglement.

## What foreground masks mean

The green overlay marks an approximate foreground selected from the actual input,
not a learned segmentation or reconstruction. Outer2px channel medians estimate
background; interior max-channel delta>.15 forms the mask; fewer than16 pixels
means invalid. The same fixed input has the same mask at every epoch (W10049/100
PNG SHA256 both `3567ed896b7f529c7913ab5eabda227ba544782b56c337679283cc862db4a6ad`).
Masks help audit whether foreground RGB/chroma errors select glyphs rather than
mostly background. They still contain noise and estimation errors; no clean
references or ground-truth masks are used. Blank confusion annotations mean
rounded0.00, possibly a small nonzero exact probability; CSV/JSON preserve it.

## Why two PhoneNums configurations

Pinned upstream: `Irislucent/variance-versus-invariance` commit
`bbbc7c9c26bda7170612814a407b76633cd22875`. Appendix B.2 and the published config
differ: PAPER uses Adam/weight decay0, V3 weights.1, commit.01, EMA.95 and dead
threshold320/(10×10)=3.2; REPO uses AdamW/.1, V3 weights1, commit.1, EMA.98 and16.
Both use C512/S512/K10, original BN, lr.001, batch32 and r15. Unspecified paper
betas/eps/scheduler details follow the repo; Adam decay0 is an explicit default.

Both use independent corrected-color PhoneNumsV2:100,000 pages,80k/10k/10k splits,
eight balanced styles and all10 digits once/page. Background245 follows source
and the paper's light-background examples; the text's light RGB(10,10,10) is
inconsistent. Keep NumberOffice glyph placement rather than uppercase fit-to-bbox.
Unique permutation sampling prevents filename collisions at this scale.

Published manifest SHA256:
`1e54fa7ce5c28255c0e4c7a22f428a8aae3005147fac8faa2dd1ab7d9f2c9fce`.
Generation and serial/worker-independence tests ran on ws-l5-006 (Job187708);
the subsequent preflight verified all100,000 published PNG hashes independently.
The full60-test regression suite and the four saved acceptance evaluations ran
on ws-l4-009 (Job187714), not on a login node.

This is a matched200-epoch, seed0 implementation of two parameter bundles, not
a bitwise reproduction or the repo's10,000-epoch budget. Main acceptance is
best-validation-loss, additionally reporting macro-best/current. Compare Table8
K10 macro89.2% separately from weighted/Hungarian and inspect Figure17-style
code×class-mean-style recombination. No Table1 quadratic retrieval/OOD baseline
expansion or single-seed statistical reproduction claim.

## Next experiments and decision

Preflight passed on ws-l4-009: three two-training-batch/one-epoch smoke runs used
complete validation (100,000 fragments per digit arm;67,600 for W50). They verified
diagnostic RNG/model/BN/VQ neutrality, W50's49→50 switch, best-val save/alias,
balanced10/26-class matrices, and Figure17 recombination. Smoke directories were
deleted after visual inspection (about1.65GiB), leaving only the lightweight
preflight summary at `logs/diagnostics/20260912_pn_w50_preflight/summary.json`.
PAPER/REPO initial model SHA256:
`70c737ffe5a7dd33a6bead0a5655aa7d2274e92cc8a7243f5addda0567f88c3c`;
first-epoch sampler SHA256:
`46cb7c17b68270824b343407a449dba9bc81143dab11981fa1185708b03c24d6`.
W50 initial hash exactly matches archived V2-W100:
`ad9ca6d0eabfdf21dc76df594f241fa4c484685adbd50d679ee8b4075c8dbdd6`.
The60-test suite covers200-epoch retention ceilings11/10, best-val ties/restore/
legacy compatibility, exact optimizer/loss/EMA branches, and generator invariants.

- VVI-PN-K10-PAPER-S0 / VVI-PN-K10-REPO-S0: each200 epochs,1GPU/32GB/4CPU/24h.
- VVI-RQ2-V2-C128-K26-W50-S0: matched W100 except release at50;200 epochs,8h.
- W50 preserves the V2 objective/BN and does not presume earlier release works.
- Compare full-test purity/usage, color, leakage and mode gap. If PhoneNums fails
  under both parameter bundles, prioritize targeted normalization/style-path
  diagnostics; don't infer that26 contents or C128 alone caused the V2 failure.
- If only one bundle works, its combined protocol is supported; further ablation
  is needed before attributing improvement to loss weights, optimizer or EMA.

## Submission provenance

All three were entered in ACTIVE before sbatch submission. Prefix
`20260912-1420__`; code HEAD `e0bb0c6b70b5798c0dddc69c6903f9cdf315568a`, dirty.
Each started job saves source archive, dirty patch, source hashes and submitted
config under `reproducibility/`; no commit or push. PAPER has not started yet,
so its runtime provenance must be checked after allocation.

| Run suffix | Job | Submitted (Asia/Dubai) | Allocation / budget | Config SHA256 |
|---|---|---|---|---|
| VVI-PN-K10-PAPER-S0 |187723|2026-09-12T14:20:33+04:00|pending QOSMaxJobsPerUserLimit;24h|`b7b45ac6c93ea01f1040f510adb2232ae048793b49f28438e4be4fb8034e6f6a`|
| VVI-PN-K10-REPO-S0 |187722|2026-09-12T14:20:31+04:00|ws-l4-014, RTX5000 Ada;24h|`f0c467d2b28f7007b7013126c8d652739944ebcf3860b4ee723c500cecd7e56f`|
| VVI-RQ2-V2-C128-K26-W50-S0 |187721|2026-09-12T14:20:28+04:00|ws-l5-006, RTX5000 Ada;8h|`d124df30967942848cbf179908a21d156a586e07d3215cb897f1b81399a10efd`|

Each requests1GPU/32GB/4CPU on ws-ia. The current account permits two running
jobs; PAPER will start automatically after a slot is released. It is queued,
not failed, and must not be claimed as runtime-verified yet.

Post-submit read-only verification ran inside W50's compute allocation. Both
running arms' backed-up config equals the submitted config apart from the run
name; manifest and initial-model hashes match preflight, and source archives
exist. ACTIVE has exactly these three Run IDs with no overlap with ARCHIVE;
all smoke directories are absent and all four V2 acceptance artifact sets exist.
