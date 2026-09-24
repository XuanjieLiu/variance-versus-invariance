# PhoneNums / C128-W50 acceptance and C512 controls — 2026-09-13

Evening status update: S7 C512 W0/W50 both subsequently completed199 with exit0,
failed semantic acceptance and were moved to ARCHIVE. The submission section
below records startup status, not current activity. See
[S7 full-test acceptance and next lr proposals](20260913_S7_ACCEPTANCE_NEXT_LR.md).

## Full-test acceptance

Saved eight health reports, paired reconstruction/mapping/foreground-mask figures,
and balanced column-normalized confusion SVG/PNG/CSV/JSON under each source run's
`acceptance_20260913/`. Digit checkpoints also have10×8 code/style-mean recombination.
Health JSON records checkpoint/config/data hashes. Historical periodic figures and
checkpoints are unchanged. Sources share prefix `logs/20260912-1420__`.

| Run / checkpoint | Test pages | Macro | Weighted | One-to-one | Perplexity | Recon |
|---|---:|---:|---:|---:|---:|---:|
| PN REPO macro-best0 |10000|.86000|.80000|.80000|9.68|.096405|
| PN REPO best-val177 |10000|.69344|.72185|.69732|9.54|.015796|
| PN REPO current199 |10000|.75915|.73343|.73343|9.70|.041476|
| PN PAPER macro-best14 |10000|.56779|.45720|.39168|8.42|.079137|
| PN PAPER best-val152 |10000|.30975|.24026|.23426|5.43|.003354|
| PN PAPER current199 |10000|.38219|.21749|.21749|4.30|.014204|
| Uppercase C128-W50 macro-best10 |2600|.76492|.66469|.66428|22.65|.048007|
| Uppercase C128-W50 current197 |2600|.43620|.07709|.07303|2.77|235.54847|

REPO is a useful partial reproduction: current uses10 codes with coverage10 and
recognizable colored reconstructions. It does not stably reproduce Table8's.892;
the label-oracle macro-best occurs at0 and has worse color/reconstruction than
the later best-val. Best-val foreground RGB/chroma=.05641/.04860; current=
.10600/.09168. Whitened style→content remains1.0, so these scores do not establish
independent representations or prove the decoder's use of each channel.
The best-val code×style-mean grid also does not preserve each row's digit shape
across all styles: several columns are dominated by a similar shape. This is a
further reason not to claim full disentanglement; style means can be off-manifold,
so this visualization alone is not a complete causal channel-use diagnosis.

PAPER best-val reconstructs extremely well (foreground RGB/chroma=.02419/.02234)
while VQ semantics remain poor. Neither recon nor V3 hinge saturation suffices
for semantic disentanglement. The bundle comparison does not isolate optimizer,
loss weights or EMA as a single cause. Do not extend either run in this round.

REPO Job187722 completed199 at2026-09-12T23:04:53+04, exit0. PAPER Job187723
started on ws-l4-011 at2026-09-12T22:20:56+04 and completed199 at
2026-09-13T07:05:57+04, exit0; it is no longer queued. Its initial model hash
matches REPO (`70c737ffe5a7dd33a6bead0a5655aa7d2274e92cc8a7243f5addda0567f88c3c`).
C128-W50 Job187721 reached the8h limit at2026-09-12T22:20:34+04, with late
buffered training output through22:21:03. Last completed epoch197; epoch198 was
partial and199 never ran. Archive as timeout, not completed200 epochs.

## C128-W50 collapse: two distinct failures

Epoch10→11 (the12th completed training epoch), while decoder is still page_mean:
validation one-to-one .66151→.09075, active25→11, perplexity22.64→2.79. Train
commit .01058→.10378, V3 0→.19106, recon .04803→.05999. Large loss excursions
around global steps7657–7668 show training instability, not merely a plot issue.
The release at50 cannot cause the earlier collapse. No registered stable window.

Read-only interventions during planning used32 fixed style-stratified test pages,
four/style, frozen VQ. The Hungarian rates below are small-subset diagnostics,
not full-test gates; the new training probe instead uses the full-val mapping.

| Checkpoint | Normal eval recon / 1:1 | Encoder BN batch-stat recon / 1:1 | Decoder BN batch-stat recon / 1:1 |
|---|---|---|---|
| W50 best10 |.04694 / .6863|.04556 / .6719|.04698 / .6863|
| W50 snapshot24 |.17371 / .0493|.10761 / .0493|.07691 / .0493|
| W50 current197 |235.6333 / .0757|235.6653 / .0853|.05887 / .0757|
| PN REPO best-val177 |.01660 / .6969|.01410 / .7094|.00391 / .6969|
| PN REPO current199 |.03948 / .7344|.04295 / .7313|.00371 / .7344|

Decoder-only interventions leave assignments exactly unchanged. Thus decoder BN
accounts for much of the reconstruction-mode failure at the measured endpoints,
while semantic/usage degradation persists when encoder statistics are changed.
This does not establish the original trigger at11; no checkpoint11 was retained.
The best10 also misses the formal gate (active25, coverage23); do not continue
current197 just to complete the remaining two epochs.

## Next controlled experiment: S7

C128-W50 already shares REPO's AdamW, loss weights, EMA and scheduler. S7 uses
the same UppercaseLettersV2 manifest and200-epoch budget, changing content/native
VQ width to512 and comparing W0 with W50. No BN architecture fix, weight pulse,
dataset regeneration or pretrained initialization. Two new runs share seed0,
initial model, sampling order and balanced diagnostic pages; a single seed does
not establish a general dimension effect or bitwise correspondence to C128.

Retain current/macro-best/best-val plus8 snapshots (max11). Every100 training
steps and every epoch log online usage, explicitly a changing-model statistic.
Every complete validation run four state-neutral BN modes on32 balanced pages,
using the one Hungarian mapping from normal full validation. These probes never
affect loss or checkpoint selection. Additional early grids10/11/12/15/20 do not
add permanent checkpoints. Resources:1GPU/32GB/4CPU/12h each, ws-ia.

Full-test acceptance uses the established macro/usage/reconstruction gate with
color separately reported. Compare first8-of-10 stable windows and persistence
after50; if both C512 runs still fail or remain BN-sensitive, next isolate
normalization instead of attributing every failure to content width.

## Implementation / preflight provenance

64 regression tests passed on GPU node ws-l4-005 (Job189965). Two scratch smoke
runs on ws-l4-003 used two training batches and the complete2600-page validation
(67,600 fragments), including all7 CSV/PNG families and balanced matrix SVG/JSON.
They verified the49→50 decoder switch, matching initialization/sampler/pages,
current/best-val/snapshot output, and unchanged model/optimizer/buffers/modes/RNG
across the four-mode BN diagnostic. Existing unit tests exercise macro and
best-val selection/rotation and snapshot retention; extra grid epochs do not
increase the periodic checkpoint budget. Smoke artifacts are not formal evidence.

Retained lightweight preflight record: `logs/diagnostics/20260913_s7_preflight/summary.json`.

| Item | SHA256 |
|---|---|
| Matched initial model | `11d6c4f25a85967b86447086f069f4f3e1a515b392ef392037619fd5c04d37a7` |
| Matched first epoch sampler order | `0f69d3ac14e130a19b5ee2ef1327c41a45d77e6bde2a1ddc50af8549291ecff1` |
| Matched32-page BN subset | `a478bbec44f89c4f163b41794ffaa0cf59760cb387be17b6e2f36bb21318ce03` |
| W0 submitted config | `96aade8ca4d96a349f4729860dc25407c41e4633a8f41b89dcd15edfc26528a4` |
| W50 submitted config | `93062809e2c42aa3522cb66589b5cf6be84cea5478b25c2839c60b18dbc60bf3` |

The shared data hash is pinned in both configs. Slurm stores Git HEAD, dirty
status, tracked patch, per-file source hashes, source archive and the exact
submitted config under each run's `reproducibility/`; no commit/push is performed.
After adding visible single-epoch plot markers, four targeted GPU tests and
redrawing/visual inspection passed on ws-l4-003. The exact smoke directories
`logs/smoke-s7-w0` and `logs/smoke-s7-w50` were then deleted (about1.27GiB);
only the lightweight summary above remains. No formal checkpoint was deleted.

## Formal submission

Both were registered PREPARED in ACTIVE before submission. Confirmed RUNNING
with CUDA training on separate RTX5000 Ada GPU nodes; Slurm allocates1GPU,
32GB,4CPU,12h per run. No QOS/concurrency settings were changed.

| Run name | Job | Node | Slurm start (Asia/Dubai) |
|---|---:|---|---|
| `20260913-1106__VVI-RQ2-V2-C512-K26-W0-S0` |189975|ws-l1-011|2026-09-13T11:06:43+04|
| `20260913-1106__VVI-RQ2-V2-C512-K26-W50-S0` |189976|ws-l4-003|2026-09-13T11:06:49+04|

Read-only verification within Job189976 confirmed both actual initial-model and
BN-page hashes match preflight. Both run-local config backups equal the source
config semantically except the timestamped name; submitted config copies are
byte-identical. Data manifest and every submitted Python/YAML/shell source hash
match the saved provenance. Git HEAD is
`e0bb0c6b70b5798c0dddc69c6903f9cdf315568a`, dirty state preserved and archived.
Online100-step usage CSVs are already being written. ACTIVE contains exactly the
two new Run IDs, with no overlap with ARCHIVE; no smoke directories remain.
No training results are inferred from these startup checks. Acceptance is due
after termination, on normal full-test macro-best/best-val/current.
