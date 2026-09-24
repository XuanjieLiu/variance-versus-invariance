# VVI-HEX-S1 — PhoneNums-to-Hex16 Extension

Status update2026-09-16: both formal runs completed199/exit0 and passed their
best/current full-test semantic gates; archived, not running. See
[acceptance and uppercase comparison](20260916_HEX_ACCEPTANCE_UPPERCASE.md).
Submission/running descriptions below are historical provenance.

Approved2026-09-15 after the S9 NoNorm acceptance. The user requests a smaller
semantic step than26 uppercase letters and a later hexadecimal-addition dataset.
This round trains VVI only, not an adder; native VQ remains512D.

## Controlled protocol

Direction RQ2; RQ2-H21 tests transfer of the useful PhoneNums REPO bundle to16
contents; RQ2-H22 tests permanent removal of fragment-specific decoder style
variation. RQ1 low native-VQ dimension remains the downstream goal, deferred
until stable semantics are demonstrated.

| Run ID | Decoder style input |
|---|---|
| VVI-HEX-K16-REPO-S0 | ordinary fragment style throughout |
| VVI-HEX-K16-REPO-MEAN-S0 | page mean of16 fragments broadcast throughout; no release |

Both scratch seed0,200 epochs0–199, C512/S512/nativeVQ512/K16, batch32,float32,
original encoder/decoder BN. AdamW lr.001/wd.1/betas.9,.999/eps1e-8; exponential
scheduler factor.98 every20 epochs (same floor.02). Relativity15,
recon/content/style/sample/fragment weights1, commitment.1; EMA.98, dead threshold16,
kmeans initialization and STE unchanged. The MEAN arm changes decoder input
only; raw fragment z_s still enters V3 statistics and probes. No checkpoint load,
loss schedules, normalization removal, warmup switch or lowered VQ dimension.
Configs: `configs/hex/v2/cfg_vvi_hex_k16_repo[_mean]_seed0.yaml`.

With80,000 train pages both runs have2500 steps/epoch,500,000 total, matching
PhoneNums REPO optimizer-step budget, not its total fragment compute. Batch
fragments320→512 and K10→16 leave uniform expected usage32/code, so dead threshold16
is not automatically rescaled. This is not a pure class-count causal experiment:
page width, pairwise statistics and glyph identities also change. Original
uppercase training had20,800 train pages and a different fit-to-bbox layout.

## Independent data contract

`../data/HexDigitsV2` never replaces PhoneNums or letter data. Alphabet
`0123456789abcdef`, labels0..15; each page contains every character exactly once,
in a unique per-style permutation. Eight original PhoneNums styles.100,000 pages:
each style10,000 train +1250 val +1250 test. Every(content,style) has1250 validation
and1250 test fragments; test totals10,000 pages/160,000 fragments.

Use original NumberOffice/ITCKRIST.TTF,48px glyph size,32×48 RGB fragments,
512×48 page; no glyph-specific offsets or uppercase fit-to-bbox. Corrected
signed-float Gaussian noise and complete final blur strip; foreground/background,
jitter/translation and blur-kernel distributions otherwise unchanged.
Generation/split seeds0; independent page seeds and workers-independent outputs.
Floyd sampling/unranking uses O(number of pages) memory, never materializes16!.
Manifest records label values, pair counts, page seeds, PNG/font/source hashes,
code SHA/dirty state and dependency versions. Build in staging and publish only
after validation; refuse existing targets and collisions. Loader validates the
manifest/files and RGB shape; full splits have deterministic order and no crop RNG.

## Diagnostics and decision gate

All existing loss/codebook/ratios/colors/leakage/BN/online-usage/optimization-scale
diagnostics remain opt-in and state/RNG neutral. Add global-mapping `0–9` versus
`a–f` accuracies to per-epoch CSV/PNG and evaluation/reconstruction/matrix JSON;
per-label and per-style results also use the same global Hungarian assignment.
Counts/probabilities remain exact; display uses conserving column rounding,
normal font, leading-dot probabilities and hidden zeros. Recon annotations white.
Paired grid16×16 and matrix16×16 every25 completed epochs; extra epochs
0,10,11,12,15,20,50,100. Masks remain noisy-input estimates, not ground truth masks.

Retain current + macro-best + best-val +8 periodic snapshots, max11 files.
Health-gated macro ranking: active>=15, perplexity>=11.2, coverage>=12,
weighted purity>=.20. Better macro wins; ties use lower validation total loss.
Validation/test labels never enter training loss; no Hungarian-best selection.

Formal full10k test evaluates all three retained roles; primary semantic outcome
requires macro/weighted/Hungarian all>=.75, active16, perplexity>=13, coverage>=15,
recon<=.30. Stable10-epoch window needs at least8 epochs with macro/Hungarian>=.70,
active16/perplexity>=13/coverage>=15. Passing only at best, not current, is transient
success. Report colors separately and do not call semantic passing full disentanglement.

- REPO success: replicate seed before lowering native VQ dimension.
- Only permanent MEAN succeeds: test maintenance and ordinary single-fragment
  decoding compatibility before downstream integration; page-context-trained
  decoding is not automatically a single-image capability.
- Both fail: inspect new-symbol confusions, cross-style disagreement and temporal
  deterioration before attributing failure to the number16 or adder capacity.

## Operational acceptance

All Python/generation/tests/evaluation/training on ws-* compute. Data tests must
pass before formal generation; full regression and a complete2500-step train +
10k-page validation smoke for each arm before submission. Check identical initial
model/sampler/BN-page hashes, permanent modes, strict state load, all artifacts,
16×8 balance, group mapping, checkpoint budget and no model/optimizer/RNG mutation.
Inspect actual a–f rendering and grid legibility; delete smoke immediately, never
add to ledger. Then register both formal runs ACTIVE before sbatch, fill jobs and
provenance. Each ws-ia/1GPU/32GB/4CPU/24h. No commit/push or old-run continuation.

## Implementation progress

Data tests:6 passed on GPU Job193695. Full regression:84 passed in27.736s on
GPU Job193697; final integration suite84 passed in14.119s on Job193708.
Generation Job193696 published100,000 pages and exited0. GPU audit193711 verified
every PNG hash/RGB/size, source/font hashes, split counts and label slicing;
the16×8 original-only grid was visually checked, including lowercase a–f.
No font size, glyph offsets or original digit rendering was changed.
The immutable manifest SHA256 is
`120228b3129feddde5c9d64b0cff06a293bceb163d385ae5ce03bf2c8947254c`.
Both configs are pinned to it. Audit/preview artifacts:
`logs/diagnostics/20260915_hex_dataset/{validation.json,hex_inputs_16x8.png}`.
Complete-epoch GPU smokes193717/193718 both passed on ws-l1-002/ws-l1-004:
2500 optimizer steps +10,000 validation pages/160,000 fragments each. All10
CSV/PNG diagnostic families,16x8 exact1250 counts, balanced matrices, checkpoint
strict reload,512-page manual/health evaluation and state/RNG neutrality passed.
Both saved current/best-val/snapshot0; neither had a health-eligible macro-best
at epoch0, which is an expected conditional selection outcome, not a test failure.
Raw smoke metrics are preflight only, not formal scientific acceptance.

The first smoke launch stopped before training because implicit historical BN
did not emit normalization metadata. Both configs now explicitly declare batch
normalization; the model and initial hash are unchanged. Four protocol tests
were rerun/passed before the successful complete smoke. A visual check found
the short grid's title overlapping its column headers; physical header spacing
was corrected without changing values or training. GPU layout check193725 passed
for both real8-page grids and retained the cached full-validation mapping/codes.
Its initial attempt193724 only failed on a diagnostic JSON key (`counts` versus
`raw_counts`), corrected before retry; no training was repeated for layout.
Both exact650MiB smoke directories were deleted after validation (about1.27GiB
released), not entered into ACTIVE/ARCHIVE. Lightweight preflight summaries and
layout previews remain in `logs/diagnostics/20260915_hex_preflight/`.

Matched provenance verified:

- Initial full state: `00d3ef1c717de57f8b412a9ce89603b7a2af5c95b36629f22c44a43de8b6444d`.
- First-epoch sampler order: `46cb7c17b68270824b343407a449dba9bc81143dab11981fa1185708b03c24d6`.
- Fixed BN pages: `ed1d00aed9bbca4d1aea956f5fb015a3c22c02408165eebfd0de2d4586eaad3b`.
- REPO config: `07e58be4aa369f39e3da8eb185f891bcd51749605a6bb23dfa210862682b109b`.
- MEAN config: `58d7dad1648856fdd4c0fdd419a08e38076c1fd8a102480022d7637527655436`.
- Git HEAD: `e0bb0c6b70b5798c0dddc69c6903f9cdf315568a`, dirty; no commit/push.

Both formally register ACTIVE before sbatch. Job-specific source archives,
tracked/untracked code hashes, dirty state and submitted config are preserved
under each run's `reproducibility/`; the immutable dataset manifest is pinned above.

## Formal submission and startup verification

Both registered before submission on2026-09-15. Slurm confirms RUNNING on two
compute nodes, each ws-ia/1GPU/32GiB/4CPU/24h; NVIDIA RTX5000 Ada, CUDA visible.

| Run name | Job | Slurm start (Asia/Dubai) | Node |
|---|---:|---|---|
| 20260915-1121__VVI-HEX-K16-REPO-S0 |193731|2026-09-15 11:21:33|ws-l1-007|
| 20260915-1121__VVI-HEX-K16-REPO-MEAN-S0 |193733|2026-09-15 11:21:39|ws-l1-008|

Both started scratch epoch0/199 and progressed beyond800/2500 train steps at
the startup check. This is launch acceptance, not a statement of scientific
success. First-step V3/commit values match across arms; only recon differs as
expected from the decoder-mode intervention. Different devices/scheduling may
still yield small floating-point differences; initialization/order equality is
not a guarantee of bitwise identical training trajectories.

A read-only check on ws-l1-007 (overlap step within193731) verified both config
backups are byte-identical to submission, effective run configs differ only in
the injected timestamped name, initial model and fixed BN-page hashes match
preflight, and data manifest hashes match. Every submitted Python/YAML/shell
source hash matches the implementation at launch. Both source archives share
SHA256 `63dd911a728754303c43bd0876c78971e1bff1f57bb103a82d0360f16cac1c76`.
Source snapshots capture pre-submission ledger state; live Markdown subsequently
records assigned jobs/nodes. ACTIVE/ARCHIVE IDs are disjoint and smoke artifacts
are absent. S9 remains archived; no old formal run is resumed or modified.
