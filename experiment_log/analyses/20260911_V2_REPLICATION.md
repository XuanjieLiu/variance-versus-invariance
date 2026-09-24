# V2 preparation and old-run acceptance check — 2026-09-11

CTRL/W100 R1 already completed epoch199 and were archived on 2026-09-11.
Their existing best/current full-test reports are retained; no duplicate evaluation
or archive rows are needed. The stale ACTIVE reference in ARCHIVE was corrected.

| Old-data arm | Macro-best epoch | Full-test macro | One-to-one | First stable window |
|---|---:|---:|---:|---|
| CTRL | 166 | .91455 | .91178 | 150–159 |
| W100 | 197 | .93249 | .93135 | 76–85 |

W100 advanced high-purity routing at this seed and maintained it after release;
neither arm recovered source colors. This motivates a controlled data-version
replication, not a claim that either rendering bug fully explains the collapse.

## V2 artifact

- Dataset: `../data/UppercaseLettersV2`, generation/split seed0, renderer version2.
- Train/val/test: 20,800/2,600/2,600 pages. Per style: 2,600/325/325.
- Each page has all26 letters once; each validation/test (letter,style) count325.
- Manifest SHA256: `d455680108040b20d4880aff2bba29b4de0878f97cb0d37fd10ca91b8f1dc5b6`.
- Generation job185676 ran on ws-l1-016; published only after per-file RGB/size,
  checksum and balance checks. Four dataset tests passed before generation.
- Independent per-page RNG; serial/parallel tiny datasets matched byte-for-byte.
- Original data and legacy rendering defaults are preserved; no clean targets or
  ground-truth masks were saved.

## Interpretation contract

Foreground masks come only from noisy input: outer2px channel median background,
interior max-channel difference>.15, minimum16 pixels. RGB MAE and RGB-chroma RMSE
are equal-fragment means over valid masks; invalid masks are counted, never scored
as zero. Predictions remain unclipped for metrics. Noise and mask errors remain
limitations; this is not CIELAB DeltaE.

Confusion plots use full balanced validation/test, actual-count checks, raw-count
Hungarian assignment, and P(code|content) columns. Display rounding conserves each
column to1.00; CSV/JSON retain unrounded probabilities and original code IDs.
Reconstruction and confusion share one mapping. All new measures are diagnostic.

Formal setting/configs and decision gate: [S5](../SETTINGS.md#vvi-rq2-s5--v2-data-correction-replication).

## GPU preflight

Job185693 on ws-l1-016 passed all54 regression tests (5.555s), verified every
published PNG checksum, and ran both smoke arms with two training batches plus
the complete2,600-page validation. Both produced loss/codebook/V3/probe/color
histories and plots, synchronized recon/mask/confusion SVG/PNG/JSON/CSV, current
and periodic checkpoints. The initial two-batch models correctly fail the
unchanged macro-best health floor; low-purity initialization is not saved as best.

- Matched initial state SHA256:
  `ad9ca6d0eabfdf21dc76df594f241fa4c484685adbd50d679ee8b4075c8dbdd6`.
- Matched first-epoch sampler-order SHA256:
  `27ee201cb1f987ae09091bf4d647db3d5621222f762b4087ab805a5fdea15146`.
- Every actual(letter,style) count325; raw and displayed probability columns sum1.
- Monitor begin/collect/finish preserve Python/NumPy/CPU/CUDA RNG. Finish preserves
  every model parameter and buffer exactly (including BN and VQ).
- W100 epoch99 uses page mean; epoch100 uses fragment style; smoke starts at0.
- Manual `run_reconstruction_grid.py --run ... --active_checkpoint current`
  also passed full-test color/matrix export with2,600 pages.
- Matrix text size/contrast and input-estimated foreground overlays were visually
  inspected. Masks follow glyphs but include some noisy pixels, as documented.
- The health evaluator's V2 manifest check initially exposed a missing Path import;
  fixed and verified on GPU with512 balanced pages, including per-style colors.
- After all checks, both smoke directories were deleted (about799MiB). No formal
  artifacts were deleted and smoke runs were not added to ACTIVE/ARCHIVE.

## Formal launch

Both registered before submission and started2026-09-11 around15:21+04:00, with
the shared batch-name prefix `20260911-1522`:

| Arm | Job | Node | Submitted config SHA256 |
|---|---|---|---|
| CTRL | 185700 | ws-l1-016 | `6737fac570f59d87db0cc4ea071a08ee1e82ce48664f69363353d686badfbae7` |
| W100 | 185701 | ws-l1-007 | `3e0a3019d2155b692983646dc6ff360725e8afdfe0bfa06d8db0dbeac6a36125` |

Each has one RTX5000Ada GPU,32GB requested RAM,4CPUs and8h. Both formal initial
state hashes match the smoke value, manifest hashes agree, and logs show the
correct0–199 range and fragment/page-mean modes. Git SHA, dirty patch, source
archive/checksums and submitted YAML are retained in each run's `reproducibility/`.
These are running experiments, not accepted results. Completion requires full
best/current acceptance and the ACTIVE→ARCHIVE + RQ2/dashboard updates.

Read-only verification inside the CTRL compute allocation confirmed both runtime
YAMLs equal their submitted configs except the intended run name, byte-identical
submitted-config backups, matching manifest/initial-state hashes, no active/archive
Run-ID overlap, and no remaining smoke directories. No training state was changed.
