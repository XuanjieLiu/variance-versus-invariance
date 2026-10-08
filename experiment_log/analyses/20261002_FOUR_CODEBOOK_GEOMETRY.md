# Four native-codebook geometry comparisons (posthoc, 2026-10-02)

Requested follow-up to the Hex16 visualization. No new training, config changes,
checkpoint selection changes, or formal-run ledger entries. Select each run's
existing validation macro-best checkpoint, not its best-val or current checkpoint.

## Results

| Run | Macro-best epoch | Native atoms | PC1 variance | PC1+2 variance | Participation-ratio dimension | Full-test Hungarian |
|---|---:|---|---:|---:|---:|---:|
| `20260912-1420__VVI-PN-K10-REPO-S0` | 0 | 10 x 512 | 99.099584% | 99.907874% | 1.018186 | .800000 |
| `20260912-1420__VVI-PN-K10-PAPER-S0` | 14 | 10 x 512 | 99.994139% | 99.999976% | 1.000117 | .391680 |
| `20260910-1635__VVI-RQ2-C128-K26-CTRL-S0-R1` | 166 | 26 x 128 | 99.387911% | 99.998873% | 1.012317 | .911775 |
| `20260910-1635__VVI-RQ2-C128-K26-W100-S0-R1` | 197 | 26 x 128 | 99.521871% | 99.994314% | 1.009609 | .931346 |

All four native codebooks are nearly one-dimensional by centered variance;
this is not merely the appearance of t-SNE/Isomap. It does not mean exact rank1,
and small-variance directions may still be functionally important. PAPER is the
most collinear here but has the lowest semantic mapping accuracy: low rank is
not sufficient for successful semantic separation. These measurements alone
do not identify the loss/optimization mechanism causing low rank.

REPO epoch0 is **after the first completed training epoch (2,500 updates)**,
not a randomly initialized codebook. The four checkpoints have different ages,
datasets and training protocols; this comparison is not a controlled ablation.

## Method and provenance

Extract raw `vq._codebook.embed` atoms; do not use decoder-space projection,
label centroids, feature standardization, normalization or whitening. Labels
are full-test Hungarian assignments used only for annotation and sorting, never
for dimensionality-reduction fitting. A displayed character is not a guarantee
that all images assigned to that atom have that character.

PhoneNums uses existing hash-verified full10,000-page acceptance counts. The
historical uppercase reports retained scores but not counts: recover the full
2,600-page test counts with encoder/VQ only, in eval/inference mode with frozen
VQ, and verify every model parameter/buffer is unchanged. Recovered accuracies
agree with the historical reports. Cached mappings include raw counts and hashes.

Match the previous Hex plotting parameters: PCA full SVD (2D uses first two of
the same three axes); t-SNE exact Euclidean, perplexity4, seed0, PCA initialization,
learning rate50, early exaggeration12, max3000 iterations; Isomap connected
5-neighbor graph, Dijkstra, ARPACK with seed0. Nonlinear 2D/3D and cross-run fits
are independent; orientation and scale are not aligned. t-SNE spread is not
evidence of extra native dimensions, and ten/26 atoms do not densely sample a
manifold. Use the native spectrum and distances for geometric conclusions.

## Outputs and reproducibility

In each run's `codebook_geometry/`, prefer:

`cp_best_macro_atom_purity_epoch<E>__native-vq__seed0__p4__k5__readable-v4/`

Each directory contains six individual PNG/SVG figures (`pca_2d`, `pca_3d`,
`tsne_2d`, `tsne_3d`, `isomap_2d`, `isomap_3d`), `overview.png/svg`,
`coordinates.csv`, native atoms/distances, and `geometry_report.json` with
checkpoint/config/script/mapping hashes, dependency versions, fit parameters,
native singular values and projection fidelity. All axes have equal geometric
scale. Deterministic label repulsion moves annotations/leader lines only, never
atom coordinates; preliminary `geometry-v3` outputs remain available.

Generic entry: `scripts/plot_codebook_geometry.py`; the historical
`scripts/plot_hex_codebook_geometry.py` entry remains compatible. On a GPU
compute allocation with `xuanjie` activated:

```bash
python scripts/plot_codebook_geometry.py --run <run-name> --output-tag <new-tag>
```

Existing output directories are refused. The tool supports K equal to the
dataset's content count, as in these runs; it does not silently label an oversized
codebook with a one-to-one mapping. Tests cover K10/16/26 mappings, inactive atoms,
invalid counts, hash-verified mapping reuse and label-order mismatch rejection.
All numerical work and tests use Slurm GPU compute allocations. No source
checkpoint modifications, commit or push.
