# Hex16 native-codebook geometry (posthoc, 2026-10-02)

Requested visualization only; no training, checkpoint selection changes or new
formal runs. Existing macro-best checkpoints: permanent MEAN epoch171 and
ordinary REPO epoch192. Extracted `vq._codebook.embed`, shape1×16×512, without
project_out, normalization, whitening, label averaging or encoder forwards.
Annotations reuse each checkpoint's existing full10000-page test Hungarian
mapping after verifying its checkpoint SHA256. Both mappings have accuracy1.
Labels and colors identify0..9/a..f but do not enter dimensionality reduction.

## Deliverables

Under each run's `codebook_geometry/`:

- MEAN: `cp_best_macro_atom_purity_epoch171__native-vq__seed0__p4__k5__readable-v2/`
- ordinary: `cp_best_macro_atom_purity_epoch192__native-vq__seed0__p4__k5__readable-v2/`

Each directory has six individual PNG/SVG figures (`pca_2d`, `pca_3d`, `tsne_2d`,
`tsne_3d`, `isomap_2d`, `isomap_3d`), an `overview.png/svg`, `coordinates.csv`,
raw atoms sorted by character (`native_atoms_by_content.npy`), native Euclidean
distances, and `geometry_report.json` (hashes, dependency versions, parameters,
mapping provenance, explained variance and projection fidelity).
The preliminary non-suffixed figures remain as intermediate outputs; prefer
`readable-v2`, which separates nearly-collinear labels with leader lines.

PCA: full SVD, shared first3 axes for2D/3D. t-SNE: independent2D/3D fits,
exact Euclidean, perplexity4, seed0, PCA initialization, learning rate50,
early exaggeration12, maximum3000 iterations. Isomap:5-neighbor Euclidean
graph, verified connected without implicit disconnected-graph repair,
Dijkstra shortest paths and ARPACK eigensolver with fixed NumPy seed0.
All axes use equal geometric scale, and labels are offset without moving atoms.

## Measured geometry

| Model | PC1 variance | PC1+PC2 variance | PC1+PC2+PC3 variance | Participation-ratio dimension |
|---|---:|---:|---:|---:|
| MEAN171 | 99.9934137% | 99.9999952% | 99.9999995% | 1.000132 |
| REPO192 | 99.9281066% | 99.9999857% | 99.9999989% | 1.001439 |

Both sets of16 atoms lie nearly on a line in512D; this is not merely a t-SNE
appearance. PCA2D preserves the complete pairwise-distance rank ordering in
these checkpoints (Spearman1.0; neighborhood trustworthiness at k3 also1.0).
Characters are not ordered numerically along that line. High semantic purity
and low geometric variance dimension therefore do not by themselves establish
an arithmetic-compatible representation or successful low-dimensional training.
Residual low-variance directions can still affect downstream predictions.

t-SNE3D looks more spatially dispersed despite the essentially1D native geometry;
do not interpret nonlinear plot distances/shapes as original atom geometry.
These are16-point visualizations, not evidence for a densely sampled manifold.
Fits across runs are independent and not axis-aligned. Parameters are fixed for
reproducibility rather than optimized for an aesthetically desired outcome.

## Reproduce / checks

On a GPU compute allocation, activate `xuanjie`, then run:

```bash
python scripts/plot_hex_codebook_geometry.py --run 20260915-1121__VVI-HEX-K16-REPO-MEAN-S0 --output-tag another-view
python scripts/plot_hex_codebook_geometry.py --run 20260915-1121__VVI-HEX-K16-REPO-S0 --output-tag another-view
```

The script refuses existing output directories, malformed mappings, mismatched
checkpoint hashes, disconnected Isomap graphs, nonfinite embeddings and login
nodes. Verified six individual PNGs per model, inspected both overviews, and
verified unchanged source checkpoint hashes. Both Slurm geometry invocations
finished successfully and released resources. No full evaluation rerun, no
changes to model weights/configs, and no commit/push.
