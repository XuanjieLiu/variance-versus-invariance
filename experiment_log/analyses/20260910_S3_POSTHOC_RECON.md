# S3 posthoc reconstruction — 2026-09-10

Run: `20260801-0615__VVI-S3-C512S512-CB512-S0` (C512/S512/K26).
GPU: ws-l1-017. Existing checkpoints frozen, model in eval mode and VQ updates
disabled; this is posthoc evaluation, **not decoder refitting**.

For each checkpoint, recompute Hungarian mapping on the complete2,600-page test
set (67,600 letters). Render the same eight test pages, one per color,26 rows ×16
columns, original/recon pairs with code→Hungarian-label annotations. Save PNG,
per-code mapping CSV, and JSON containing checkpoint hash and selected filenames.

| Checkpoint | Full-test macro | Full-test Hungarian | Active | Perplexity | Coverage | Eight-page grid recon MSE |
|---|---:|---:|---:|---:|---:|---:|
| macro-best145 | .834025 | .809423 | 26 | 25.1824 | 25 | .153276 |
| current149 | .810914 | .771198 | 26 | 24.2648 | 26 | .161027 |

The reconstruction column is **only the fixed eight-page grid**, not full-test
reconstruction. Previously archived full-test reconstruction values are .154404
and .159430; they were not recomputed by this grid command.

Visual inspection of both grids shows predominantly gray-brown glyphs instead
of preserving the eight source colors. Epoch149 also has visibly worse output
on the selected strongly corrupted red page. Some letter reconstructions are
inconsistent even where Hungarian annotation is green; assignment statistics
do not directly measure the glyph actually produced by the decoder.

Conclusion: the color-reconstruction problem also occurs at native D512, so it
cannot be attributed exclusively to reducing C/VQ dimensionality to128. High
atom purity / one-to-one assignment is insufficient to establish successful
content-style reconstruction. These frozen images do not isolate the encoder
versus decoder cause or prove complete removal of color information. See the
[earlier channel interventions](20260910_STYLE_BYPASS_PHASE_TRANSITION.md) for
the evidence of a change in reconstruction routing and its causal limitations.

Artifacts:

- [Epoch145 grid](../../logs/20260801-0615__VVI-S3-C512S512-CB512-S0/posthoc_recon/reconstruction_grid__cp_best_macro_atom_purity_epoch145__test-full-map__8styles.png)
- [Epoch149 grid](../../logs/20260801-0615__VVI-S3-C512S512-CB512-S0/posthoc_recon/reconstruction_grid__cp_current_epoch149__test-full-map__8styles.png)
- [Epoch145 JSON](../../logs/20260801-0615__VVI-S3-C512S512-CB512-S0/posthoc_recon/reconstruction_grid__cp_best_macro_atom_purity_epoch145__test-full-map__8styles.json)
- [Epoch149 JSON](../../logs/20260801-0615__VVI-S3-C512S512-CB512-S0/posthoc_recon/reconstruction_grid__cp_current_epoch149__test-full-map__8styles.json)

Next: user-authorized instrumented S3 rerun under RQ2-S4, plus resumption of
the three RQ2-S3 arms. Do not change objective weights or decoder policy in S3.
