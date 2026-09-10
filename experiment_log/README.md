# VVI research dashboard

This directory separates cross-run research conclusions from the operational run
ledger. Direction files synthesize evidence; `SETTINGS.md` defines controlled
experiments; `ACTIVE_RUNS.md` and `ARCHIVE.md` own run lifecycle state.

| Direction | Current conclusion | Confidence | Active setting | Next decision |
|---|---|---|---|---|
| [RQ1: content width and native low-D VQ](directions/RQ1_LOW_DIMENSION.md) | At flat K128 scratch, C32 fails while C128 reaches one transient `.771` macro checkpoint and then decays. Since the application only needs a compact discrete interface, keep the encoder wide and target `vq_codebook_dim=128`. | high | none | Warm-expand K52-WARM to K128/D128 instead of continuing either scratch run. |
| [RQ2: phase catalysis](directions/RQ2_PHASE_CATALYSIS.md) | Frozen early/late checkpoints show reconstruction switching from a continuous-style shortcut to VQ content; color fidelity degrades. The trigger remains unknown. | high for endpoints; catalyst being tested | VVI-RQ2-S3: CTRL/W100 running; MEAN QOS-queued | Matched200-epoch C128/K26 CTRL, page-mean,100-epoch mean warmup; compare onset, leakage, color and persistence. |
| [RQ3: joint K/D scaling](directions/RQ3_CODEBOOK_SCALING.md) | Warm K52 and PCA-warm K104 succeed, while flat K128 scratch is unstable even at C128. Healthy usage is not enough: semantic parent structure must be inherited or explicitly constrained. | high | none | Generalize the successful PCA-warm transform from K52 to K128 with D128. |
| [RQ4 bridge: arithmetic-guided unseen-style adaptation](directions/RQ4_ARITHMETIC_STYLE_ADAPTATION.md) | V5 Teacher A passed its gate; V5 is canonical for all AGUSA hypotheses and conclusions. | teacher qualified | none in VVI | Follow the [V5 dashboard](../../v5/EXPERIMENT_SUMMARY.md): matched Teacher B, then OOD controls. |

## Metric contract

- Primary selection metric: validation macro atom purity over active codes.
- Anti-degeneration metrics: usage-weighted purity, active codes, usage
  perplexity, dominant-label coverage, and dominant-code count min/max/CV.
- Hungarian accuracy is diagnostic only when `K != content_count`.
- For `K>26`, atom purity is necessary but not sufficient. Also report native
  VQ alias within/between geometry and whether a small native-space addition
  probe generalizes across styles.
- The K104 probes show that even purity plus the current alias-distance metrics
  are not sufficient for arithmetic: a per-image alias target can remain
  multimodal. Keep VVI acceptance and addition acceptance as separate gates.
- RQ2-S1 shows that unconstrained macro purity can favor rarely used atoms:
  its macro-best had macro=.564 but weighted purity=.142 and perplexity=5.32.
  Before the next formal run, checkpoint selection should rank macro purity only
  among candidates meeting configurable active/perplexity/coverage floors while
  still retaining exactly one best plus current checkpoint.
- Formal conclusions use the full 2,600-sample test split; quick subsets and
  smoke runs never enter the archive.

## Lifecycle

1. Register a hypothesis in one direction file and define its setting in
   `SETTINGS.md`.
2. Add a formal run to `ACTIVE_RUNS.md` before submission.
3. On completion, failure, timeout, or cancellation, remove it from ACTIVE and
   add one row to `ARCHIVE.md`.
4. Update only the evidence synthesis and decision gate in the relevant
   direction file; do not duplicate long conclusions in the run tables.

## Prioritized next experiment queue

Current user priority: **VVI-RQ2-S3 three-arm style-bypass ablation**, with
25-epoch reconstruction grids and permanent snapshots. Other proposals below
remain deferred until this experiment is reviewed.

1. **VVI-RQ13-S3 K52-WARM→K128/D128.** Generalize local warm expansion while
   keeping the discrete interface 128D and the encoder/decoder pathway 512D.
2. **VVI-RQ1A C96/K26 midpoint.** Narrow `(64,128]`, then replicate the boundary.
3. **Alias-invariant arithmetic objective.** Explain and remove the multimodal
   per-image alias target before interpreting another small-adder capacity test.
4. **VVI-RQ2 matched branch pair.** Use the retained historical epoch80 and
   compare original decoding with page-mean style decoding to test continuous
   content leakage. See the [2026-09-10 mechanism analysis](analyses/20260910_STYLE_BYPASS_PHASE_TRANSITION.md).
5. **VVI-RQ3-S2 full-width K52 cold start.** The C64/K52 pilot failed but is
   content-width-confounded; retain the C512 control only as a mechanism test.
