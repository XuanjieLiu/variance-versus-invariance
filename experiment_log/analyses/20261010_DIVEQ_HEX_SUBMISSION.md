# 2026-10-10 — Hex16 DiVeQ, ordinary fragment decoder

User requested two DiVeQ experiments, C/S512 and C/S4, and explicitly clarified
**no MEAN**. Both start from scratch for200epochs, without v3_rank, with native
dimension equal to content dimension, K16 and the original Hex REPO BN protocol.
The previously cancelled rank pair is not restarted.

## Provenance and scope

- Repository HEAD: `6f8bf8423b2e372dc0c93be60b4cb8abd703c9fc`, dirty; prior user
  documentation edits preserved. No commit/push. Per-run provenance archives
  actual source (including untracked implementation/configs and MIT license),
  tracked diff, git status, source SHA256 and submitted config checksum.
- DiVeQ reference: `7e8cc2c3885707576510273da097465949f04063`, ordinary stochastic
  variant. See [implementation/defaults and engineering differences](../../manuals/diveq.md).
- HexDigitsV2 manifest:
  `120228b3129feddde5c9d64b0cff06a293bceb163d385ae5ce03bf2c8947254c`.
- C512 config SHA256:
  `bde07f6f7579610c037f54f0d976c722246992100f1068432f0aa0943732820e`.
- C4 config SHA256:
  `bc1df5a20e108c6b4751d26aef9caefaabe7d208de5ce3e80be231c90a001334`.
- No auxiliary commitment loss/EMA. All other V3 weights, optimizer/BN/data and
  budget copy the ordinary Hex REPO reference. This changes a quantizer bundle;
  C4/S4 also changes style capacity. Neither contrast isolates a single mechanism.
- Different widths have different initialization and sampled training-order
  hashes under the existing RNG protocol. Same manifest/fixed diagnostic pages
  confirmed; do not claim identical model states or first-epoch order.

## GPU preflight

Job224394, ws-l1-009,2026-10-10 15:44:03–15:46:45+04,COMPLETED/exit0.
Python compilation and41unit tests passed:11DiVeQ,11rank regressions,
4training diagnostics,10codebook health,5evaluation path tests. The initial
standalone test attempt had one inference-tensor assertion error in the test
itself, fixed by detaching lookup atoms; the complete suite was then rerun.

Each arm trained100real updates, triggering the100-step replacement hook,
then evaluated all10000validation pages/160000fragments. Checks include nonzero
finite encoder/decoder/codebook gradients, all loss/metric/diagnostic artifacts,
16×8actual validation count1250, ordinary-fragment recon and balanced matrices,
diagnostic model/BN/VQ/optimizer/RNG neutrality, strict checkpoint+Adam reload,
retention bound and manual512-sample evaluation using the current alias.
Temporary smoke directories were deleted immediately. Only compact reports
remain in `logs/diagnostics/20261010_diveq_preflight/` (not formal ledger runs).
These100-step metrics are not experiment acceptance or evidence of success.

## Formal jobs

Registered under VVI-HEX-DIVEQ-S1 before submission. Each job requests ws-ia,
one GPU,32GB,4CPU,24h;200epochs=500000updates. ACTIVE records actual Job IDs.

| Run ID | Run name | Job | Node | Started(Asia/Dubai) |
|---|---|---|---|---|
| VVI-HEX-K16-DIVEQ-C512S512-S0 | 20261010-1549__VVI-HEX-K16-DIVEQ-C512S512-S0 | 224397 | ws-l1-013 | 2026-10-10 15:49:40 |
| VVI-HEX-K16-DIVEQ-C4S4-S0 | 20261010-1549__VVI-HEX-K16-DIVEQ-C4S4-S0 | 224398 | ws-l1-009 | 2026-10-10 15:49:40 |

At15:50+04 both are RUNNING and have advanced beyond500updates in epoch0/199,
with finite training losses and commit_loss0. A read-only srun check inside
job224397 verified both saved configs equal submitted configs except run name,
the source config bytes/hash match the provenance copy, each initial model hash
matches its respective smoke, MIT attribution is included, and neither Run ID
appears in ARCHIVE. These are startup checks, not semantic acceptance. No Python
or experimental computation was run on a login node.

At completion evaluate macro-best/best-val/current on full10000-page test,
including stability, colors/leakage and native spectrum, then archive.
