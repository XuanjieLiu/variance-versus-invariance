# v3_rank Hex16 MEAN pair — 2026-10-08

User-confirmed name `v3_rank`; k=4/2 means effective-rank target, NOT codebook K.
Setting `VVI-HEX-RANK-S1`, hypothesis `RQ1-H13`. Both from scratch, no parent
checkpoint loaded. Reference: `20260915-1121__VVI-HEX-K16-REPO-MEAN-S0`.

| Run ID | Run directory name | Job | Status |
|---|---|---|---|
| VVI-HEX-K16-MEAN-RANK4-S0 | 20261008-1623__VVI-HEX-K16-MEAN-RANK4-S0 | 222839 | running ws-l1-008; started16:24:54+04 |
| VVI-HEX-K16-MEAN-RANK2-S0 | 20261008-1623__VVI-HEX-K16-MEAN-RANK2-S0 | 222840 | running ws-l1-014; started16:24:57+04 |

Both configs exactly match the Hex MEAN reference except name, method alias and
rank block: C512/S512/native512/K16, permanent page mean, original BN, batch32,
seed0,200epochs/500k updates, original AdamW lr.001/wd.1/exponential scheduler,
EMA.98/dead16, r15, recon/four V3 weights1,commit.1. Rank weight.01,epsilon1e-12.
No rank warmup or additional regularizer. Original diagnostics and current +
macro-best + best-val + eight25-epoch snapshots retained (max11 files).

Submission: existing `scripts/submit_slurm.sh`, `VVI_TIME=24:00:00`, ws-ia,
1GPU/32GB/4CPU each. Register ACTIVE before sbatch; resources release on job exit.
No commit/push. No modifications to old checkpoints or datasets.

Both formal jobs confirmed RUNNING on compute nodes with NVIDIA RTX5000 Ada
32GB GPUs; epoch0 optimizer updates and rank metrics are present in their loss
CSVs. Formal initial-model hashes match each other and the preflight below.

## Preflight evidence

Slurm222837, ws-l1-008, exit0. Two protocol tests plus11 rank tests passed.
Each arm then ran4 real batch32 optimizer updates and full10,000-page validation
(160,000 fragments;1250 per(content,style)). Verified all existing diagnostics,
rank CSV/PNG fields, balanced matrix, page-mean recon, strict checkpoint load,
`method: v3_rank` evaluation resolution, and diagnostic model/optimizer/RNG
neutrality. These checks are engineering tests, not semantic acceptance.

Both temporary smoke directories were deleted; only small preflight reports
remain at `logs/diagnostics/20261008_v3_rank_preflight/{rank4,rank2,matched}.json`.
Console log: `slurm_logs/20261008_v3_rank_preflight.log`.

Matching hashes across both arms:

- Initial model: `00d3ef1c717de57f8b412a9ce89603b7a2af5c95b36629f22c44a43de8b6444d`
- Full first-epoch sampler: `46cb7c17b68270824b343407a449dba9bc81143dab11981fa1185708b03c24d6`
- Fixed BN diagnostic pages: `ed1d00aed9bbca4d1aea956f5fb015a3c22c02408165eebfd0de2d4586eaad3b`

## Provenance

- Git HEAD: `c2789a7d9ae05de1bbee4af17da34d80a717d195`; dirty implementation intentionally uncommitted.
- Rank4 config: `configs/hex/v2/cfg_vvi_hex_k16_mean_rank4_seed0.yaml`, SHA256 `8e00b685e6ddcb7e3dd84a911d5de6f12ba6126665d42a89ff377d77ebcced9f`.
- Rank2 config: `configs/hex/v2/cfg_vvi_hex_k16_mean_rank2_seed0.yaml`, SHA256 `49e0d16ae05fa49b56b5a5d97fa7f881eaa63116a73e2854a76506e73fd0a603`.
- HexDigitsV2 manifest: `120228b3129feddde5c9d64b0cff06a293bceb163d385ae5ce03bf2c8947254c`.
- Regularizer source: `150324fe303e6ddfcd4efc55215ca30463768272053401be3984a58d89cb885d`.

Each job writes source archive, per-source checksums, dirty patch/status,
submitted config and checksum under its run's `reproducibility/` directory.
The backed-up training `config.yaml` differs only in its actual timestamped name
from the submitted config. A read-only overlapping Slurm step on ws-l1-008
verified semantic equality excluding that override, exact submitted-config
hashes, matching formal initial-model hashes, and exactly one ACTIVE row and no
ARCHIVE row for each new Run ID. No smoke directories remain.

Full acceptance is deferred: evaluate best/current/best-val on complete test,
keep semantic/usage/color/leakage gates, and compare per-page rank with the
unweighted native codebook spectrum. A broader spectrum alone is not success.
