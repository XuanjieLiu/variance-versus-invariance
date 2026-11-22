# RD-V3 HEX source pair submission

Registered/submitted2026-11-20 (Asia/Dubai), setting IBI-RD-HEX-S1. User
selected image HEX and deferred both recon-only source pretraining and OOD
fine-tuning. [Manual/config reference](../../manuals/representation_distillation.md)
owns data/loss flow. [V5 RQ6](../../../v5/research/directions/RQ6_INDUCTIVE_BIAS_PERSISTENCE.md)
owns the IBI hypothesis and interpretation.

| Arm | Job | Run | Config |
|---|---|---|---|
| LR1e-4 | 310475 | `20261120-0935__VVI-IBI-RD-HEX-RANK2-LR1E4-S0` | `configs/hex/rd/cfg_rd_v3_hex_rank2_lr1e4_seed0.yaml` |
| LR3e-4 | 310476 | `20261120-0935__VVI-IBI-RD-HEX-RANK2-LR3E4-S0` | `configs/hex/rd/cfg_rd_v3_hex_rank2_lr3e4_seed0.yaml` |

Config SHA256 respectively:
`59987b117b78ec149e00bda9c61c878ec30babcda9f4210757386e82f51b2fe2`,
`22a56ca168db5e38067e4ee832364ec40177606b612f7338e632ad33de5eebb2`.

Both: AdamW WD.1,seed0,32pages/512images,200epochs/500kupdates, recon1,
commit.1, contentNMSE1,styleNMSE1, no V3/rank optimization. Exponential
.98/20epochs withfloorfactor.02. Same teacher epoch59, data, initialization,
calibration and code version; only LR differs. Independently learned image-EMA
student codebook, not a copied teacher codebook. Same-style source grouping and
mean-style decoding remain; this does not test removal of data organization.

Resources: standard `scripts/submit_slurm.sh`,ws-ia,1GPU,32GB,4CPU,24h each.
Interpreter `/home/xuanjie.liu/.venvs/v3_audio/bin/python`; despite its path,
these jobs are image-only. Standard source tar/hash/config snapshots live in
each run's `reproducibility/`. Logs in `logs/<run>/log.txt` and
`slurm_logs/vvi-ibi-rd-hex-rank2-lr{1,3}e4-s0-<job>.{out,err}`.

Both started09:40:41+04 on RTX5000Ada, nodesws-l1-009/ws-l3-020. Initial
training updates are finite with nonzero encoder/decoder gradients. Their
entire captured source SHA256 manifests match, as do initial model states and
the actual2048-page teacher calibration:Vc=7.185284132819946,
Vs=2.168317481931993; page-index hash
`f767389e78b7010578a53d7922de0474333eb9774e7aac72cf0bf9d71042b848`.

## Preflight and teacher verification

- GPU310464: all6new objective/config tests passed; full source teacher
  validation10000pages/160000fragments,1:1/purity.99998125,ppl~16,16active,
  recon.0432777595. All model state unchanged. Source reconstruction and
  leakage/color/BN diagnostics at `logs/ibi_rd_hex_preflight_20261120/teacher_reference/`.
- Both real-image smokes:10freshupdates +10resumedupdates each; exact
  validation loss reproduction from saved target calibration; no teacher
  gradient/BN/VQ mutation, student encoder/decoder gradients nonzero, no
  shared parameter objects, both LR arms share initial state SHA256
  `00d3ef1c717de57f8b412a9ce89603b7a2af5c95b36629f22c44a43de8b6444d`.
- Smoke calibration uses64train pages only to shorten the check; formal
  calibration uses2048. No scientific conclusion is drawn from smoke metrics.
  Temporary checkpoint/plot directories were deleted automatically.
- Full regression310470 exposed a checkpoint-helper compatibility issue in
  mock trainers without a loss object. Fixed the optional metadata lookup;
  rerun310473 passed all172tests. Static compile passed on the compute node.
- No Python or experiments were executed on login.
- V5 research-link validation on the compute allocation passed22Markdown files
  and the150-line dashboard limit. Both repositories pass `git diff --check`.

## Acceptance contract

Validate complete source-val each epoch, retain latest/current, eligible
macro-best and independent minimum-RD-validation-objective checkpoint.
Primary selection is the latter; do not compare its total objective numerically
with original V3 total loss. Audit common recon/Hungarian/purity/per-style/
usage/leakage/color metrics plus raw and normalized distillation errors.
Inspect best AND latest before deciding whether the student is adequately
matched. Symbol-readiness >=.99,16active,ppl>=15,recon<=1.25xteacher and
leakage screening as registered. Label-based gates do not backpropagate.

The teacher has near-perfect source symbols but imperfect color rendering;
the archived source reference preserves that limitation. A student success
does not prove IBI, and a poorly matched student failure would not refute it.
No OOD job or third pretraining baseline is automatically chained.
