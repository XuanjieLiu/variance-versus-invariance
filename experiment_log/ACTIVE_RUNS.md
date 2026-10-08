# Active formal runs

Smoke runs are intentionally excluded.

| Run ID | Started (Asia/Dubai) | Slurm Job | Direction | Hypothesis | Setting | Status | Run name | Key change | Purpose / comparison |
|---|---|---|---|---|---|---|---|---|---|
| VVI-HEX-K16-MEAN-RANK4-S0 | 2026-10-08 16:24:54 | 222839 | RQ1-B | RQ1-H13 | VVI-HEX-RANK-S1 | running ws-l1-008;24h | 20261008-1623__VVI-HEX-K16-MEAN-RANK4-S0 | v3_rank target4,weight.01; K16/D512; permanent MEAN; scratch200epochs | Test non-collinearity while retaining Hex MEAN purity; compare rank2 and historical HEX MEAN |
| VVI-HEX-K16-MEAN-RANK2-S0 | 2026-10-08 16:24:57 | 222840 | RQ1-B | RQ1-H13 | VVI-HEX-RANK-S1 | running ws-l1-014;24h | 20261008-1623__VVI-HEX-K16-MEAN-RANK2-S0 | v3_rank target2,weight.01; all other settings matched | Minimal rank floor versus rank4 and historical HEX MEAN |
| VVI-RQ2-V2-C512-K26-HEXREPO-S0-R1 | 2026-09-17 13:20:17 | 197821 | RQ2 | RQ2-H23; H25a | VVI-RQ2-S10 | running ws-l1-002;24h | 20260917-1319__VVI-RQ2-V2-C512-K26-HEXREPO-S0-R1 | Current199→769;570 additional epochs; fragment; preserve optimizer/LR | 500500 cumulative steps vs parent130k and Hex500k; not LR-by-step matched |
| VVI-RQ2-V2-C512-K26-HEXREPO-MEAN-S0-R1 | 2026-09-17 13:20:18 | 197822 | RQ2 | RQ2-H24; H25a | VVI-RQ2-S10 | running ws-l1-004;24h | 20260917-1319__VVI-RQ2-V2-C512-K26-HEXREPO-MEAN-S0-R1 | Current199→769; permanent page mean; all other settings retained | Matched long-budget ordinary arm and Hex MEAN; no release |

S10 originals completed199/exit0 and are archived2026-09-17 with explicitly
validation-only interim metrics; full-test acceptance is deferred to the extended
budget. R1 resumes latest current, not macro-best. See
[500500-step continuation](analyses/20260917_S10_CONTINUATION.md).

HEX-S1 completed199/exit0 and was full-test accepted/archived2026-09-16; both
best/current states pass the semantic gate, permanent MEAN has better current
color. No Hex continuation. See [Hex acceptance and uppercase comparison](analyses/20260916_HEX_ACCEPTANCE_UPPERCASE.md).
S9 completed199/exit0 and
was accepted/archived on2026-09-15; neither is resumed. See
[NoNorm full-test acceptance](analyses/20260915_S9_ACCEPTANCE.md).
S8 BN/GN8 completed199 with exit0 and were accepted
and archived on2026-09-14. BN has transient high aggregate purity, blue-style
collapse and failed persistence/color; GN fails from the first epoch.
See [S8 full-test acceptance](analyses/20260914_S8_ACCEPTANCE.md).
The proposed matched snapshot49 continuation pair is not submitted; neither
current199 checkpoint has been resumed.
S7 W0/W50 both completed199 with exit0 and were accepted/archived on2026-09-13
evening; neither passes the content gate. They are not resumed.

Old-data CTRL/W100 R1 were accepted on2026-09-11; V2 CTRL/W100 were accepted
on2026-09-12 and also moved to ARCHIVE. Neither V2 run passes the content gate;
do not resume them. S5/S6 use manifest SHA256
`d455680108040b20d4880aff2bba29b4de0878f97cb0d37fd10ca91b8f1dc5b6`.
Neither cancelled MEAN nor C512 is resubmitted.

Submission/preflight provenance and config hashes:
[2026-09-12 acceptance note](analyses/20260912_V2_ACCEPTANCE_AND_PHONENUMS.md).
PN PAPER/REPO and C128-W50 were accepted and archived on2026-09-13. PAPER did
start on ws-l4-011 and completed199; W50 timed out during198, retaining197.
No continuation of these three runs. S7 C512 is a new V2 experiment, not a
resubmission of the cancelled historical S4 C512 run. See the
[S7 acceptance/protocol note](analyses/20260913_PN_W50_ACCEPTANCE_C512.md).
