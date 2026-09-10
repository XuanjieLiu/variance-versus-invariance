# Active formal runs

Smoke runs are intentionally excluded.

| Run ID | Started (Asia/Dubai) | Slurm Job | Direction | Hypothesis | Setting | Status | Run name | Key change | Purpose / comparison |
|---|---|---|---|---|---|---|---|---|---|
| VVI-RQ2-C128-K26-CTRL-S0 | 2026-09-10T13:48:19+04:00 | 183134 | RQ2; RQ1-A control | RQ2-H8 | VVI-RQ2-S3 | running;ws-l1-001 | `20260910-1348__VVI-RQ2-C128-K26-CTRL-S0` | original fragment decoder;200ep;25ep snapshots | Reproduce original C128/K26; concurrent control for MEAN/W100. |
| VVI-RQ2-C128-K26-MEAN-S0 | submitted2026-09-10T13:48:19+04:00;not started | 183135 | RQ2 | RQ2-H8 | VVI-RQ2-S3 | pending;QOSMaxJobsPerUserLimit | `20260910-1348__VVI-RQ2-C128-K26-MEAN-S0` | page-mean decoder all200ep;raw style in V3 | Test removal of fragment-specific shortcut versus CTRL; auto-starts when a slot frees. |
| VVI-RQ2-C128-K26-W100-S0 | 2026-09-10T13:48:16+04:00 | 183133 | RQ2 | RQ2-H9 | VVI-RQ2-S3 | running;ws-l1-011 | `20260910-1348__VVI-RQ2-C128-K26-W100-S0` | page mean0–99;fragment100–199;no optimizer reset | Test persistence/color after release versus CTRL/MEAN. |
