# Completed formal runs

Smoke runs are intentionally excluded. Metrics below use the common full
2,600-sample test protocol unless explicitly noted.

| Run ID | Completed | Setting | State | Run directory | Best checkpoint | Core metrics | Compared with | Conclusion |
|---|---|---|---|---|---|---|---|---|
| VVI-H0-C4S4 | 2026-07-28 | VVI-H0 | completed | `20260723-1249__v3_uppercase_c4_s4` | epoch 123 | total=.1755; recon=.1716; V3=.0039; active=26; perplexity=12.14; 1:1=.266; coverage=19; eval/train=1.24 | VVI-H0-C512S512 | Even its best-validation checkpoint lacks one-to-one alignment; by epoch 741 usage and validation also collapse. |
| VVI-H0-C512S512 | 2026-05-14 | VVI-H0 | completed | `uppercase_letters_run2_batch32_resume2` | epoch 472 | total/recon=.1548; V3=0; active=26; perplexity=25.60; 1:1=.779; coverage=25; eval/train=1.02 | VVI-H0-C4S4 | High-dimensional representations retain balanced code usage and substantially better one-to-one alignment. |
| VVI-S1-C4S512-S0 | 2026-07-30T19:36+04:00 | VVI-S1 | completed | `20260730-1447__VVI-S1-C4S512-S0` | epoch 18 | val=.1743; test total/recon=.1802; V3=0; active=21; perplexity=8.60; 1:1=.196; coverage=14; eval/train=1.05 | VVI-H0-C4S4 and VVI-H0-C512S512 | Restoring style to 512D does not rescue the codebook; the four-dimensional content/VQ path is the primary bottleneck, so this run will not be continued. |
