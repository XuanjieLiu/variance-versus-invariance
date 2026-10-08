# RQ1 — Content-width and native low-dimensional VQ representations

## Research question and success criteria

This direction now separates two questions. `RQ1-A` asks how narrow the entire
content embedding can be while the original K26 V3 still undergoes a semantic
phase transition. `RQ1-B` asks whether the atom interface exposed to a small
arithmetic model can be low-dimensional while the encoder/style pathways remain
wide.

The two dimensions must not be conflated. In RQ1-A, lowering `d_emb_c` also
lowers the native codebook dimension because no VQ projection is used. In RQ1-B,
`d_emb_c=512` and only `vq_codebook_dim` is compact.

## Current conclusion

End-to-end K26 has a clear seed-0 content-width boundary under the current V3
protocol: C128 transitions at epoch 160 and passes the full-test gate, whereas
C64 and C32 never transition. C64 retains healthy reconstruction and eventually
uses all 26 codes, making it the clean lower bound; the threshold lies in
`(64,128]`. C32 additionally develops severe train/eval normalization mismatch.
At K128 scratch, C32 still fails (best macro=.423), while native C128 reaches a
single full-test macro=.771 checkpoint before falling to .450 four epochs later.
This shows C128 capacity is feasible but does not yield stable flat-codebook
semantics from scratch. Since the application only requires a compact discrete
interface, the next design should keep `d_emb_c=512` and set
`vq_codebook_dim=128` rather than narrowing the entire encoder pathway.
For RQ1-B, PCA-warm K104/D4 and D8 both preserve near-perfect semantic purity,
healthy usage, reconstruction, and alias geometry at macro-best. However, the
registered 2x64 observed-alias regression probes failed badly in both dimensions
(test-style number accuracy .048/.091). Compact VQ is representationally viable;
purity and local alias separation alone do not make the current arithmetic
target learnable.

## Evidence runs

| Run | Subproblem | Evidence | Conclusion |
|---|---|---|---|
| VVI-H0-C4S4 | RQ1-A | test perplexity 12.14, 1:1=.266 | joint 4D reference fails |
| VVI-S1-C4S512-S0 | RQ1-A | active=21, perplexity=8.60, 1:1=.196 | 512D style does not rescue 4D content |
| VVI-S2-C512S512-CB4-S0-R1 | RQ1-B | active=26, perplexity=25.34, recon=.141, 1:1=.306 | VQ projection rescues usage/reconstruction |
| VVI-S2-C512S512-CB4-S0-R2 | RQ1-B | best test macro=.505, weighted=.476, coverage=20; current recon=.423 | longer training improves purity only modestly and later destabilizes |
| VVI-RQ3-K52-WARM-S0 | RQ1-B source | K52 best macro=.957, weighted=.966, active=52 | high-purity source for native-D projection |
| VVI-RQ13-K104-D4-PCAWARM-S0 | RQ1-B/RQ3 | best test macro=.9997, weighted=.9997, active=104, alias margin=.990; probe test=.048 | VVI succeeds; registered addition target fails |
| VVI-RQ13-K104-D8-PCAWARM-S0 | RQ1-B/RQ3 | best test macro=.9992, weighted=.9992, active=104, alias margin=.981; probe test=.091 | failure is not specific to D4 capacity |
| VVI-RQ1A-C128-K26-S0 | RQ1-A | onset=160; best test macro=.861, weighted=.848, active=26, perplexity=25.49, coverage=26, recon=.157 | C128 passes, although current later fluctuates below the formal purity gate |
| VVI-RQ1A-C64-K26-S0 | RQ1-A | no phase; best macro=.428, weighted=.318; current active=26/perplexity=24.92 but macro=.225 | clean low-purity lower bound |
| VVI-RQ1A-C32-K26-S0 | RQ1-A | no phase; best macro=.465; eval/train recon=896.9 | fails and is normalization-confounded |
| VVI-RQ1A-C64-K52-S0 | RQ1-A/RQ3 | best macro=.514, weighted=.374; current active=16/perplexity=4.36/recon=2.84 | more atoms do not rescue C64 cold start |
| VVI-RQ13-C32-K128-S0 | RQ1-A/RQ3 | best macro=.423, weighted=.403, active=128, perplexity=121.33 | redundancy does not rescue native C32 |
| VVI-RQ13-C128-K128-S0 | RQ1-A/RQ3 | epoch195 macro=.771/weighted=.768; current macro=.450 | 128D is feasible but the scratch semantic state is transient |

## Hypothesis register

| ID | Hypothesis | State | Evidence / test |
|---|---|---|---|
| RQ1-H1 | `d_emb_c=4` causes effective-capacity and usage collapse. | supported | H0 4/4 and S1 C4/S512 |
| RQ1-H2 | `vq_codebook_dim=4` preserves usage but enters a high-purity basin less readily. | supported | S2 R1/R2 |
| RQ1-H3 | A phase catalyst that succeeds in RQ2 will also raise CB4 purity. | blocked on catalyst | RQ2-S1 did not pass; wait for matched RQ2-H3 test |
| RQ1-H4 | CB4 needs a different projection/VQ optimization strategy rather than more epochs. | untested | compare after RQ1-H3 |
| RQ1-H5 | A high-purity K52 state can be compressed to a fixed native D4/D8 interface and expanded to K104 without losing semantics. | supported | both macro-best full-test checkpoints pass every VVI/alias gate |
| RQ1-H6 | Native D4/D8 atoms can support addition with the registered observed-alias 2x64 MLP objective and generalize to unseen styles. | not supported | 20k-step train/test number=.043/.048 for D4 and .095/.091 for D8 |
| RQ1-H7 | `d_emb_c=128` retains enough end-to-end capacity for the original K26 V3 phase transition. | supported for seed 0 | onset=160; macro-best passes every full-test gate |
| RQ1-H8 | `d_emb_c=32` retains enough end-to-end capacity for the original K26 V3 phase transition. | not supported | no stable window; low purity plus severe train/eval mismatch |
| RQ1-H9 | The C128/C32 phase boundary lies above C64 under the matched seed-0 protocol. | supported for seed 0 | C64/K26 has healthy reconstruction/usage but no semantic phase in 200 epochs |
| RQ1-H10 | C128 can retain a high-purity semantic phase when K grows from 26 to 128 from scratch. | partial, not stably supported | epoch195 full macro=.771/weighted=.768, but it is the only epoch>=.75 and current falls to .450 |
| RQ1-H11 | K128 redundancy can lower the native content-width threshold enough to rescue C32. | not supported | best full macro=.423 despite active=128/perplexity=121.33 |
| RQ1-H12 | A wide encoder with `vq_codebook_dim=128` can preserve a compact discrete interface and high purity through warm K52→K128 expansion. | planned | VVI-RQ13-S3; directly matches the downstream requirement |
| RQ1-H13 | A native-VQ effective-rank floor of 2–4 can prevent near-collinearity without degrading content purity or addition usability. | implementation verified; running222839/222840 from2026-10-08 | [v3_rank](../../manuals/v3_rank.md), VVI-HEX-RANK-S1: matched Hex16 permanent-MEAN rank4/rank2, weight.01. Geometry alone is not semantic success. |

## Tried, rejected, and backlog

- Rejected: simply increasing style dimension while leaving `d_emb_c=4`.
- Weak support only: extending CB4 from epoch 149 to 299; best macro rose to
  .505, then generalization/reconstruction degraded.
- Supported: label-free PCA-warm compression/expansion; macro-best D4/D8 retain
  all 104 active codes and near-perfect semantic metrics.
- Rejected as currently formulated: direct MSE+L1+commit regression to whichever
  alias the observed target image selected. Multiple aliases make the target
  conditional/multimodal even when all aliases are semantically pure.
- Supported: a C128 semantic phase exists, but its onset is about 20 epochs later
  than S3 C512 and the current checkpoint partially leaves the best state.
- Rejected: increasing K from 26 to 52 does not rescue a C64 scratch trajectory.
- Rejected: flat K128 scratch does not rescue C32; C128 produces only a one-epoch
  high-purity state and is not stable enough for downstream use.
- Backlog: arithmetic losses invariant to same-content alias choice, centroid or
  set-valued targets without using class labels as supervision, plus C96 and
  multi-seed boundary replication.

## Next decision gate

Latest user decision2026-10-08: run `v3_rank` on the successful Hex16 permanent
MEAN protocol, comparing target4/2 with nativeD512 and K16 unchanged. This is a
rank-regularization mechanism test, not yet a native-width reduction or an
addition experiment. Inspect both page-level and all-atom spectra, purity,
usage, recon and leakage before deciding whether to reduce native width.

Use the distinction established earlier: the downstream adder sees the native
VQ dimension, not the internal encoder width. The highest-probability next run
is therefore `d_emb_c=512`, `vq_codebook_dim=128`, K128, warm-expanded from the
stable K52 macro-best. If the entire content pathway must also be 128D, use the
C128/K26 epoch165 source and warm-split it rather than another scratch run.
C96/K26 remains the separate clean midpoint experiment for native pathway width.
