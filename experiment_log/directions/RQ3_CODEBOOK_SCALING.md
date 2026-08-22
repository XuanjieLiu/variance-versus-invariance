# RQ3 — Scaling the codebook beyond content count

## Research question and success criteria

Can a redundant codebook (`K > 26`) retain semantic purity while its native VQ
dimension D stays small enough for arithmetic? RQ3 now studies the joint `(K,D)`
frontier. Macro atom purity is primary and Hungarian accuracy is diagnostic only,
but purity alone is insufficient: aliases for a content must be geometrically
coherent and usable by a small downstream addition model.

For K=52, success requires active `>=47`, perplexity `>=40`, coverage=26,
macro and weighted purity `>=.75`, reconstruction `<=.30`, dominant-code counts
between 1 and 3, and count CV `<=.35`.

For K104/D4-D8, success requires active>=94, perplexity>=80, coverage=26,
macro and weighted purity>=.90, reconstruction<=.30, dominant counts 2–7,
CV<=.35, alias margin>=.90, and within/between ratio no more than 20% worse than
initialization.

## Current conclusion

S3 epoch 145 is a strong K=26 source. Before any update, the formal RQ3 run's
full 2,600-sample validation measured 52 active codes, macro=.8512, weighted
purity=.8785, perplexity=45.37, coverage=26, dominant-code min/max=1/3, and
CV=.310. At macro-best epoch 275, full test improved to macro=.9572,
usage-weighted purity=.9663, active=52, perplexity=48.06, coverage=26,
dominant-code min/max=1/3, CV=.310, and reconstruction=.1498. Current epoch 295
also passed every gate (macro=.9525, weighted=.9525, perplexity=44.50, CV=.277,
reconstruction=.1499). The warm expansion therefore both inherits and stably
maintains redundant semantics. A deterministic, label-free PCA transform of the
K52 macro-best to D4/D8 followed by pair splitting to K104 also passes all
initialization gates. After training, macro-best D4/D8 reached full-test
macro=.9997/.9992, weighted=.9997/.9992, all 104 codes active,
perplexity=98.00/98.61, alias margin=.990/.981, and recon=.1221/.1228. Thus
K104 is stable in both native dimensions at an appropriate checkpoint.
Epoch149 purity stayed high while alias margin degraded to .874/.817, showing
that alias geometry must remain a checkpoint/acceptance metric. The registered
2x64 addition probes nevertheless failed (D4/D8 test-style number=.048/.091),
so healthy aliases under these geometric metrics are not sufficient for the
current observed-alias regression objective. A C64/K52 scratch pilot provides
the complementary negative result: its best full-test macro=.514 but weighted
purity=.374 and perplexity=26.43, then epoch199 collapsed to 16 active codes,
perplexity=4.36, and reconstruction=2.84. Extra atoms therefore do not compensate
for a content pathway below the observed C64/C128 phase boundary. Because width
changed simultaneously, this pilot does not answer the planned C512/K52
cold-start mechanism control. The matched K128 scratch runs keep usage healthy
but expose the missing semantic constraint: C32 best full-test macro=.423, and
C128 reaches macro=.771 only at epoch195 before dropping to .450 by epoch199.
Only one validation epoch crosses .75, compared with K52-WARM staying above .75
for essentially its entire 150-epoch continuation. Warm inheritance, not
dead-code pressure, is the dominant difference.

## Evidence runs

| Run | Initialization | Evidence | Conclusion |
|---|---|---|---|
| VVI-S3-C512S512-CB512-S0 | K=26 from scratch | test macro=.834, weighted=.820 | accepted expansion source |
| VVI-RQ3-K52-WARM-S0 | pairwise warm split | best test macro=.957, weighted=.966, active=52, perplexity=48.06 | successful stable K=52 expansion |
| VVI-RQ13-K104-D4-PCAWARM-S0 | K52→PCA-D4→K104 | best test macro=.9997, weighted=.9997, active=104, perplexity=98.00, alias margin=.990 | VVI gate passed at epoch 28 |
| VVI-RQ13-K104-D8-PCAWARM-S0 | K52→PCA-D8→K104 | best test macro=.9992, weighted=.9992, active=104, perplexity=98.61, alias margin=.981 | VVI gate passed at epoch 38 |
| VVI-RQ1A-C64-K52-S0 | C64/K52 from scratch | best macro=.514, weighted=.374; current active=16, perplexity=4.36 | conditional negative: K52 does not rescue low-width C64 |
| VVI-RQ13-C32-K128-S0 | K128 scratch at native C32 | best macro=.423, weighted=.403, active=128, perplexity=121.33 | healthy usage without semantic purity |
| VVI-RQ13-C128-K128-S0 | K128 scratch at native C128 | best macro=.771, weighted=.768; current macro=.450 | transient feasibility, no stable semantic phase |

## Hypothesis register

| ID | Hypothesis | State | Evidence / test |
|---|---|---|---|
| RQ3-H1 | Splitting every learned atom locally into two inherits high purity. | supported | RQ3-S1 initialization and final full test |
| RQ3-H2 | K=52 from scratch can independently form a redundant semantic steady state. | still untested at C512; C64 pilot negative | full-width control remains required to isolate initialization |
| RQ3-H3 | Halving the dead-code threshold from 16 to 8 preserves comparable expected-usage pressure at K=52. | supported jointly with warm initialization | RQ3-S1 retained 52 active codes and high perplexity |
| RQ3-H4 | PCA-compressing K52 to native D4/D8 and splitting to K104 preserves a trainable redundant semantic state. | supported | both VVI-RQ13-S1 full-test macro-best checkpoints pass every gate |
| RQ3-H5 | High-purity aliases form geometry that the registered small observed-alias regression model can use across styles. | not supported | both 20k-step native probes fail train and test number-accuracy gates |
| RQ3-H6 | A K128 codebook can form healthy redundant semantics from scratch at native C128, and may rescue native C32. | not stably supported | C32 best macro=.423; C128 has one `.773` validation spike then decays to `.456` |
| RQ3-H7 | Generalized warm expansion from stable K52 to K128 can retain high purity with a 128D discrete interface. | planned | VVI-RQ13-S3 |

## Initialization and metrics

- Pair centers are `source ± delta`, with `||delta||` equal to 1% of the source
  median nearest-neighbor distance and a fixed random direction seed.
- Each source EMA cluster mass is divided equally; `embed_avg` is recomputed from
  the expanded embedding and cluster size.
- Record the complete per-content dominant-code count vector plus min/max/CV.
- In native VQ space, record same-content alias RMS distance, between-content
  centroid/nearest distance, within/between ratio, and the fraction whose nearest
  same-content alias is closer than the nearest different-content atom.

## Next decision gate

Preserve macro-best rather than late current: late training can retain
near-perfect purity while damaging local alias geometry. Before scaling to
K208, redesign the arithmetic target so it is invariant to which same-content
alias an image selected; the current MSE/L1 target is multimodal. The C64/K52
pilot shows that redundant capacity is not a substitute for sufficient content
width at K52. The K128 scratch pair now shows the complementary optimization
problem: C32 fails with healthy usage, while C128 briefly reaches macro=.771 but
cannot retain it. V3 is already zero and EMA usage is healthy, so loss-weight or
dead-code tuning is not the first intervention. Generalize the successful
K52-WARM/PCA-WARM recipe to K128/D128 and preserve initialization as macro-best.
