# S9 NoNorm acceptance — 2026-09-15

Both formal jobs finished epoch199 normally: Decoder-NoNorm192280 at
2026-09-14T18:09:27+04, All-NoNorm192281 at17:31:37+04, both exit0. Neither is
continued. All11 retained checkpoints in each run remain unchanged.

## Full-test results

All six checkpoint roles evaluated on2600 unchanged UppercaseLettersV2 test pages,
67,600 fragments,325 per(content,style). Normal eval with frozen VQ and the saved
decoder epoch: best34 uses page mean; best-val/current use fragment style.
Test manifest SHA256 `d455680108040b20d4880aff2bba29b4de0878f97cb0d37fd10ca91b8f1dc5b6`.

| Arm / checkpoint | Macro | Weighted | One-to-one | Active | Perplexity | Coverage | Recon | FG RGB MAE | FG chroma RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Decoder-NoNorm best34 |.75611|.67506|.67506|25|22.859|24|.04612|.18660|.15541|
| Decoder-NoNorm best-val188 |.53144|.51469|.49796|26|25.186|23|.00314|.03421|.02512|
| Decoder-NoNorm current199 |.51121|.51374|.49175|26|24.772|22|.00449|.05062|.04609|
| All-NoNorm best34 |.80491|.61516|.61516|25|18.622|24|.03480|.13249|.08475|
| All-NoNorm best-val195 |.33567|.32530|.29632|26|25.025|20|.00352|.03779|.02691|
| All-NoNorm current199 |.33830|.31749|.29590|26|25.327|20|.00366|.03903|.02767|

Six health JSONs with checkpoint/config/data hashes, six paired reconstruction
and foreground-mask grids, six balanced column-normalized confusion SVG/PNG/CSV/JSON,
Hungarian mappings, trajectory summaries and acceptance indexes are under each
source run's `acceptance_20260915_s9/`. Existing training artifacts are not overwritten.
GPU Job193694 saved all six health reports and five figure sets; the final figure
subprocess encountered an import while shared diagnostic code was being updated.
After the code was stable and84 regression tests passed, Job193704 reused cached
hash-verified results and completed the missing figure/index, exit0. This was an
acceptance-tool retry, not a formal training failure or rerun of all test forwards.

## Interpretation

Neither arm has a registered8-of-10 stable window. Best macro alone is misleading:
both best checkpoints have25 active atoms, and All-NoNorm additionally has lower
perplexity18.62. Neither passes the original full semantic gate.

Decoder-NoNorm validation1:1 peaks.67121 at34, then49→50 drops.65895→.58695,
reaching.29173 at74 and recovering only to.49638 at199. All-NoNorm peaks.61547
at34, already declines to.42277 before release at49, and finishes.29470.
Thus release is not a sufficient explanation of every deterioration.
All sampled optimization-scale summaries were finite; no NaN/Inf training crash.

Late reconstruction/color is substantially improved, but codes mix semantic
classes. Final active26 and perplexity24.77/25.33 distinguish this endpoint from
simple low-usage collapse. Relative to S8 BN current1:1.20112, Decoder-NoNorm
improves to.49175, but does not restore the early BN best.79728 or high purity.
Removing encoder BN as well yields poorer final semantics in this matched seed.

Final frozen-batch eval/train recon ratio is1.0909 for Decoder-NoNorm and exactly1
for All-NoNorm (the latter follows from having no mode-dependent normalization).
V3 hinge sum is0 at both current endpoints. Validation whitened z_s→content
readout is.97416/.92773; this is decodability, not an information-theoretic MI
estimate. The paired current grid also shows the same code representing multiple
correctly reconstructed glyphs, consistent with decoder use of continuous style
information to distinguish content. Norm removal alone does not force disentanglement.

Per-style full-validation shared-mapping accuracy at199:

| Arm | black | blue | green | red | teal | purple | orange | brown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Decoder-NoNorm |.528|.366|.528|.523|.498|.500|.516|.513|
| All-NoNorm |.513|.333|.499|.048|.360|.322|.054|.229|

All-NoNorm's red/orange mismatch despite high global usage reinforces that usage
health is not semantic alignment. Color metrics compare noisy observed inputs,
with estimated masks; they are not clean-reference or perceptual DeltaE scores.

## Next action

User prioritized a progressive hexadecimal extension of PhoneNums REPO and a
matched permanent-mean arm; see [HEX16 protocol](20260915_HEX16_PROTOCOL.md).
Return both backbones to REPO BN and lr.001; do not combine NoNorm with the data
change.100k pages restore the PhoneNums step budget; preserve its renderer rather
than uppercase fit-to-bbox. No new conclusion that class count alone caused failure.
