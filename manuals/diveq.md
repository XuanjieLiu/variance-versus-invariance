# DiVeQ backend

The historical/default V3 quantizer is `vector_quantize_pytorch.VectorQuantize`
with EMA updates, k-means initialization and STE (rotation trick disabled).
An absent `model_config.quantizer` still means `ema`; old checkpoints/configs
are not reinterpreted or converted. The new experiments train from scratch.

## Protocol and implementation

`model/diveq.py` adapts ordinary stochastic **DiVeQ**, not SF-DiVeQ or
DiVeQDetach. Primary references: [paper, Eq.8](https://arxiv.org/html/2509.26469v1)
and [official implementation](https://github.com/AaltoML/DiVeQ/blob/7e8cc2c3885707576510273da097465949f04063/vqvae_compression/diveq.py),
pinned commit `7e8cc2c3885707576510273da097465949f04063`.
The MIT notice is retained in `licenses/DiVeQ.txt` and run source snapshots.
This local adapter avoids upgrading the Python3.10 training environment to meet
the upstream package's Python>=3.11 requirement.

For encoder vector z, select nearest trainable atom e, set d=e-z and sample
epsilon~Normal(0,noise_std). Training returns
`z + ||d|| * stop_gradient((d+epsilon)/max(||d+epsilon||,1e-12))`.
Both encoder and codebook receive task gradients through the distance term;
this is not an identity STE. Evaluation returns the exact nearest atom, without
noise. No EMA, auxiliary codebook loss or commitment loss is used. Original V3
equations, reconstruction loss and raw fragment-style supervision are unchanged.

```yaml
model_config:
  quantizer: diveq
  d_emb_c: 512
  d_emb_s: 512
  n_atoms: 16
  decoder_style_mode: fragment
  diveq_config:
    noise_std: 0.001
    replacement_iters: 100
    discard_threshold: 0.01
    perturb_eps: 1.0e-9
loss_config:
  weights:
    commit_loss: 0
```

Keep the other required model/loss fields from a complete config. Remove EMA
options; they are rejected rather than silently ignored. This adapter currently
requires native VQ dimension equal to `d_emb_c` (identity project_in/project_out).
It supports existing native-code lookup/capture and checkpoint geometry tools.
`vq._codebook.embed` remains shaped `[1,K,D]`, but is now a Parameter, included
in the existing AdamW group, including its weight decay. Do not load an EMA
checkpoint as DiVeQ. Strict checkpoint loading protects against missing state.

## Defaults and declared engineering differences

- Official uniform initialization is retained: atom components in `[0,1/K)`.
  This is different from historical EMA k-means initialization.
- Upstream calls the noise argument `noise_var`, but supplies it as Normal's
  scale (standard deviation). Here the name explicitly says `noise_std`.
- Replacement uses fraction of training **batches containing a code**, not
  fraction of tokens. Every100 forwards, codes present in fewer than1% of batches
  are copied from donors sampled proportional to batch presence, plus1e-9 jitter.
  This makes the upstream repeated-index presence counting explicit.
- Replacement is deferred until after backward and a successful optimizer step,
  so a parameter is not mutated while its autograd graph is alive. Upstream does
  replacement inside forward. Donors use post-step values here. Adam moments
  are retained, as upstream does; this is an adapter, not bitwise reproduction.
- Noise is sampled on the input device. Its distribution matches the reference,
  but RNG streams differ from the reference CPU sampling implementation.
- Inference/no-grad diagnostics use exact atoms even if a caller temporarily
  sets model.train(); they never update usage, sample noise or replace atoms.
  Explicit gradient-enabled eval retains directional surrogate gradients with
  exact hard-quantized forward values; freeze_codebook suppresses atom gradients
  and usage/replacement updates but not encoder gradients.
- Usage counts, replacement phase, atoms and Adam state survive resume. Every
  run writes `quantizer_config.json`; replacement events appear in the train log.

Consequently the EMA comparison changes the **quantizer bundle**, including
gradient estimator, atom update, initialization, replacement and auxiliary loss.
It cannot isolate the causal effect of EMA alone. Train reconstruction sees
slightly perturbed DiVeQ outputs; formal eval/reconstruction sees discrete atoms.

## Hex16 experiments, 2026-10-10

Setting `VVI-HEX-DIVEQ-S1`: ordinary fragment decoder throughout, no MEAN,
warmup, v3_rank, addition training or checkpoint warm start. Two configs in
`configs/hex/v2/`: `cfg_vvi_hex_k16_diveq_c512_s512_seed0.yaml` and
`cfg_vvi_hex_k16_diveq_c4_s4_seed0.yaml`.
Both retain the original Hex REPO dataset, BN, optimizer, scheduler and V3
weights except commitment=0. Seed0, batch32,200 epochs:2500 steps/epoch,
500000 optimizer updates. C512/S512 versus C4/S4 also changes style capacity;
a small-model failure would not prove that native content4D is impossible.
Different widths consume different initialization RNG: record both model/order
hashes, but do not claim identical initialization or train sample order.

All compute must run on a ws-* GPU node. Preflight script
`scripts/smoke_hex_diveq.py` runs100 training steps per arm, complete10000-page
validation, checkpoint/optimizer reload and512-page quick evaluation; it tests
diagnostic state/RNG neutrality and deletes temporary smoke runs immediately.
Formal submission uses `VVI_TIME=24:00:00 scripts/submit_slurm.sh` after ACTIVE
registration. Retain current, health-gated macro-best, best-val and25-epoch
snapshots; at most11 checkpoints at200 epochs. Recon and balanced confusion
remain synchronized, including the extra early diagnostic nodes.

Full-test acceptance uses10000 pages at macro-best/best-val/current: macro,
weighted purity and Hungarian>=.75; active16, perplexity>=13, coverage>=15,
recon<=.30. Stability requires8/10 consecutive epochs with macro/Hungarian>=.70
and those usage floors. Report color, leakage and native-code spectrum separately.
High purity does not prove multidimensional geometry or arithmetic usability.
