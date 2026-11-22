# IBP mel REPO pair submission, 2026-11-14

Protocol: [IBP-AUDIO-S2](../SETTINGS.md#ibp-audio-s2--mel-repo-fragment-versus-mean-style-pretraining).
Both fresh seed0,2000epochs/60000updates, source-only. No OOD fine-tuning.

| Arm | Job / node | Run | Config |
|---|---|---|---|
| Fragment style | 298789 / ws-l1-012 | `20261114-1706__VVI-IBP-INSNOTES-MEL-REPO-K12-S0` | `configs/insnotes/cfg_vvi_insnotes_mel_repo_k12_seed0.yaml` |
| Mean style | 298790 / ws-l5-020 | `20261114-1706__VVI-IBP-INSNOTES-MEL-REPO-MEAN-K12-S0` | `configs/insnotes/cfg_vvi_insnotes_mel_repo_mean_k12_seed0.yaml` |

Both submitted through the repository wrapper, ws-ia/1GPU/32GB/4CPU/24h.
Dedicated Python `~/.venvs/v3_audio/bin/python`. Source/config/data/package
provenance archived inside each ignored run. Training entered epoch0 at17:07,
with finite losses and expected fragment/page_mean decoder modes.

Config SHA256, fragment:
`83c43704ae1c5f8339b7f091f0b3beb30b1247ecdd59177935cc8887d0027a02`.
Config SHA256, mean:
`d4f4c351690f89af2ca06444a2022a168ae70ab6abb74c73143b365b99a37941`.
Shared initial model-state SHA256:
`f40dee859df33eb9dbfb7d3c74c67cdda464fe59c8b515ac91d5c67b3f19c944`
(the formal trainer's sorted, dtype/shape-aware hash in initial_model_state.json).
The smoke helper uses a different byte-hash convention; its two identical
hashes are `89d35059fc41503dcf12487c949d008340d0f718eacf20e63a42c96ea944f35c`.

Preflight298785 onws-l5-021 completed exit0:162unit tests, each config60CUDA
updates, complete14400-note validation, all1200data hashes, real synthesis
reproduction, encoder/decoder nonzero gradients, finite objectives, independent
loss-selected checkpoint rotation, strict loads and resume optimizer/scheduler.
Each arm exports27WAVs during the two-epoch smoke (initial/final/best roles);
all finite/non-silent/no clipping. Re-exporting best is byte-identical and leaves
the trainer model and global RNG unchanged. Temporary smoke directories deleted.
Lightweight records remain in `logs/insnotes_mel_preflight_298785_cfg_*.json`.

Regular outputs: loss/code/ratio curves; full-validation diagnostics/mel grids
every50epochs; fixed audio every200epochs plus0/final1999; best-validation
audio after completion. Portable listening entry: `audio_examples/index.html`.
Samples include C/E/G/B in Soprano Sax/Pipe Organ/Trumpet. Compare original,
mel roundtrip and reconstruction, not original/reconstruction alone.

Retain current, health-gated macro-best and independent best validation-total
loss. Loss-selected retention must not depend on the macro eligibility gate.
Evaluate source health before any target experiment; no automatic continuation.
