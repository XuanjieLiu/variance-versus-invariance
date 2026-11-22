# InsNotes Mel V3: IBP audio reproduction

Registered2026-11-06. Cross-project hypotheses and supervised-control proposal
are canonical in [V5 RQ6](../../v5/research/directions/RQ6_INDUCTIVE_BIAS_PERSISTENCE.md).
This manual owns audio generation, config semantics and execution.

## Scope and audit

Only source V3 pretraining is launched now. New-style FT follows source
acceptance; new content and supervised pretraining are not yet launched.
This is a paper-objective, mel-input reproduction, NOT a bitwise STFT replica.

Legacy files are retained for historical reproducibility. New entry points:

- `dataset/insnotes/generate_mel_dataset.py`: seeded immutable source generation.
- `dataloader/insnotes_mel_dataloader.py`: no augmentation; random cyclic source
  groups in training; deterministic all-note validation/test with interleaved styles.
- `utils/audio_diagnostics.py`: recording-disjoint readouts, mel reconstructions
  and current Hungarian confusion; no image/color-specific diagnostics.
- `configs/insnotes/cfg_vvi_insnotes_mel_k12_seed0.yaml`: formal source config.

The legacy renderer referenced missing `FluidR3_GM.sf2` although GeneralUser GS
is bundled. Its envelope used120 seconds while notes span126.72seconds, padded
centrally instead of respecting note onsets; min/max bounds also allowed values
outside.8..1.2. The new generator uses the bundled, hashed soundfont and independent
bounded per-note ramps. It uses dry synthesis (no reverb/chorus) to avoid effect
tails across notes; that is a declared change, not claimed identical rendering.
The legacy loader randomly crops/masks validation and drops a frequency bin;
the new mel loader has neither problem and never drops a mel band.

## Data contract

`../data/InsNotesMelV1` contains1200 source recordings:12styles x100takes.
Each take contains120 notes (12pitches x10repetitions). Source train/val/test
are80/10/10takes per style, i.e.960/120/120recordings. Each pitch x style has
800/100/100notes respectively. Split complete recordings before fragmenting.
Random velocity80..120 and envelopes distinguish repetitions; deterministic
software synthesis is not a claim of independent real-world recordings.
A training epoch visits each take once but samples only one12-note group per
take (11520notes/epoch,30updates), not all120notes. Evaluation visits all groups.

Features:16kHz, FFT1024, hop512, Hann window,128Slaney mel bands, magnitude
power1, center=True, fmin0/fmax8000, natural log(mel+1e-5). No per-example feature
normalization. Waveforms use the historical per-recording peak normalization.9.
Each file `[128,3960]` contains120slots of33frames; each note uses32frames.
Model batch `[32,12,128,32]`. Mel is a filterbank over an STFT internally; the
model receives mel bands, not the original linear-frequency STFT bins.

Manifest includes source GMprogram IDs, reserved OOD IDs[0,4,73,78], seed,
versions, soundfont/generator/file hashes, velocities and counts. Refuse any
overwrite. `status=complete` is written only after generation/duplicate checks.
OOD content is absent. Audio examples are12source-only shortWAVs; bulkWAV/MIDI
intermediates are temporary, so the dataset stays around2.3GiB.

## Config reference

The existing VVI trainer remains the training entry; no separate training loop.
Unchanged generic keys are described by their inline config names and existing
VVI manuals. Audio-specific settings and deliberate differences:

| Setting | Value / meaning |
|---|---|
| dataloader | `insnotes_mel_dataloader`; validates manifest and exact feature shape |
| data_dir / dataset_manifest_sha256 | immutable source root / pinned manifest digest before launch |
| model_config.n_feature / fragment_len | 128mel bands /32time frames |
| model_config.n_fragments | 12different pitches from one instrument per source sample |
| model_config.n_channels / d_emb_c / d_emb_s / n_atoms | 512 /512 /512 /12; original audio backbone |
| model_config.decoder_style_mode | fragment; no mean-style intervention |
| model_config.vq_ema_decay / threshold_ema_dead_code | paper.95 /384/(10*12)=3.2; original YAML used.98 /16 |
| loss_config.weights | recon1; content/style/sample/fragment each1; commit.01 (paper, original YAML.1) |
| loss_config.relativity / monitor_raw_mpd | 15 /true; monitor raw variability without changing objective |
| optimizer_config.optimizer / lr / weight_decay | paperAdam /.001 /0, not original YAMLAdamW /.1WD |
| optimizer_config.lr_decay_* | repo exponential.98 per20epochs, minimumfactor.02 |
| batch_size / num_workers / precision | 32 /4 /float32; preserve optimizer numerical mode |
| epochs | initial2000natural epochs =60000updates; not a fixed claim of convergence |
| val/plot/save_every_n_epochs | 10 plus forced final; entire balanced validation, no test selection |
| audio_diagnostics.enabled | enable audio-specific validation monitoring, default off for older configs |
| audio_diagnostics.every_n_epochs | 50; evaluate at epoch0, each50epochs and final |
| audio_diagnostics.pca_components / ridge | 32 /.01 for deterministic recording-disjoint diagnostic readouts |
| audio_diagnostics.plot_content_ids | optional distinct pitch IDs (default0,3,6,9), at most4 displayed notes per instrument; all12 are archived |
| audio_diagnostics.final_epoch | trainer-derived absolute final epoch, not a user scheduling override |
| save_best_macro_atom_purity / checkpoint_policy | latest + health-gated macro-best, no top3/snapshots |
| macro_best_health_gate | all12active, ppl>=10, at least10dominant labels; scientific gate is stricter |
| training_usage_monitor | online atom counts every100updates, never affects model |
| VVI_PYTHON (environment) | optional Python executable in SLURM wrapper; default `python`, old jobs unchanged |

Read source supervision into checkpoint selection honestly: VVI macro-best uses
validation metadata. The paper chose validation loss. Neither uses test labels;
source gate and later FT scientific selection must be explicitly registered.
Final evaluation/plots/checkpoints are now also emitted when the last epoch is
not divisible by the requested interval.
Validation losses/ratios are sample-weighted means of batch quantities, not
unweighted means of batches; a final partial batch no longer receives excess
weight. V3 ratios remain batch statistics, not global-dataset MPD estimates.
Resume restores model/optimizer/scheduler/scaler but not sampler/worker RNG;
therefore resume is supported but not claimed bitwise-identical.

## Commands (compute nodes only)

```bash
srun -p ws-ia --gres=gpu:1 --mem=32G -c4 -t00:30:00 \
  bash scripts/setup_insnotes_audio_env.sh
srun -p ws-ia --gres=gpu:1 --mem=32G -c4 -t02:00:00 \
  bash scripts/prepare_insnotes_mel.sh
srun -p ws-ia --gres=gpu:1 --mem=32G -c4 -t00:30:00 \
  bash scripts/smoke_insnotes_mel.sh
VVI_PYTHON="$HOME/.venvs/v3_audio/bin/python" VVI_TIME=24:00:00 \
  bash scripts/submit_slurm.sh configs/insnotes/cfg_vvi_insnotes_mel_k12_seed0.yaml \
  YYYYMMDD-HHMM__VVI-IBP-INSNOTES-MEL-K12-S0 VVI-IBP-AUDIO-MEL-K12-S0
```

Environment setup uses a dedicated venv with xuanjie's existing torch packages,
plus pinned librosa.10.2.post1/soundfile.12.1/pretty_midi.2.10. FluidSynth2.3.7
lives in a separate conda tools prefix. The xuanjie environment is not modified.

Before submission, run data/unit/full-epoch CUDA smoke, verify finite losses,
nonzero encoder/decoder gradients, frozen BN/EMA during eval, balanced full
validation counts, deterministic repeated metrics, and real checkpoint restore.
Temporary smoke artifacts are removed; retain only a lightweight verification
record under ignored logs. Record formal job ID in ACTIVE before handoff.

## Read-only acceptance and listening

[2026-11-14 acceptance](../experiment_log/analyses/20261114_INSNOTES_MEL_SOURCE.md):
the first source run failed its validation gate; no target FT launched.
`scripts/accept_insnotes_mel.py --run <run-directory> --output <new-directory>`
runs under a SLURM GPU allocation. It checks all14400 validation notes,
reversible encoder/decoder BN interventions, a mean-style decoding probe, exact
model-state preservation and deterministic re-evaluation. Test stays sealed.
The output directory must not exist, so historical acceptance is never replaced.

The listening table `audio.html` is self-contained with its adjacent `audio/`.
`utils/audio_playback.py` inverts magnitude log-mel with a fixed Griffin-Lim
seed, not a learned vocoder. Always include input-mel roundtrip alongside
original and model output. `render_take(..., save_preview=True)` optionally
exports any recording's first12 notes; the default generator and features are
unchanged. Re-rendered reference features must match the original SHA256.

## REPO pair and training-time listening (IBP-AUDIO-S2)

Registered2026-11-14: [matched protocol](../experiment_log/SETTINGS.md#ibp-audio-s2--mel-repo-fragment-versus-mean-style-pretraining).
The two `cfg_vvi_insnotes_mel_repo*_k12_seed0.yaml` configs restore the original
repository optimizer/VQ bundle and differ only in decoder style mode and name.
The existing `save_best_validation_loss: true` retains one best validation-total
loss checkpoint even if macro health gating rejects every candidate. With
current/macro-best, there are at most3checkpoint files. Audio uses the actual
loss-selected state, not the last model with a misleading best label.

New nested config keys under `audio_diagnostics.playback` (old configs default
to disabled):

| Key | Default | Meaning |
|---|---|---|
| enabled | false | Export fixed validation audio; requires audio diagnostics |
| every_n_epochs | 200 | Zero-based epoch multiples plus0 andfinal; must be a multiple of diagnostic interval |
| griffin_lim_iterations | 64 | Phase-recovery iterations; magnitude power stays1 |
| seed | 20261114 | Local inversion seed plus pitch ID, no global RNG draws |
| style_ids | [0,1,4] | Distinct instrument IDs, selected before observing recon quality |
| pitch_ids | [0,4,7] | Distinct pitch IDs; new pair explicitly uses [0,4,7,11], C/E/G/B |
| export_best_validation | true | Export loss-selected checkpoint after training; requires save_best_validation_loss |

Open `logs/<run>/audio_examples/index.html`, with per-epoch directories and
`best_validation/`. Each directory contains portable HTML,9WAVs (3instruments
x3routes), manifest and raw mel arrays. Models are never updated for exports.
Periodic examples consume detached existing validation outputs; final best
export uses a separate strict-loaded model and preserves global RNG. Originals
are re-rendered once per process and cached after feature-hash verification.
Do not compare perceptual audio quality without the input-mel roundtrip control.

Pair acceptance: `scripts/accept_insnotes_repo_pair.py --fragment <run-dir>
--mean <run-dir> --output <new-directory>`, under a SLURM GPU allocation.
It reloads loss-best, macro-best and current for both runs, reproduces full-val
metrics, adds per-instrument global-mapping diagnostics and gain-invariant
spectral-shape error, and probes loss-selected states with unchanged BN/VQ
buffers. Within-instrument Hungarian results are supplementary diagnostics,
never substituted for global content accuracy. `index.html` plus adjacent
`audio/` is an offline side-by-side listening bundle. See
[2026-11-19 results](../experiment_log/analyses/20261119_INSNOTES_REPO_ACCEPTANCE.md).

## Extending the source budget

[IBP-AUDIO-S2-R1](../experiment_log/SETTINGS.md#ibp-audio-s2-r1--source-budget-continuation-to10000-epochs)
continues current1999 to9999 in separate run directories. Use the two
`cfg_vvi_insnotes_mel_repo*_k12_seed0_r1_10000e.yaml` configs: `epochs: 8000`
means additional epochs, not the absolute endpoint. `inherit_resume_history`
now also copies `audio_diagnostic_history.csv`, checking the same audio
manifest, recording partitions, grid and probe settings. Historical WAVs
remain in the parent; `resume_lineage.json` links that directory. Restore both
best criteria and their actual weights; never seed a best score with latest
weights. RNG is not stored in these legacy checkpoints, so record the seed0
restart rather than claiming bitwise uninterrupted equivalence.
