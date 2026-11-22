#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Run this setup inside a SLURM allocation, never on the login node." >&2
    exit 2
fi
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate xuanjie
audio_python="$HOME/.venvs/v3_audio/bin/python"
audio_tools="$HOME/miniconda3/envs/v3_audio_tools"
if [[ ! -x "$audio_python" ]]; then
    python -m venv --system-site-packages "$HOME/.venvs/v3_audio"
fi
"$audio_python" -m pip install 'librosa==0.10.2.post1' 'soundfile==0.12.1' 'pretty_midi==0.2.10'
if [[ ! -x "$audio_tools/bin/fluidsynth" ]]; then
    conda create -y -p "$audio_tools" --override-channels -c conda-forge 'fluidsynth=2.3.7'
fi
"$audio_python" -c 'import torch, librosa, pretty_midi, soundfile; assert torch.cuda.is_available(); print("CUDA", torch.cuda.get_device_name(0), "librosa", librosa.__version__)'
"$audio_tools/bin/fluidsynth" --version
