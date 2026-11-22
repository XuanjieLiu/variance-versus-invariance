#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Use a SLURM GPU node." >&2
    exit 2
fi
cd /home/xuanjie.liu/Projects/variance-versus-invariance
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export MPLBACKEND=Agg SDL_AUDIODRIVER=dummy
PYTHON="$HOME/.venvs/v3_audio/bin/python"
"$PYTHON" -m compileall -q trainer.py dataset/insnotes dataloader/insnotes_mel_dataloader.py utils/audio_diagnostics.py scripts/smoke_insnotes_mel.py
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" scripts/smoke_insnotes_mel.py
