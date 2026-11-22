#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Use srun/sbatch for audio preparation." >&2
    exit 2
fi
cd /home/xuanjie.liu/Projects/variance-versus-invariance
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export MPLBACKEND=Agg SDL_AUDIODRIVER=dummy
PYTHON="$HOME/.venvs/v3_audio/bin/python"
"$PYTHON" -m unittest discover -s tests -p test_insnotes_mel_generation.py -v
"$PYTHON" -m dataset.insnotes.generate_mel_dataset \
    --output "${1:-../data/InsNotesMelV1}" --workers "${SLURM_CPUS_PER_TASK:-4}"
