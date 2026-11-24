#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run this script as a Slurm job.}"

repo="$HOME/Projects/variance-versus-invariance"
run_name=
while (( $# )); do
    case "$1" in
        --name) run_name=$2 ;;
        --config) [[ $2 == cfg_v3_insnotes.yaml ]] ;;
        *) exit 2 ;;
    esac
    shift 2
done
[[ -n "$run_name" && "$run_name" != */* ]]
run="$repo/logs/$run_name"
capsule="$run/reproduction"
data="$repo/../data/InsNotesUpstreamSTFTV1"
upstream="$capsule/upstream"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate xuanjie
py="$capsule/venv/bin/python"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLBACKEND=Agg
test -s "$capsule/dataset_manifest.json"
cmp "$capsule/dataset_manifest.json" "$data/manifest.json"
test "$(git -C "$upstream" rev-parse HEAD)" = bbbc7c9c26bda7170612814a407b76633cd22875
git -C "$upstream" diff --exit-code
cd "$upstream"

if [[ ! -f "$run/validation_selection.csv" ]]; then
    smoke=$(mktemp -d)
    trap 'rm -rf -- "$smoke"' EXIT
    srun --ntasks=1 "$py" "$repo/scripts/run_insnotes_upstream.py" smoke "$smoke/upstream-smoke" "$data"
    rm -rf -- "$smoke"
    trap - EXIT
fi
if [[ ! -f "$run/training_complete.json" ]]; then
    code=0
    "$py" "$repo/scripts/run_insnotes_upstream.py" train "$run" "$data" || code=$?
    if (( code == 75 )); then
        echo "Requeueing own job $SLURM_JOB_ID from the saved complete epoch"
        scontrol requeue "$SLURM_JOB_ID"
        exit
    fi
    if (( code != 0 )); then exit "$code"; fi
fi
"$py" "$repo/scripts/run_insnotes_upstream.py" evaluate "$run" "$data"
