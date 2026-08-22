#!/usr/bin/env bash
set -euo pipefail

report_exit() {
    exit_code=$?
    echo "finished_at=$(date --iso-8601=seconds)"
    echo "exit_code=$exit_code"
}

report_signal() {
    signal_name="$1"
    exit_code="$2"
    echo "received_signal=$signal_name" >&2
    exit "$exit_code"
}

trap report_exit EXIT
trap 'report_signal TERM 143' TERM
trap 'report_signal INT 130' INT

REPO_DIR="/home/xuanjie.liu/Projects/variance-versus-invariance"
CONDA_INIT="/home/xuanjie.liu/miniconda3/etc/profile.d/conda.sh"

source "$CONDA_INIT"
conda activate xuanjie
cd "$REPO_DIR"

config_path=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    if [[ "${args[$i]}" == "--config" && $((i + 1)) -lt ${#args[@]} ]]; then
        config_path="${args[$((i + 1))]}"
        break
    fi
done

echo "started_at=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "slurm_job_id=${SLURM_JOB_ID:-unknown}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "git_head=$(git rev-parse HEAD)"
echo "git_status_begin"
git status --short
echo "git_status_end"
if [[ -n "$config_path" ]]; then
    sha256sum "$config_path"
fi
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version \
    --format=csv,noheader

python run_training.py "$@"
