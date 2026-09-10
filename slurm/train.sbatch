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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

config_path=""
run_name=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    if [[ "${args[$i]}" == "--config" && $((i + 1)) -lt ${#args[@]} ]]; then
        config_path="${args[$((i + 1))]}"
    fi
    if [[ "${args[$i]}" == "--name" && $((i + 1)) -lt ${#args[@]} ]]; then
        run_name="${args[$((i + 1))]}"
    fi
done

# Preserve uncommitted implementation as well as its Git identity. Only source
# files are included; datasets, model artifacts and Slurm output are excluded.
if [[ -n "$run_name" && "$run_name" != */* && "$run_name" != . && "$run_name" != .. ]]; then
    provenance_dir="$REPO_DIR/logs/$run_name/reproducibility"
    mkdir -p "$provenance_dir"
    git rev-parse HEAD > "$provenance_dir/git_head.txt"
    git status --short > "$provenance_dir/git_status.txt"
    git diff --binary HEAD > "$provenance_dir/tracked_changes.patch"
    git ls-files -co --exclude-standard -z -- '*.py' '*.yaml' '*.sh' '*.sbatch' '*.md' \
        | sort -zu > "$provenance_dir/source_files.list0"
    xargs -0 sha256sum < "$provenance_dir/source_files.list0" > "$provenance_dir/source_sha256.txt"
    tar -czf "$provenance_dir/source.tar.gz" --null -T "$provenance_dir/source_files.list0"
    if [[ -n "$config_path" ]]; then
        cp "$config_path" "$provenance_dir/submitted_config.yaml"
        sha256sum "$config_path" > "$provenance_dir/submitted_config.sha256"
    fi
fi

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
