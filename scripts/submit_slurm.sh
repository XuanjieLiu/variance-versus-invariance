#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/xuanjie.liu/Projects/variance-versus-invariance"
PARTITION="${VVI_PARTITION:-ws-ia}"
MEMORY="${VVI_MEMORY:-32G}"
CPUS="${VVI_CPUS:-4}"
TIME_LIMIT="${VVI_TIME:-08:00:00}"
JOB_SCRIPT="${VVI_JOB_SCRIPT:-slurm/train.sbatch}"

if [[ $# -ne 3 ]]; then
    echo "Usage: scripts/submit_slurm.sh <config> <run-name> <run-id>" >&2
    exit 2
fi

config_path="$1"
run_name="$2"
run_id="$3"

cd "$REPO_DIR"
if [[ ! -f "$config_path" ]]; then
    echo "Config does not exist: $config_path" >&2
    exit 2
fi

mkdir -p slurm_logs
if [[ ! -f "$JOB_SCRIPT" ]]; then
    echo "Job script does not exist: $JOB_SCRIPT" >&2
    exit 2
fi
dependency_args=()
if [[ -n "${VVI_DEPENDENCY:-}" ]]; then
    dependency_args+=(--dependency="$VVI_DEPENDENCY")
fi
job_name="$(printf '%s' "$run_id" | tr '[:upper:]' '[:lower:]')"

job_id="$(
    sbatch --parsable \
        --partition="$PARTITION" \
        --nodes=1 \
        --gres=gpu:1 \
        --mem="$MEMORY" \
        --cpus-per-task="$CPUS" \
        --time="$TIME_LIMIT" \
        --job-name="$job_name" \
        --output="slurm_logs/%x-%j.out" \
        --error="slurm_logs/%x-%j.err" \
        "${dependency_args[@]}" \
        "$JOB_SCRIPT" \
        --config "$config_path" \
        --name "$run_name"
)"

echo "$job_id"
