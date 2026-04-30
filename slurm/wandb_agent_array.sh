#!/bin/bash
#SBATCH --account=torch_pr_790_general
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yh6415@nyu.edu
#SBATCH --job-name=wandb_gittins_eval
#SBATCH --output=slurm_logs/wandb_agent_%A_%a.out
#SBATCH --error=slurm_logs/wandb_agent_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --array=0-999%1000

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: sbatch slurm/wandb_agent_array.sh ENTITY/PROJECT/SWEEP_ID"
  exit 2
fi

SWEEP_PATH="$1"
PROJECT_DIR="${PROJECT_DIR:-/scratch/yh6415/GittinsBanditEval}"
PYTHON="${PYTHON:-/scratch/yh6415/conda_envs/gittins/bin/python}"

cd "${PROJECT_DIR}"
mkdir -p slurm_logs wandb /scratch/yh6415/pip_cache /scratch/yh6415/wandb_cache /scratch/yh6415/matplotlib_cache /scratch/yh6415/cache

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export WANDB_DIR="${PROJECT_DIR}/wandb"
export WANDB_MODE="${WANDB_MODE:-online}"
export PIP_CACHE_DIR="/scratch/yh6415/pip_cache"
export WANDB_CACHE_DIR="/scratch/yh6415/wandb_cache"
export WANDB_DATA_DIR="/scratch/yh6415/wandb_cache"
export WANDB_ARTIFACT_DIR="/scratch/yh6415/wandb_cache"
export MPLCONFIGDIR="/scratch/yh6415/matplotlib_cache"
export XDG_CACHE_HOME="/scratch/yh6415/cache"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "SWEEP_PATH=${SWEEP_PATH}"
echo "PYTHON=${PYTHON}"

"${PYTHON}" --version

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "ERROR: WANDB_API_KEY is not set."
  exit 3
fi

test -x "${PYTHON}"
test -f scripts/run_simple_regret_wandb.py
test -d data/BanditEval_matrices
test -d data_analysis/pricing

"${PYTHON}" -m wandb agent --forward-signals --count "${WANDB_AGENT_COUNT:-1}" "${SWEEP_PATH}"

echo "Finished task ${SLURM_ARRAY_TASK_ID}"
