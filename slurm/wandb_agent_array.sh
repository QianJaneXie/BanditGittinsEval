#!/bin/bash
#SBATCH --account=torch_pr_790_general
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yh6415@nyu.edu
#SBATCH --job-name=wandb_gittins_eval
#SBATCH --output=wandb_agent_%A_%a.out
#SBATCH --error=wandb_agent_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-14

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: sbatch --export=ALL slurm/wandb_agent_array.sh ENTITY/PROJECT/SWEEP_ID"
  exit 2
fi

SWEEP_PATH="$1"

cd "${SLURM_SUBMIT_DIR}"

mkdir -p slurm_logs
mkdir -p outputs
mkdir -p outputs/figures
mkdir -p wandb

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
else
  module load anaconda3/2024.02
  source /share/apps/anaconda3/2024.02/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-mygpu}"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export WANDB_DIR="${SLURM_SUBMIT_DIR}/wandb"
export WANDB_MODE="${WANDB_MODE:-online}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}"
echo "SWEEP_PATH=${SWEEP_PATH}"
echo "PYTHON=$(which python)"
python --version

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "ERROR: WANDB_API_KEY is not set."
  exit 3
fi

test -f scripts/plot_simple_regret_single_policy_wandb.py
test -f scripts/config/GSM8KSimpleRegretSinglePolicy.yml
test -f data/BanditEval_matrices/gsm8k_1_samples_various_models_seed1.npy
test -f data_analysis/pricing/gsm8k_various_models_configurations_price_ratio_1to2_rounded.json

wandb agent --forward-signals --count 1 "${SWEEP_PATH}"

echo "Finished task ${SLURM_ARRAY_TASK_ID}"

if [ -n "${VIRTUAL_ENV:-}" ]; then
  deactivate
elif command -v conda >/dev/null 2>&1; then
  conda deactivate
fi