#!/bin/bash
#SBATCH --job-name=gittins-eb
#SBATCH --partition=cs
#SBATCH --account=torch_pr_790_general
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm_logs/%x_%A_%a.out
#SBATCH --error=outputs/slurm_logs/%x_%A_%a.err

set -euo pipefail

REPO_ROOT=/scratch/yh6415/project/BanditGittinsEval
PYTHON=/scratch/yh6415/conda_envs/gittins/bin/python

cd "$REPO_ROOT"
mkdir -p outputs/slurm_logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

CONFIG_PATH="$1"
OUTPUT_ROOT="$2"
shift 2

"$PYTHON" scripts/run_gittins_eb_local_sweep.py \
  --config "$CONFIG_PATH" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --output-root "$OUTPUT_ROOT" \
  "$@"
