#!/bin/bash
#SBATCH --job-name=gittins-old-download
#SBATCH --partition=cs
#SBATCH --account=torch_pr_790_general
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm_logs/%x_%A_%a.out
#SBATCH --error=outputs/slurm_logs/%x_%A_%a.err

set -euo pipefail
cd /scratch/yh6415/project/BanditGittinsEval
export TMPDIR=/scratch/yh6415/project/BanditGittinsEval/.runtime_tmp
/scratch/yh6415/conda_envs/gittins/bin/python scripts/download_gittins_pilot_baselines.py --index "$SLURM_ARRAY_TASK_ID"
