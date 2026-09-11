#!/bin/bash
#SBATCH --job-name=gittins-eb-plot
#SBATCH --partition=cs
#SBATCH --account=torch_pr_790_general
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm_logs/%x_%j.out
#SBATCH --error=outputs/slurm_logs/%x_%j.err

set -euo pipefail
cd /scratch/yh6415/project/BanditGittinsEval
export TMPDIR=/scratch/yh6415/project/BanditGittinsEval/.runtime_tmp
export MPLCONFIGDIR=/scratch/yh6415/project/BanditGittinsEval/.runtime_tmp/matplotlib
/scratch/yh6415/conda_envs/gittins/bin/python scripts/plot_gittins_eb_pilot_comparison.py
