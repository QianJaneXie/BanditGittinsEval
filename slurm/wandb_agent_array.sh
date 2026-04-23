#!/bin/bash
#SBATCH -J wandb_gittins_eval
#SBATCH -o slurm_logs/wandb_agent_%A_%a.out
#SBATCH -e slurm_logs/wandb_agent_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@nyu.edu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 24:00:00
#SBATCH --array=0-3
#SBATCH --partition=YOUR_PARTITION
# 如果 NYU Torch 需要 account，再加：
# #SBATCH --account=YOUR_ACCOUNT

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: sbatch --array=0-3 slurm/wandb_agent_array.sh ENTITY/PROJECT/SWEEP_ID"
  exit 2
fi

SWEEP_PATH="$1"

mkdir -p slurm_logs

# ===== 进入项目目录：这里改成你自己的仓库绝对路径 =====
cd /ABSOLUTE/PATH/TO/BanditGittinsEval

# ===== 激活环境：这里按你自己的环境改 =====
# 方案 A: conda
#source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh
#conda activate automl_env

# 方案 B: venv（如果你用 .venv，就把上面的 conda 两行注释掉，改成下面这行）
source .venv/bin/activate

# ===== 必须提前设置 W&B key；不要在集群里交互式 login =====
if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "ERROR: WANDB_API_KEY is not set."
  echo "Please run: export WANDB_API_KEY=your_key_before_sbatch"
  exit 3
fi

# 可选：固定默认 project / entity
export WANDB_ENTITY=EfficientLLMEval
export WANDB_PROJECT=GittinsBanditEval

# 确保依赖齐全
python -m pip install -e .

# 每个 array task 只领 1 个 run，最稳
wandb agent --forward-signals --count 1 "${SWEEP_PATH}"

# 退出环境
conda deactivate