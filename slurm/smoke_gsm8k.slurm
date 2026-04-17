#!/bin/bash
# Minimal SLURM smoke test (CPU): run 1 GSM8K matrix once.
#
# Run:
#   sbatch slurm/smoke_gsm8k.slurm
#
# Check:
#   logs/smoke_gsm8k_<jobid>.out
#   outputs/figures/gsm8k/slurm_smoke.png

#SBATCH --account=torch_pr_790_general
#SBATCH --job-name=gsm8k_smoke
#SBATCH --output=logs/smoke_gsm8k_%j.out
#SBATCH --error=logs/smoke_gsm8k_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

# If your envs live on scratch, keep this for batch jobs too.
export CONDA_ENVS_PATH=/scratch/$USER/conda/envs
export CONDA_PKGS_DIRS=/scratch/$USER/conda/pkgs
conda activate gittins-torch

# Non-interactive SLURM bash sometimes leaves `python` on the wrong PATH even after
# `conda activate`. Always call the env interpreter explicitly.
PYTHON="${CONDA_ENVS_PATH}/gittins-torch/bin/python"
if [ ! -x "${PYTHON}" ]; then
  echo "ERROR: missing interpreter: ${PYTHON}" >&2
  exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs outputs/figures/gsm8k

MATRIX="data/matrices/gsm8k_1_samples_various_models_seed1.npy"
OUT="outputs/figures/gsm8k/slurm_smoke.png"
TRACES="outputs/figures/gsm8k/slurm_smoke_traces.npz"

echo "pwd=$(pwd)"
echo "python=${PYTHON}"
echo "which python fallback=$(command -v python || true)"
"${PYTHON}" --version
"${PYTHON}" -c "import torch; print('torch', torch.__version__)"
echo "matrix=${MATRIX}"
ls -lh "${MATRIX}"

"${PYTHON}" scripts/plot_simple_regret_gsm8k.py \
  --matrix "${MATRIX}" \
  --algorithms gittins \
  --eval-budget-fraction 0.02 \
  --out "${OUT}" \
  --traces-out "${TRACES}"

echo "DONE. Wrote ${OUT}"

