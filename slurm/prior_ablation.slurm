#!/bin/bash
# Prior ablation for Gittins (CPU, SLURM array).
# - Outputs are written under outputs/figures/{gsm8k,piqa,mmlu}/
# - Each array task runs one (dataset, prior) combination.
#
# Usage:
#   1) Edit --account and conda env below.
#   2) (Optional) override MMLU_SUBJECT, MATRIX_* via sbatch --export
#      e.g. sbatch --export=ALL,MMLU_SUBJECT=world_religions slurm/prior_ablation.slurm
#   3) sbatch slurm/prior_ablation.slurm
#
# Notes:
# - This script uses scripts/plot_simple_regret_gsm8k.py for all datasets; it works for any matrix .npy.
# - If your cluster uses different module names/paths, adjust the module load + conda.sh lines.

#SBATCH --account=torch_pr_790_general
#SBATCH --job-name=gittins_prior
#SBATCH --output=logs/gittins_prior_%j_%a.out
#SBATCH --error=logs/gittins_prior_%j_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-71

set -euo pipefail

module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

# Envs live on scratch (recommended); keep this for batch jobs.
export CONDA_ENVS_PATH=/scratch/$USER/conda/envs
export CONDA_PKGS_DIRS=/scratch/$USER/conda/pkgs
conda activate gittins-torch

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
mkdir -p outputs/figures/gsm8k outputs/figures/piqa outputs/figures/mmlu

# ---------- dataset matrices ----------
MATRIX_GSM8K_DEFAULT="data/matrices/gsm8k_1_samples_various_models_seed1.npy"
MATRIX_PIQA_DEFAULT="data/matrices/piqa_1_samples_various_models_seed1.npy"
# MMLU matrices in this repo live under data/MMLU_matrix/*.npy
MMLU_DIR_DEFAULT="data/MMLU_matrix"

MATRIX_GSM8K="${MATRIX_GSM8K:-$MATRIX_GSM8K_DEFAULT}"
MATRIX_PIQA="${MATRIX_PIQA:-$MATRIX_PIQA_DEFAULT}"
MMLU_DIR="${MMLU_DIR:-$MMLU_DIR_DEFAULT}"

# Pick 10 MMLU subjects: top-5 by #cols + top-5 with 200–300 cols (precomputed from this repo).
MMLU_SUBJECTS=(
  professional_law
  moral_scenarios
  miscellaneous
  professional_psychology
  high_school_psychology
  professional_accounting
  professional_medicine
  high_school_mathematics
  clinical_knowledge
  security_studies
)

# ---------- prior grid (6 combos) ----------
# prior_idx in [0..5]
prior_idx=$(( SLURM_ARRAY_TASK_ID % 6 ))
case "${prior_idx}" in
  0) PM=0.5; PV=0.01 ;;
  1) PM=0.5; PV=0.02 ;;
  2) PM=0.5; PV=0.04 ;;
  3) PM=0.2; PV=0.01 ;;
  4) PM=0.2; PV=0.04 ;;
  5) PM=0.3; PV=0.04 ;;
  *) echo "bad prior_idx=${prior_idx}" >&2; exit 2 ;;
esac

# ---------- dataset selection (12 datasets = 2 + 10 MMLU) ----------
# dataset_idx in [0..11]
dataset_idx=$(( SLURM_ARRAY_TASK_ID / 6 ))
case "${dataset_idx}" in
  0)
    DATASET="gsm8k"
    MATRIX="${MATRIX_GSM8K}"
    OUT_DIR="outputs/figures/gsm8k"
    ;;
  1)
    DATASET="piqa"
    MATRIX="${MATRIX_PIQA}"
    OUT_DIR="outputs/figures/piqa"
    ;;
  *)
    mmlu_i=$(( dataset_idx - 2 ))
    if [ "${mmlu_i}" -lt 0 ] || [ "${mmlu_i}" -ge "${#MMLU_SUBJECTS[@]}" ]; then
      echo "bad dataset_idx=${dataset_idx} (expected 0..11)" >&2
      exit 2
    fi
    subj="${MMLU_SUBJECTS[$mmlu_i]}"
    DATASET="mmlu_${subj}"
    MATRIX="${MMLU_DIR}/${subj}.npy"
    OUT_DIR="outputs/figures/mmlu"
    ;;
esac

FIG="${OUT_DIR}/simple_regret_${DATASET}_pm${PM}_pv${PV}.png"
TRACES="${OUT_DIR}/simple_regret_${DATASET}_pm${PM}_pv${PV}_traces.npz"

echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "DATASET=${DATASET}"
echo "MATRIX=${MATRIX}"
echo "PRIOR mean=${PM} var=${PV}"
echo "FIG=${FIG}"
echo "TRACES=${TRACES}"
echo "python=$(which python)"
python --version

python scripts/plot_simple_regret_gsm8k.py \
  --matrix "${MATRIX}" \
  --algorithms gittins \
  --eval-budget-fraction 0.10 \
  --gittins-prior-mean "${PM}" \
  --gittins-prior-variance "${PV}" \
  --out "${FIG}" \
  --traces-out "${TRACES}"

