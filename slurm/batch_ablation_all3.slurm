#!/bin/bash
# Batch-size ablation with all three algorithms on one plot (UCB-E, UCB-E-LRF, Gittins).
# Uses the same 12 matrices as slurm/prior_ablation.slurm (GSM8K, PIQA, 10 MMLU subjects).
#
# For each (dataset, batch size B): runs
#   --algorithms ucb lrf gittins
#   --batch-size B --gittins-batch-size B
# so UCB/LRF and Gittins share the same per-step batch (fair comparison; τ² for Gittins uses 1/(4B)).
#
# Prior is fixed to script defaults (μ=0.2, v=0.01) unless you edit PM/PV below.
#
# Total tasks: 12 datasets × 4 batch sizes = 48  →  --array=0-47
#
# Run:
#   sbatch slurm/batch_ablation_all3.slurm
#
# Outputs (per task):
#   outputs/figures/{gsm8k,piqa,mmlu}/simple_regret_<DATASET>_all3_bs<B>_pm*_pv*.png
#   ..._traces.npz, ..._traces.meta.json
# Logs:
#   logs/batch_all3_%j_%a.out / .err

#SBATCH --account=torch_pr_790_general
#SBATCH --job-name=batch_all3
#SBATCH --output=logs/batch_all3_%j_%a.out
#SBATCH --error=logs/batch_all3_%j_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-47

set -euo pipefail

module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

export CONDA_ENVS_PATH=/scratch/$USER/conda/envs
export CONDA_PKGS_DIRS=/scratch/$USER/conda/pkgs
conda activate gittins-torch

PYTHON="${CONDA_ENVS_PATH}/gittins-torch/bin/python"
if [ ! -x "${PYTHON}" ]; then
  echo "ERROR: missing interpreter: ${PYTHON}" >&2
  exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
mkdir -p outputs/figures/gsm8k outputs/figures/piqa outputs/figures/mmlu

# ---------- same matrices as prior_ablation.slurm ----------
MATRIX_GSM8K_DEFAULT="data/matrices/gsm8k_1_samples_various_models_seed1.npy"
MATRIX_PIQA_DEFAULT="data/matrices/piqa_1_samples_various_models_seed1.npy"
MMLU_DIR_DEFAULT="data/MMLU_matrix"

MATRIX_GSM8K="${MATRIX_GSM8K:-$MATRIX_GSM8K_DEFAULT}"
MATRIX_PIQA="${MATRIX_PIQA:-$MATRIX_PIQA_DEFAULT}"
MMLU_DIR="${MMLU_DIR:-$MMLU_DIR_DEFAULT}"

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

# ---------- batch grid (4 values; meeting: try 8/16/20/32) ----------
NUM_BS=4
batch_idx=$(( SLURM_ARRAY_TASK_ID % NUM_BS ))
case "${batch_idx}" in
  0) B=8 ;;
  1) B=16 ;;
  2) B=20 ;;
  3) B=32 ;;
  *) echo "bad batch_idx=${batch_idx}" >&2; exit 2 ;;
esac

# ---------- dataset index [0..11] ----------
dataset_idx=$(( SLURM_ARRAY_TASK_ID / NUM_BS ))
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

# Fixed prior (match script defaults; change here if you want one grid for all runs)
PM="${PM:-0.2}"
PV="${PV:-0.01}"

STEM="simple_regret_${DATASET}_all3_bs${B}_pm${PM}_pv${PV}"
FIG="${OUT_DIR}/${STEM}.png"
TRACES="${OUT_DIR}/${STEM}_traces.npz"

echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "DATASET=${DATASET}"
echo "MATRIX=${MATRIX}"
echo "B=${B} (UCB/LRF batch-size == Gittins batch-size)"
echo "PRIOR mean=${PM} var=${PV}"
echo "FIG=${FIG}"
echo "TRACES=${TRACES}"
echo "python=${PYTHON}"
"${PYTHON}" --version
"${PYTHON}" -c "import torch; print('torch', torch.__version__)"

"${PYTHON}" scripts/plot_simple_regret_gsm8k.py \
  --matrix "${MATRIX}" \
  --algorithms ucb lrf gittins \
  --batch-size "${B}" \
  --gittins-batch-size "${B}" \
  --eval-budget-fraction 0.10 \
  --gittins-prior-mean "${PM}" \
  --gittins-prior-variance "${PV}" \
  --out "${FIG}" \
  --traces-out "${TRACES}"

echo "DONE. Wrote ${FIG}"
