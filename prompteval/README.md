# Efficient multi-prompt evaluation of LLMs

Welcome to the [*PromptEval* GitHub repository](https://github.com/felipemaiapolo/prompteval)! Here you will find more information about our implementation of *PromptEval* and datasets introduced in

[Maia Polo, Felipe, Ronald Xu, Lucas Weber, Mírian Silva, Onkar Bhardwaj, Leshem Choshen, Allysson Flavio Melo de Oliveira, Yuekai Sun, and Mikhail Yurochkin. "Efficient multi-prompt evaluation of LLMs." arXiv preprint arXiv:2405.17202 (2024).](https://arxiv.org/abs/2405.17202)

## Running BAI (this repo)

Best-arm identification (BAI): successive **halving** + L2-regularized logistic regression, under a 10% observation budget. Supported benchmarks: **MMLU**, **GSM8K**, and **PIQA**.

Run all commands from the **BanditGittinsEval repo root**. Python deps: `numpy`, `scikit-learn`, `matplotlib` (for plotting).

### MMLU

**Prerequisites**

- `prompteval/data/Ys.pickle` — correctness matrices (15 LLMs × 100 prompt templates per subject)
- `prompteval/data/Xs.pickle` — prompt covariates (discrete features, sentence-transformer PCA, fine-tuned BERT PCA)

Pickles keep **15 separate** `(100 × n_questions)` matrices per subject. At runtime with the default `--combine-models`, they are stacked into **1500 arms** (each arm = one `(LLM, prompt template)` pair).

```bash
# Default: abstract_algebra + professional_law, combine_models=True, sklearn
python prompteval/bai_evaluation.py --bench MMLU

# Set number of sampling repeats (default is 20) → seeds 0..N-1
python prompteval/bai_evaluation.py --bench MMLU --random-seeds 20

# Or a single fixed seed (overrides --random-seeds)
python prompteval/bai_evaluation.py --bench MMLU --tasks anatomy --seed 0

# Chosen subjects
python prompteval/bai_evaluation.py --bench MMLU --tasks abstract_algebra,anatomy --random-seeds 20

# Local / debug: fewer seeds
python prompteval/bai_evaluation.py --bench MMLU --tasks anatomy --random-seeds 2

# All 57 MMLU subjects
python prompteval/bai_evaluation.py --bench MMLU --all-tasks --random-seeds 20

# Cost accounting (sampling unchanged; x-axis later in cost units)
python prompteval/bai_evaluation.py --bench MMLU --cost-aware --random-seeds 20

python prompteval/bai_evaluation.py --help
```

| Setting | Default (`--bench MMLU`) |
|--------|--------|
| Data | `prompteval/data/` |
| Tasks | `abstract_algebra`, `professional_law` |
| Arms | 1500 stacked `(LLM, template)` pairs (`--combine-models`) |
| Budget | 10% of matrix cells, ≤ 5 successive-halving rounds |
| Backend | `sklearn` |
| Seeds | `--random-seeds 20` → seeds `0..N-1` (averaged); or `--seed K` for a single repeat |

Use `--no-combine-models` for 100 template arms **per LLM** (then average over LLMs).

Each MMLU subject is written as soon as it finishes (**one raw + one processed file per subject**), so a long `--all-tasks` run can be plotted subject-by-subject without waiting for the rest.

**Plot** (loads **one processed file per subject**):

```bash
python prompteval/plot_bai_regret.py --bench MMLU --tasks abstract_algebra
python prompteval/plot_bai_regret.py --bench MMLU --tasks abstract_algebra,professional_law
python prompteval/plot_bai_regret.py --bench MMLU --tasks abstract_algebra,professional_law --cost-aware
```

See [Output format](#output-format) below for result filenames and array layout.

### GSM8K and PIQA

BanditEval matrices: rows = **configurations** (arms), columns = i.i.d. question samples. For GSM8K `various_models`, each row is a `(model, prompt, temperature, max_tokens)` config (122 arms; not a full cartesian product). PIQA is analogous (103 arms).

**Covariates today:** `Xs` views are empty, so BAI only runs the **one-hot / identity baseline** (`X = I_{n_arms}`, e.g. 122-dim for GSM8K). A factored embedding (one-hot model ⊕ one-hot prompt ⊕ …) is a natural alternative and is *not* wired in yet (see also `data/bo_inputs/gsm8k/*_bo_inputs.npz`, which already stores `model_id` / `prompt_id` / …).

**Build pickles once** from `data/BanditEval_matrices/`:

```bash
python prompteval/build_banditeval_pickle.py
# writes prompteval/pickle/Ys.pickle and Xs.pickle
```

Each benchmark has five tasks `various_models_seed1`…`seed5` (same questions, different LLM-query seeds).

```bash
# Defaults: all 5 data seeds, combine_models=False, tag _unitcost
python prompteval/bai_evaluation.py --bench GSM8K
python prompteval/bai_evaluation.py --bench PIQA

python prompteval/bai_evaluation.py --bench GSM8K --cost-aware
python prompteval/bai_evaluation.py --bench PIQA --tasks various_models_seed1 --random-seeds 20
```

| Setting | Default (`--bench GSM8K` / `PIQA`) |
|--------|--------|
| Data | `prompteval/pickle/` |
| Tasks | all `various_models_seed1`…`seed5` |
| Arms | 122 (GSM8K) or 103 (PIQA) configs; **no** LLM×template stacking |
| Covariates | one-hot baseline only (`Xs` views are empty) |
| Results tag | `_unitcost` by default, or `_costaware` with `--cost-aware` (mutually exclusive) |

**Plot** (needs the **processed** `.npy`; averages the 5 data seeds into **one** curve):

```bash
python prompteval/plot_bai_regret.py --bench GSM8K
python prompteval/plot_bai_regret.py --bench PIQA
python prompteval/plot_bai_regret.py --bench GSM8K --cost-aware
```

Figures go to `prompteval/outputs/bai_regret_plots/`. Curves are drawn as a **staircase** (`steps-post`): simple regret is constant between evaluations. Use `--per-task` only if you want one panel per data seed.

If plotting fails with “Processed results not found” but the raw file exists, re-run `bai_evaluation.py` for that bench/tag (or regenerate processed from raw with the same hold-last aggregation).

### Output format

**MMLU** writes **one raw + one processed file per subject** (saved as soon as that subject finishes):

```text
prompteval/results/bai_results_MMLU_<subject><tag>.npy
prompteval/results/bai_processed_results_MMLU_<subject><tag>.npy
```

**GSM8K / PIQA** write a single multi-task pair (all data seeds in one file):

```text
prompteval/results/bai_results_<BENCH><tag>.npy              # raw (required to rebuild processed)
prompteval/results/bai_processed_results_<BENCH><tag>.npy    # seed-averaged (what the plotter reads)
```

**Filename tags** (mutually exclusive cost mode; `_combined` only for MMLU stacking):

| Condition | Suffix |
|-----------|--------|
| GSM8K / PIQA (observation / unit cost; default) | `_unitcost` |
| `--cost-aware` | `_costaware` (replaces `_unitcost`, never `_unitcost_costaware`) |
| `--combine-models` (MMLU default) | `_combined` |

Examples: `bai_processed_results_MMLU_abstract_algebra_combined.npy`, `bai_processed_results_GSM8K_unitcost.npy`, `bai_processed_results_GSM8K_costaware.npy`.

#### Processed results (what the plot script reads)

Saved as a **dict** (via `np.save`):

```python
{
  "curves": ndarray,          # shape (n_tasks, n_budgets, n_blocks, 2, n_points)
  "channels": ["simple_regret", "budget"],
  "tasks": [...],             # task names in axis-0 order
  "seeds": [...],             # RNG seeds that were averaged
  "bench": "MMLU" | "GSM8K" | "PIQA",
  "combine_models": bool,
  "cost_aware": bool,
  "aggregation": "phase_mean" | "cost_grid_hold_last",
  "n_models_stacked": int,
  "update_fields": [...],     # fields present in each raw phase update
  "note": "...",
}
```

`curves` axes:

| Axis | Meaning |
|------|---------|
| `n_tasks` | MMLU per-subject file: always `1`. GSM8K/PIQA multi-task file: number of data seeds |
| `n_budgets` | Always `1` (single budget = 10% of matrix cells) |
| `n_blocks` | Covariate blocks: MMLU = **4**, GSM8K/PIQA = **1** |
| channel (`2`) | `0` = simple regret, `1` = cumulative spend (plot x-axis) |
| `n_points` | Unit-cost: successive-halving phases (≤ 5). Cost-aware: # of distinct evaluation spends on the shared grid (scales with seeds × phases; multi-task files align tasks onto one grid) |

**How seeds are combined**

- **Unit cost** (`aggregation: "phase_mean"`): observation counts align across seeds → average regret (and spend) **per phase** → ≤ 5 points. If a run stops early, later phase slots **keep the last regret** (unchanged), so that seed stays in the average.
- **Cost-aware** (`aggregation: "cost_grid_hold_last"`): different seeds sample different arms → different spend at each phase. x-grid = sorted **unique evaluation spends** from the random-seed runs. At each cost, each seed contributes its **last evaluated** simple regret (hold-last / step — no linear interpolation). A seed that never reaches a higher spend still contributes that held regret there. The plotter draws the same step semantics (`steps-post`).

Sampling is always unit-cost (10% of cells); `--cost-aware` only changes the **recorded** spend channel used for the x-axis.

MMLU block order (index → covariates):

0. one-hot / identity baseline (`X=None` → `I_{n_arms}`)
1. discrete template features
2. sentence-transformer (PCA)
3. fine-tuned BERT (PCA)

GSM8K/PIQA only have block `0` (flat one-hot over configuration arms).

Spend channel units: observation counts by default; cost units with `--cost-aware`.

#### Per-update records (raw file — chosen model / regret / budget)

Each successive-halving phase writes an **update dict** (one “return” moment):

| Field | Meaning |
|-------|---------|
| `phase` | Round index `0 .. n_phases-1` |
| `budget` | Cumulative spend after sampling this phase |
| `chosen_arm` | Arm index returned by logistic regression (the identified model / `(LLM, template)` row) |
| `chosen_mean` | True mean reward of `chosen_arm` |
| `oracle_mean` | Best arm’s true mean |
| `simple_regret` | `oracle_mean - chosen_mean` |
| `n_active` | Number of arms still active before elimination |

If a run stops before later phases (labels exhausted, one arm left, …), remaining phase slots **copy the last budget and simple regret** (chosen arm unchanged).

These live in the **raw** file (not averaged — arm identity is per seed):

```python
raw = np.load("prompteval/results/bai_results_MMLU_abstract_algebra_combined.npy", allow_pickle=True).item()
# One evaluate_bai run:
run = raw["out"][0]           # aligned with raw["jobs"][0] == (task, seed)
updates = run[0][0]           # budget 0, covariate block 0
print(updates[0])
# {'phase': 0, 'budget': 2440.0, 'chosen_arm': 17, 'chosen_mean': 0.42,
#  'oracle_mean': 0.55, 'simple_regret': 0.13, 'n_active': 1500}
```

Load processed curves (one subject file; `n_tasks=1`):

```python
import numpy as np
proc = np.load(
    "prompteval/results/bai_processed_results_MMLU_abstract_algebra_combined.npy",
    allow_pickle=True,
).item()
r = proc["curves"]            # (1, 1, n_blocks, 2, n_points)
regret = r[0, 0, 0, 0, :]     # baseline block, simple regret vs phase / cost
spend  = r[0, 0, 0, 1, :]     # same, cumulative budget (x-axis)
print(proc["tasks"], proc["seeds"], proc.get("aggregation"))
```

#### Plots

`plot_bai_regret.py` reads the **processed** file(s) and writes PNGs to:

```text
# MMLU (one subject)
prompteval/outputs/bai_regret_plots/bai_simple_regret_MMLU_<subject>_combined.png
# GSM8K / PIQA
prompteval/outputs/bai_regret_plots/bai_simple_regret_<BENCH><tag>.png
```

Regret vs budget is plotted as a **step function** (flat between evaluations, jump at the next evaluation).
---

## Overview

Most popular benchmarks for comparing LLMs rely on a limited set of prompt templates, which may not fully capture the LLMs’ abilities and can affect the reproducibility of results on leaderboards. This repository introduces our implementation of *PromptEval*, a method for estimating performance across a large set of prompts by borrowing strength across prompts and examples to produce accurate estimates under practical evaluation budgets.

##  Quick start

Please check our [demo](https://github.com/felipemaiapolo/prompteval/blob/main/notebooks/PromptEval_demo.ipynb) on how to use *PromptEval* in your own data.

## Repository Structure

- `data/`: Contains the evaluation data used in the experiments.
- `prompteval/`: Source code for the PromptEval method and utilities.
- `notebooks/`: Jupyter notebooks used to create plots for the PromptEval paper.
- `results/`: Results from the experiments conducted in the paper.
- `mmlu_data/`: Contains code for gathering evaluation data.

## Installation

To use the code in this repository, clone the repo and install the required dependencies:

```bash
git clone https://github.com/felipemaiapolo/prompteval.git
cd prompteval
pip install -e .
```

## NEW: Efficient multi-prompt evaluation with `PromptBench` and `lm-evaluation-harness`

### PromptBench

PromptEval was implemented by PromptBench. Please check it [here](https://github.com/microsoft/promptbench).

### lm-evaluation-harness

To learn how to combine PromptEval with `lm-evaluation-harness`, please follow these steps:

1. Please Git clone our version of [`lm-evaluation-harness`](https://github.com/mirianfsilva/lm-evaluation-harness) into the main directory of PromptEval; we have already submitted a [PR](https://github.com/EleutherAI/lm-evaluation-harness/pull/2520) to the main version of the package.
2. Git checkout the `examples-arg` branch of the `lm-evaluation-harness` repository you have just cloned.
3. Inside the `lm-evaluation-harness` main directory, please install `pip install -e .`.
4. Please check our [demo](https://github.com/felipemaiapolo/prompteval/blob/main/notebooks/llm-eval-harness_demo.ipynb). We focus on MMLU; however, the ideas we present can be used for other benchmarks as well.


## Reproducing the main results of the paper

To reproduce the results in our paper, please follow the steps after cloning the repo and installing dependencies:
1. Download the BBH and LMentry data, produced by the authors of "[State of What Art? A Call for Multi-Prompt LLM Evaluation](https://arxiv.org/abs/2401.00595)", from [here](https://www.dropbox.com/scl/fo/y9dd8zbteyf0xrjxdtm3e/h/raw%20open-source%20model%20responses%20with%20gold%20and%20auto%20validation%20values.zip?rlkey=okp52gleuibw72fhe62egr6lp&e=1&dl=0). Place the unzipped folder "raw open-source model responses with gold and auto validation values" inside the data directory;
2. Process data by running ``./prompteval/create_data.py``;
3. Run main experiments by running ``./prompteval/dist_evaluation.py``. Example: ``python ./prompteval/dist_evaluation.py --bench 'BBH' --random_seeds 5``;
4. Run best prompt identification by running ``./prompteval/bai_evaluation.py``. Example: ``python ./prompteval/bai_evaluation.py --bench 'BBH' --random_seeds 5``.
5. Create plots using the notebooks in the notebooks directory.

### Fine-tuning embeddings

To fine-tune BERT representations run the following:

```bash
python ./prompteval/ft_representations.py --model_name "bert-base-uncased" \
                             --lr 2e-05 \
                             --weight_decay 1e-06 \
                             --gamma .99995 \
                             --bs 96 \
                             --n_epochs 5 \
                             --warmup_steps 200 \
                             --bench "BBH" 
```
Note, that this requires the file `./data/Ys.pickle` to contain correctness data for the respective benchmark as the `create_data.py` script creates it. Add `--push_to_hub`, to automatically push the resulting model to your namespace on the huggingface hub (remember to `huggingface-cli login` before training).

## LLM-as-a-judge experiment
To run the LLM-as-a-judge experiment, please follow the steps:
1. Install AlpacaEval 2.0 using the command `pip install alpaca-eval==0.6.4`;
2. Run ``python ./prompteval/generate_prompts.py`` to generate prompt variations. Having a GPU will accelerate this step because we use SentenceTransformers to encode texts;
3. Move the directories `./prompteval/data/templates/AlpacaEval/configs` and `./prompteval/data/templates/AlpacaEval/templates` to your `evaluators_configs` AlpacaEval folder; for example, if you are using a Miniconda 3 (or Anaconda) environment, your folder should be in the directory `miniconda3/envs/{ENV_NAME}/lib/python{PYTHON_VERSION}/site-packages/alpaca_eval`;
4. Open `./prompteval/evaluate.py` and, at the top of the file, create an object called `evaluators_configs_path` and paste the path to the `evaluators_configs` directory to it; if you are using a Miniconda 3 (or Anaconda) environment, your `evaluators_configs` directory should be in the directory `home/miniconda3/envs/{ENV_NAME}/lib/python{PYTHON_VERSION}/site-packages/alpaca_eval/evaluators_configs`;
5. Export your OpenAI API key following https://pypi.org/project/alpaca-eval/0.6.4/ and run `./prompteval/evaluate.py` to conduct the evaluation step;
6. Run the notebook `./notebooks/llm_judge_plots.ipynb` to get the plots.

## MMLU Data

We make our MMLU collected data available on [Hugging Face](https://huggingface.co/PromptEval). The data includes evaluation for 15 different SOTA LLMs and 100 different prompt templates.

## Citing

    @article{polo2024efficient,
    title={Efficient multi-prompt evaluation of LLMs},
    author={Polo, Felipe Maia and Xu, Ronald and Weber, Lucas and Silva, M{\'\i}rian and Bhardwaj, Onkar and Choshen, Leshem and de Oliveira, Allysson Flavio Melo and Sun, Yuekai and Yurochkin, Mikhail},
    journal={arXiv preprint arXiv:2405.17202},
    year={2024}
    }
