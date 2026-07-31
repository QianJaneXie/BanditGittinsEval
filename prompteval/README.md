# Efficient multi-prompt evaluation of LLMs

Welcome to the [*PromptEval* GitHub repository](https://github.com/felipemaiapolo/prompteval)! Here you will find more information about our implementation of *PromptEval* and datasets introduced in

[Maia Polo, Felipe, Ronald Xu, Lucas Weber, Mírian Silva, Onkar Bhardwaj, Leshem Choshen, Allysson Flavio Melo de Oliveira, Yuekai Sun, and Mikhail Yurochkin. "Efficient multi-prompt evaluation of LLMs." arXiv preprint arXiv:2405.17202 (2024).](https://arxiv.org/abs/2405.17202)

## Running BAI (this repo)

Best-arm identification (BAI): successive halfing + L2-regularized logistic regression, under a 10% observation budget. Supported benchmarks: **MMLU**, **GSM8K**, and **PIQA**.

Run all commands from the **BanditGittinsEval repo root**. Python deps: `numpy`, `scikit-learn`, `joblib`, `matplotlib` (for plotting).

### MMLU

**Prerequisites**

- `prompteval/data/Ys.pickle` — correctness matrices (15 LLMs × 100 prompt templates per subject)
- `prompteval/data/Xs.pickle` — prompt covariates (discrete features, sentence-transformer PCA, fine-tuned BERT PCA)

Pickles keep **15 separate** `(100 × n_questions)` matrices per subject. At runtime with the default `--combine-models`, they are stacked into **1500 arms** (each arm = one `(LLM, prompt template)` pair).

```bash
# Default: abstract_algebra + professional_law, 20 sampling seeds, combine_models=True, sklearn
python prompteval/bai_evaluation.py --bench MMLU

# Chosen subjects
python prompteval/bai_evaluation.py --bench MMLU --tasks abstract_algebra,anatomy,professional_law

# All 57 MMLU subjects
python prompteval/bai_evaluation.py --bench MMLU --all-tasks

# Cost accounting (sampling unchanged; x-axis later in cost units)
python prompteval/bai_evaluation.py --bench MMLU --cost-aware

python prompteval/bai_evaluation.py --bench MMLU --random-seeds 20 --n-jobs 4
python prompteval/bai_evaluation.py --help
```

| Setting | Default (`--bench MMLU`) |
|--------|--------|
| Data | `prompteval/data/` |
| Tasks | `abstract_algebra`, `professional_law` |
| Arms | 1500 stacked `(LLM, template)` pairs (`--combine-models`) |
| Budget | 10% of matrix cells, ≤ 5 successive-halving rounds |
| Backend | `sklearn` |
| Seeds | `--random-seeds 20` (observation-sampling repeats; averaged in processed results) |

Use `--no-combine-models` for 100 template arms **per LLM** (then average over LLMs).

**Outputs** under `prompteval/results/`:

| File | Contents |
|------|----------|
| `bai_results_MMLU_combined.npy` | Raw nested results |
| `bai_processed_results_MMLU_combined.npy` | Seed-averaged `(n_tasks, n_budgets, n_blocks, 2, n_phases)` |

Channel `0` = simple regret, `1` = cumulative spend. `n_blocks = 4` (one-hot + 3 embedding views). With `--cost-aware`, filenames include `_costaware`.

**Plot** (one panel per MMLU subject):

```bash
python prompteval/plot_bai_regret.py --bench MMLU --tasks abstract_algebra,professional_law
python prompteval/plot_bai_regret.py --bench MMLU --tasks abstract_algebra,professional_law --cost-aware
```

### GSM8K and PIQA

BanditEval matrices: rows = **models** (arms), columns = i.i.d. samples. No prompt templates, so only the one-hot (baseline) covariate block runs.

**Build pickles once** from `data/BanditEval_matrices/`:

```bash
python prompteval/build_banditeval_pickle.py
# writes prompteval/pickle/Ys.pickle and Xs.pickle
```

Each benchmark has five tasks `various_models_seed1`…`seed5` (same questions, different LLM-query seeds).

```bash
# Defaults: all 5 data seeds, combine_models=False, tag _banditeval
python prompteval/bai_evaluation.py --bench GSM8K
python prompteval/bai_evaluation.py --bench PIQA

python prompteval/bai_evaluation.py --bench GSM8K --cost-aware
python prompteval/bai_evaluation.py --bench PIQA --tasks various_models_seed1 --random-seeds 20
```

| Setting | Default (`--bench GSM8K` / `PIQA`) |
|--------|--------|
| Data | `prompteval/pickle/` |
| Tasks | all `various_models_seed1`…`seed5` |
| Arms | 122 (GSM8K) or 103 (PIQA) models; **no** LLM×template stacking |
| Covariates | one-hot baseline only (`Xs` views are empty) |
| Results tag | `_banditeval` (e.g. `bai_processed_results_GSM8K_banditeval.npy`) |

**Plot** (average the 5 data seeds into **one** curve — they are repeated measurements, not different datasets):

```bash
python prompteval/plot_bai_regret.py --bench GSM8K
python prompteval/plot_bai_regret.py --bench PIQA
python prompteval/plot_bai_regret.py --bench GSM8K --cost-aware
```

Figures go to `prompteval/outputs/bai_regret_plots/`. Use `--per-task` only if you want one panel per data seed.

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
