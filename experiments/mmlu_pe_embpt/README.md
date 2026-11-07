# MMLU PE-EmbPT interaction experiment

This experiment compares a model-specific prompt effect with the completed concatenation version. For each MMLU subject, the 1,500 candidates are 15 LLMs crossed with 100 prompt templates. The interaction design assigns each LLM its own 25 prompt coefficients and retains 14 reference-coded model offsets, giving 389 arm coefficients. Question identity features add one shared effect per question.

The response matrices and pretrained prompt features are already recorded. Running this experiment fits local regressions and replays evaluations; it makes no model API calls and does not train or download an encoder.

## Controlled comparison

Both the random-question and balanced-question variants retain the first version's native ExtendedRasch estimator, observed-plus-imputed row scores, L2 LIBLINEAR settings (`C=100`, no separate intercept), and five successive-halving phases at 2%, 4%, 6%, 8%, and 10% of exhaustive observations. The interaction features use the same cached 25-dimensional prompt coordinates without rescaling. Model 0 has no separate model offset. Question effects are shared across the candidate rows.

The interaction blocks belong to LLMs. All prompts for one LLM activate that LLM's block with their respective prompt embeddings. This permits different prompt effects across LLMs while preserving generalization across prompts within a model. It is an adaptation of PromptEval to joint model-prompt selection.

Holding `C=100` fixed compares the complete parameterizations. It does not match their penalty on every equivalent prompt score function: repeating the same prompt coefficient vector in 15 model blocks incurs 15 times its concatenated prompt-term L2 penalty. The comparison includes this regularization difference.

## Environment and execution

The experiment uses Python 3.12.14 and the pinned dependencies below. From the repository root:

```sh
python -m pip install -r experiments/mmlu_pe_embpt/requirements.txt
python experiments/mmlu_pe_embpt/run_interaction.py --all-subjects --workers 12 --seed-chunk-size 2
```

The default output is `outputs/mmlu_rasch389/results/`, separate from the first version. Per-trajectory checkpoints support resuming the same configuration. Use a new output directory if the implementation or input configuration changes. Each subject has 20 seeds for each of `rasch389_random` and `rasch389_balanced`. The seed chunks distribute large subjects across workers. Reduce `--workers` to fit the available CPU and memory; the per-trajectory seeds and estimator remain unchanged.

Before a full run, check native-estimator parity on the pilot:

```sh
python experiments/mmlu_pe_embpt/audit/validate_native.py
```

After all trajectories finish, audit the archives and build the comparison:

```sh
python experiments/mmlu_pe_embpt/audit/validate_archives.py --out outputs/mmlu_rasch389/audit/archive_validation.json
python experiments/mmlu_pe_embpt/report_interaction.py --out-dir outputs/mmlu_rasch389/report
```

The report builder reads the new archives together with the first version's frozen CSVs and Gittins caches. The command above writes fresh figures, phase-level data, and checkpoint tables to `outputs/mmlu_rasch389/report/`. This keeps a rerun separate from the published, checksummed bundle in `PE_EmbPT_interaction_data_2026-11-07/`. Regenerate figures from that published bundle with its `replot.py`; use its `validate_bundle.py` to check the published records.

The first version's frozen comparison data are in [PE_EmbPT_data_2026-11-07](../../PE_EmbPT_data_2026-11-07/). The [interaction experiment report](../../docs/mmlu_pe_embpt_interaction.md) describes both designs, budget interpretation, and reported findings. The [first-version report](../../docs/mmlu_pe_embpt_experiments.md) documents the preceding question-effect experiment. Gittins comparisons cover the 19 Figure 3 subjects; the BAI experiments cover all 57 MMLU subjects.
