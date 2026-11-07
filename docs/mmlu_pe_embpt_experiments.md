# MMLU PE-EmbPT experiments

At 10% of MMLU evaluation cells, balanced-question PE-EmbPT reduced mean simple regret from **0.02171928 to 0.00905031** and improved on the 39-feature logistic baseline in all 57 subjects. In the separate 19-subject Figure 3 comparison, GittinsEval-S had lower cohort mean regret than both PE-EmbPT variants at all 36 comparable checkpoints.

[Main figure (PNG)](../PE_EmbPT_data_2026-11-07/figure/Figure3_PE_EmbPT.png) · [Main figure (PDF)](../PE_EmbPT_data_2026-11-07/figure/Figure3_PE_EmbPT.pdf) · [Sampling comparison (PNG)](../PE_EmbPT_data_2026-11-07/figure/Figure3_PE_EmbPT_sampling.png) · [Sampling comparison (PDF)](../PE_EmbPT_data_2026-11-07/figure/Figure3_PE_EmbPT_sampling.pdf)

![GittinsEval and PE-EmbPT comparison across the four Figure 3 cohorts](../PE_EmbPT_data_2026-11-07/figure/Figure3_PE_EmbPT.png)

## What changed

The submitted Figure 3 described its baseline as PE-OneHot. This experiment adds PromptEval-style question effects to an arm-only logistic baseline that already uses pretrained prompt embeddings. All three BAI variants share the same 39 arm features. The first comparison changes the fitted model and recommendation score while retaining random-question sampling. The second holds the fitted model and recommendation score fixed and compares random with balanced questions.

Each MMLU subject has 1,500 arms from crossing 15 LLMs with 100 prompt templates. The arm design contains 25 PCA coordinates from pretrained DPR prompt embeddings and 14 model indicators, with model 0 as the reference. The prompt coordinates are shared across models for the same template. The Rasch variants add one question indicator shared across all 1,500 arms for each of the subject's $N$ questions.

| Method | Fitted design | Arm score | Question sampling |
| --- | --- | --- | --- |
| Logistic BAI (39) | 39 arm features and a constant. No question effects. | Fitted arm score | Uniform among the chosen arm's unseen questions |
| PE-EmbPT, random questions | 39 arm features and $N$ question effects. No separate intercept. | Full-row mean of observed outcomes and fitted probabilities on unseen questions | Same random-question rule as logistic BAI |
| PE-EmbPT, balanced questions | Same $39 + N$ design as random PE-EmbPT | Same full-row mean as random PE-EmbPT | Uniform among the chosen arm's least-covered unseen questions, counting observations across all arms, including eliminated arms |

All fits use observed binary outcomes and native L2 LIBLINEAR logistic regression with $C=100$. The Rasch logit is the sum of the fitted arm score and the fitted question coefficient. The full-row estimate retains each observed answer and fills only unseen entries with fitted probabilities.

The replay uses five successive-halving phases with cumulative observation budgets of 2%, 4%, 6%, 8%, and 10% of the response matrix. Each evaluation chooses among the least-sampled active arms. The active set sizes before elimination are 1,500, 750, 375, 187, and 93. Each phase reports the highest-scoring active arm. The upstream phase formula would give 11 phases on a 1,500-arm pool. Original PromptEval ranked prompts separately within each model and used the observed empirical mean among surviving prompts for its final recommendation. The results here describe the joint model-prompt adaptation. [Full protocol](../PE_EmbPT_data_2026-11-07/protocol.json)

### Embedding provenance

The replay uses saved upstream PromptEval feature arrays. The upstream code identifies the encoder as `sentence-transformers/facebook-dpr-question_encoder-multiset-base` and leaves the weight revision unpinned. It encodes unfilled prompt templates, with placeholders for the topic, question, and choices, and applies PCA within each subject. The cached arrays preserve the exact inputs to the experiment. The original encoder snapshot and feature-generation environment are unavailable. PE-EmbFT is outside the scope of this comparison. [Feature provenance](../PE_EmbPT_data_2026-11-07/provenance/feature_provenance.json)

## Results across all 57 subjects

Simple regret is the best arm's full-matrix mean minus the recommended arm's full-matrix mean. Each method uses 20 sampling seeds per subject. We average seeds within subjects and then weight subjects equally.

| Unit budget | Logistic BAI (39) | PE-EmbPT, random | PE-EmbPT, balanced |
| ---: | ---: | ---: | ---: |
| 2% | 0.02624004 | 0.02445305 | 0.02516509 |
| 4% | 0.02458780 | 0.02290746 | 0.02289494 |
| 6% | 0.02393083 | 0.02039984 | 0.02056374 |
| 8% | 0.02322175 | 0.01586805 | 0.01649863 |
| 10% | 0.02171928 | 0.00951678 | 0.00905031 |

At 10% unit cost, the paired balanced-minus-logistic difference was **−0.01266897** with a subject standard error of **0.00100077**. The corresponding balanced-minus-random difference was −0.00046647 with a subject standard error of 0.00030297. The random-question variant accounts for most of the gain over arm-only logistic BAI at this endpoint.

Because all three methods use the same 39 arm features, this comparison does not isolate the benefit of the 25 prompt embedding coordinates over the 14 model indicators. It measures the effect of adding question effects and changing the recommendation score, followed by a separate question-sampling comparison.

The price-weighted view applies each arm's model-level input-token price quote to the same BAI evaluations. This gives a relative spend measure while the allocator still budgets 10% of matrix cells. At 10% price-weighted spend, the means were 0.02177324 for logistic BAI, 0.00978393 for random PE-EmbPT, and 0.00943791 for balanced PE-EmbPT. Checkpoints hold the latest available recommendation.

[All-subject checkpoints](../PE_EmbPT_data_2026-11-07/data/all57_bai_checkpoints.csv) · [Per-subject checkpoints](../PE_EmbPT_data_2026-11-07/data/all57_bai_subject_checkpoints.csv) · [Seed-level phase records](../PE_EmbPT_data_2026-11-07/supplementary/pe_seed_traces.csv)

## Comparison with GittinsEval on Figure 3 subjects

The figure uses the original four cohorts with 6 Easy–Small, 2 Easy–Large, 9 Hard–Small, and 2 Hard–Large subjects. GittinsEval-S and GittinsEval-G come from the supplied per-subject LCB curve caches with 20 runs per subject, method, and cost mode. The curves use the dataset and general priors, respectively.

Each task's horizontal coordinate is actual cumulative observations or price-weighted spend divided by its exhaustive evaluation cost. This differs from the published Figure 3 grouped display, which normalized Gittins runs by their observed endpoints. Here Gittins subject means are interpolated within the raw-spend interval where all 20 runs contribute. Each cohort curve ends at the earliest full-support endpoint, approximately 9.97%–10%. The BAI curves use the latest completed phase and carry the final recommendation forward through 10%.

| Rasch variant | Lower than GittinsEval-S | Lower than GittinsEval-G |
| --- | ---: | ---: |
| Random questions | 0 of 36 | 4 of 36 |
| Balanced questions | 0 of 36 | 4 of 36 |

The 36 comparisons count cohort mean regret at the reported checkpoints with full coverage. The 2% price-weighted BAI cells lack full coverage and are blank. These counts describe the checkpoint means. The shaded bands give ±1 standard error across subject means, and the Gittins caches support subject-level comparisons. Per-seed Gittins significance tests and claims about the intervening parts of the curves are outside the scope of this report.

At unit cost, the 9.9% Figure 3 checkpoint still uses the 8% BAI recommendation because the final phase lands at exactly 10%. The all-subject BAI table includes that final phase.

[Cohort checkpoints](../PE_EmbPT_data_2026-11-07/data/checkpoints.csv) · [Figure 3 subjects](../PE_EmbPT_data_2026-11-07/data/selected_tasks.csv) · [Plotted curves](../PE_EmbPT_data_2026-11-07/data/plotted_curves.csv) · [Gittins source](../PE_EmbPT_data_2026-11-07/provenance/gittins_source.json)

## Replot and audit

Create the figures from the repository root with these commands.

```sh
python -m pip install -r PE_EmbPT_data_2026-11-07/requirements.txt
python PE_EmbPT_data_2026-11-07/validate_bundle.py
python PE_EmbPT_data_2026-11-07/replot.py
```

The validator checks file hashes and independently recomputes the BAI checkpoint tables from the phase records. The plotting script regenerates both figures in `PE_EmbPT_data_2026-11-07/reproduced/` from the bundled CSVs. The tested environment recreated both PNGs byte for byte.

The bundle contains phase-level BAI traces, Gittins subject-level curve caches, source provenance, and verification records. The audit checked all 2,280 Rasch trajectories and 11,400 Rasch phase records, and independently recomputed all report tables and plotted points. The source snapshots document the implementation. Running the experiments again requires the original repository and data, including the response matrices and saved feature inputs. The bundled Gittins data cover subject aggregates, run summaries, and stopping diagnostics.

[Bundle instructions](../PE_EmbPT_data_2026-11-07/README_文件说明.txt) · [Validation](../PE_EmbPT_data_2026-11-07/validation.json) · [Source snapshot manifest](../PE_EmbPT_data_2026-11-07/code/source_snapshot_manifest.csv)
