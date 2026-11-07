PE-EmbPT interaction comparison | 2026-11-07
============================================

Start with experiments_summary.txt and figure/Figure3_PE_EmbPT_interaction.png.
The main figure compares GittinsEval-S, GittinsEval-G, and both balanced-question
PE-EmbPT designs. Figure3_PE_EmbPT_interaction_sampling.png adds the two
random-question PE-EmbPT variants. Both figures also have PDF copies.

Scope
-----
This is the second joint model-prompt PE-EmbPT experiment. Each MMLU subject
has 1,500 arms from 15 LLMs and 100 prompt templates. The new interaction
variants cover 57 subjects, 20 seeds per subject, and both samplers. The
figures use the 19 subjects and four cohorts from Figure 3. The all-subject
tables compare five BAI variants, including the arm-only logistic reference.

The concatenated PE-EmbPT design uses 25 prompt coefficients shared across
models and 14 reference-coded LLM offsets. The interaction design gives
each LLM its own block of 25 prompt coefficients, while retaining the same
14 offsets. All 100 prompts for an LLM use its block with their own stored
prompt coordinates. The design therefore has 389 arm coefficients
(15 * 25 + 14). Both versions add N question identity coefficients shared
across arms for a subject with N questions.

Both designs reuse the same pretrained DPR feature values and fit observed
binary outcomes with L2 LIBLINEAR at C=100. The Rasch fits use no separate
intercept and leave feature values at their original scale. The shared C
sets the penalty strength on coefficients, while the penalty on equivalent
prompt score functions depends on the design. Repeating one common prompt
coefficient vector in all 15 model blocks incurs 15 times its concatenated
prompt penalty. The comparison includes this regularization difference.

Files
-----
figure/
  Figure3_PE_EmbPT_interaction.png and .pdf compare the balanced designs.
  Figure3_PE_EmbPT_interaction_sampling.png and .pdf add random questions.

data/
  plotted_curves.csv contains the exact points used in both figures.
  checkpoints.csv and subject_checkpoints.csv give cohort and per-subject
  estimates at 2%, 4%, 6%, 8%, and 9.9% on the 19 figure subjects.
  all57_bai_checkpoints.csv and all57_bai_subject_checkpoints.csv give
  BAI-only estimates at 2%, 4%, 6%, 8%, and 10% on all 57 subjects.
  selected_tasks.csv defines the four Figure 3 cohorts.

supplementary/
  interaction_seed_traces.csv contains both interaction variants' five
  phases per seed, including both spend coordinates and raw regret.
  subject_metadata.csv gives subject dimensions, design hashes, and
  exhaustive denominators. report_inputs.json identifies the sources.
  runtime_diagnostics.csv records elapsed time, convergence warnings,
  and the checkpoint checksum for each of the 2,280 interaction runs.
  reference/ contains copies of the first bundle's portable BAI traces,
  subject metadata, model index, price weights, and Gittins caches.

protocol.json records the exact model, budget, and aggregation rules.
provenance/ records the execution, per-subject inputs, first-bundle
manifest, and original Gittins source. code/source_snapshot/ preserves
the execution and reporting source files, with hashes in
code/source_snapshot_manifest.csv. The licenses are in licenses/.
The root validation.json and validation/ record the verification checks.
FILE_MANIFEST.csv gives bundle file sizes and SHA-256 checksums.

Budget and score interpretation
-------------------------------
Both PE-EmbPT designs use the same five-phase successive-halving replay
at 2%, 4%, 6%, 8%, and 10% of matrix cells. The random sampler draws
uniformly from a chosen arm's unseen questions. The balanced sampler
draws among its least-covered unseen questions, using counts across
all arms. Each phase ranks active arms by the average of the observed
responses and the fitted probabilities for unseen responses.

The horizontal axis divides actual observations or price-weighted spend
by the corresponding exhaustive subject total. The BAI price-weighted
view applies fixed input-token quote weights to the same unit-budget
allocation histories. Gittins subject means are interpolated within
their full-20-run raw-budget support, ending around 9.97%-10%. BAI
points use the latest completed phase and hold the final recommendation
through 10%. At 9.9% unit cost the last available BAI phase is 8%.
The all57 table includes the distinct 10% endpoint.

Figure bands show one standard error across equally weighted subject
means. Cohort checkpoint cells require all 20 seeds for every subject.
Subject rows show available seed counts and paired BAI differences,
including interaction minus concatenated for each matched sampler.
Gittins caches support subject-level comparison. Raw per-seed Gittins
trajectories are outside the bundle scope.

Replot and verify
-----------------
Open a terminal in this folder and run these commands.

  python -m pip install -r requirements.txt
  python validate_bundle.py
  python replot.py

The replot reads the portable CSVs and writes both figures to reproduced/.
Both PNGs regenerated byte for byte with Python 3.12.14 and the pinned
dependencies. Use python replot.py --out-dir <folder> for another
destination. Replotting reuses recorded results. The runner and builder are in
the repository under experiments/mmlu_pe_embpt/run_interaction.py and
experiments/mmlu_pe_embpt/report_interaction.py. An experiment rerun needs
the repository's response matrices and stored prompt features. Full
response matrices, encoder weights, and PE-EmbFT are outside this bundle.
