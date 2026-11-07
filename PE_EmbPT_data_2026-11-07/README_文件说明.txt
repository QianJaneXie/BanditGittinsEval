PE-EmbPT MMLU comparison | 2026-11-07
====================================

Start with figure/Figure3_PE_EmbPT.png and experiments_summary.txt. The
main figure compares GittinsEval-S, GittinsEval-G, the 39-feature logistic
baseline, and the balanced-question PE-EmbPT adaptation. The sampling
figure adds the random-question PE-EmbPT adaptation. PDF copies are included.

Scope
-----
The figures use the same 19 MMLU subjects and four cohorts as Figure 3.
The all-subject tables cover the three BAI variants on all 57 MMLU subjects.
Each subject and method has 20 seeds. Every BAI run ranks 1,500 model-prompt
pairs jointly, with 15 models and 100 prompts per model. The original
PromptEval experiment ranked prompts separately for each model.

Files
-----
figure/
  Figure3_PE_EmbPT.png and .pdf are the main comparison.
  Figure3_PE_EmbPT_sampling.png and .pdf add the random-question ablation.

data/
  plotted_curves.csv contains the exact points drawn in the two figures.
  checkpoints.csv gives cohort estimates at 2%, 4%, 6%, 8%, and 9.9%.
  subject_checkpoints.csv gives the corresponding per-subject estimates
  and contributing seed counts.
  all57_bai_checkpoints.csv and all57_bai_subject_checkpoints.csv give
  BAI-only estimates at 2%, 4%, 6%, 8%, and 10% across all 57 subjects.
  selected_tasks.csv defines the four Figure 3 cohorts.

supplementary/
  pe_seed_traces.csv contains the five recorded phases per BAI trajectory
  with both spend coordinates, chosen arm, oracle mean, and regret.
  subject_metadata.csv gives subject dimensions and budget denominators.
  model_index.csv maps model indices to names and marks the reference.
  gittins/per_task/ contains the supplied Gittins subject-level curve
  caches and metadata. gittins/runs_summary.csv and runs_stopping.csv
  give the 1,520 run summaries and stopping diagnostics for the 19 figure
  subjects.

protocol.json records the model, allocation, budget, and aggregation
rules. provenance/ records source identities. code/source_snapshot/
preserves reference copies of the experiment and reporting scripts.
Its file hashes are in code/source_snapshot_manifest.csv, and licenses/
contains both projects' licenses. Experiment reruns require the
original repository and source data.

The root validation.json summarizes the bundle checks. validation/
contains the detailed independent archive and report checks.
FILE_MANIFEST.csv records bundled file sizes and SHA-256 checksums.

Reading the comparison
----------------------
Simple regret is the best full-matrix arm mean minus the recommended
arm's full-matrix mean. The horizontal axis is actual cumulative
observations or price-weighted spend divided by the subject's exhaustive
cost. The earlier Figure 3 group cache used each Gittins run's endpoint
to normalize its horizontal axis. Here Gittins curves come from the
per-subject caches in raw spend units.

All BAI methods allocate a budget of 10% of matrix cells. The price-weighted
view reuses those evaluations and weights each cell by its model's quoted
rate per million input tokens. This is a relative accounting proxy.
The Gittins cost-aware runs come from the supplied Figure 3 export.

Gittins subject means are interpolated only where all 20 runs contribute.
Each Gittins curve ends at the earliest fully supported endpoint in its
cohort, about 9.97%-10%. BAI curves use the latest completed phase and
hold the last recommendation through 10%. Curves and cohort checkpoint
cells require all 20 seeds for every included subject. Subject tables
also show partial seed coverage. Blank fields indicate insufficient
coverage. The error bands show one standard error across equally weighted
subject means. Paired BAI contrasts pair seeds within each subject first.

At unit cost the last BAI phase lands at exactly 10%. The figure table's
9.9% checkpoint therefore still uses the 8% recommendation. Use the
all-subject BAI table for the separate 10% endpoint. The figures expand
the vertical display below 0.05. CSV values retain raw simple regret.

Replot
------
Unzip the bundle, open a terminal in its root directory, and run these
commands.

  python -m pip install -r requirements.txt
  python replot.py

To verify file checksums, phase records, and the BAI estimates and paired
contrasts in all four checkpoint tables, run:

  python validate_bundle.py

The validator uses only the Python standard library.

The standalone script reads the CSVs and redraws both figures in
reproduced/. With Python 3.12.14 and the pinned dependencies in
requirements.txt, the regenerated PNGs matched the bundled PNGs
byte for byte. The --out-dir option selects a different destination.
The bundle supports replotting and phase-level BAI analysis. Full
response matrices, trained encoder weights, Gittins per-seed
trajectories, and PE-EmbFT results are outside its scope.
