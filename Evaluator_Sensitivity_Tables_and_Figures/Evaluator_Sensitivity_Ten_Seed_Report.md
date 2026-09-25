# Evaluator-sensitivity: audited ten-seed descriptive report

**10/10 seeds PASS.** Seeds 4102–4110 were executed once each in new directories.
The previously approved corrected pilot supplies seed4101; its earlier v1 copy is not counted.
There were no failed runs, retries, substitutions, or changes to the frozen experiment.
No significance, superiority, fairness, or incentive conclusion is made.

## Frozen release and execution

Tag `study-v1.1`; commit `84ed97a818c82c4472a540f00e3be1c60e265e8b`.
Source archive SHA-256: `21dc02c1b0491c9ae7e763d188107f392da85266e4a59c97a4958a172d043a0a`.
Dataset, dependency lock and installed versions, config and tracked source were checked before
launch and before each new run. All runs used the same corrected source hash.
Execution was serial, CPU, one thread, in immutable directories under
`har-evaluator-sensitivity-v2/results/`. Existing output directories were rejected. All 194 prior preservation hashes also remained unchanged.
The immutable README's earlier pilot-only authorization is superseded by the user's explicit
nine-seed approval; the document itself was not changed.

## Integrity audit

| Check | Observed audit result |
|---|---|
| Intended independent seeds | 4101–4110, exactly once |
| Coalition models reconstructed | 640/640 |
| Evaluator utilities reconstructed | 1920/1920 |
| Contribution scores reconstructed | 1440/1440 |
| Independent ranking calculations | 240/240 |
| Shapley efficiency | 120/120 games; max residual 1.39e-17 |
| Raw artifact manifest hashes | 1020/1020 |
| Descriptive test metrics reconstructed | 150/150 |
| File inventory, counts, finite values | PASS for every run |
| Completion ordering | Each pre-completion audit passed before its marker; marker/audit/manifest hashes checked again |
| Leakage controls | Train-only normalization; subject-disjoint train/validation/test; no cross-split duplicate windows; test loaded after valuation freeze |
| Paired evaluator inputs | Identical saved coalition logits and validation windows; only evaluator weights differ |
| Conditions across seeds | Identical split, normalization, validation index, config except seed, and environment hashes |
| Initializations | Ten distinct model hashes; deterministic streams keyed by seed and invocation |
| Source integrity | Unchanged frozen commit, source archive, config, dataset and dependency versions |
| Corrected pilot preservation | Original corrected-pilot hashes unchanged |

There are four fixed cohorts of four clients, three evaluators, sixteen coalitions per cohort,
and three scoring references. These repeated measurements are not independent replicates.
Full per-seed audits, external process exits, stdout/stderr, and the cross-seed audit are included.
The independent reconstruction uses saved evidence and a separate Shapley permutation formula;
it shares the model forward implementation and data loader with the runner, so it cannot rule
out every possible shared implementation error. Unit/regression tests supplement this check.

## Prespecified descriptive endpoints and uncertainty

Unit: **independent training seed, n=10**. First average allocation total variation (TV)
over the four fixed cohorts within each seed. Then summarize those ten values.
TV = half the sum of absolute differences in normalized positive-part score shares.
Signed scores are retained separately; all-nonpositive scores produce the flagged uniform fallback.

Bootstrap: percentile, 50,000 resamples of ten seeds with replacement, NumPy RNG seed91001.
The same index matrix is used for both evaluator contrasts. The two primary mean intervals
are nominal97.5% each, targeting approximately95% family coverage via Bonferroni. Bootstrap
coverage is approximate with ten seeds; these are not guarantees or hypothesis tests.
Median intervals at the same nominal level are available in the CSV as descriptive estimates,
not an additional confirmatory family. No p-values are computed. A separate verification reconstructed all 20 primary seed-level
values directly from saved allocation shares, reproduced the 500,000 bootstrap indices, and
checked all 12 reported primary summary fields. All raw analysis-input hashes still matched.

| Evaluator contrast (Shapley) | Mean TV | 97.5% bootstrap interval for mean | Median TV |
|---|---:|---:|---:|
| equal class vs sample frequency | 0.120866 | [0.104623, 0.139118] | 0.114539 |
| equal subject vs sample frequency | 0.009466 | [0.008006, 0.010998] | 0.009475 |

| Seed | Equal-class vs sample-frequency | Equal-subject vs sample-frequency |
|---|---:|---:|
| 4101 | 0.112687 | 0.010687 |
| 4102 | 0.100605 | 0.011415 |
| 4103 | 0.149963 | 0.008116 |
| 4104 | 0.116391 | 0.006856 |
| 4105 | 0.110145 | 0.009945 |
| 4106 | 0.101584 | 0.010809 |
| 4107 | 0.085805 | 0.008142 |
| 4108 | 0.166881 | 0.013511 |
| 4109 | 0.147437 | 0.006176 |
| 4110 | 0.117159 | 0.009005 |

These summaries are conditional on UCI HAR, the fixed subject split and cohorts, the fixed
base-training protocol, and this one-round fixed-update game. They do not estimate variability
from choosing other subjects, cohorts, datasets, architectures, or training budgets.

## Exploratory diagnostics

The following fractions are first calculated within seed and then averaged across ten seeds.
They disclose the contribution of signed-score clipping and uniform fallback to allocation TV.

| Evaluator | Scoring reference | Mean negative-score fraction | Mean uniform-fallback cohort fraction |
|---|---|---:|---:|
| sample_frequency | shapley | 0.0750 | 0.0000 |
| sample_frequency | leave_one_out | 0.3750 | 0.0000 |
| sample_frequency | data_size | 0.0000 | 0.0000 |
| equal_class | shapley | 0.0688 | 0.0000 |
| equal_class | leave_one_out | 0.3812 | 0.0000 |
| equal_class | data_size | 0.0000 | 0.0000 |
| equal_subject | shapley | 0.0938 | 0.0000 |
| equal_subject | leave_one_out | 0.3688 | 0.0000 |
| equal_subject | data_size | 0.0000 | 0.0000 |

All client scores and shares, rank reversals/ties/top-set overlaps, LOO/data-size disagreement
with Shapley, and test metrics are exported without filtering for favorable observations.
Secondary metrics and figures are descriptive/exploratory; no interval or inferential status
is implied for the rank heatmap or scoring-reference scatterplot.
The official test set is used only for five descriptive model evaluations per seed, not to
choose evaluators, update steps, cohorts, checkpoints, hyperparameters, or stopping rules.

## Figures and tables

- `analysis/primary_seed_intervals.png` and `.pdf`: all ten seed values, means, and primary intervals.
- `analysis/rank_reversal_descriptive.png` and `.pdf`: cohort-specific mean pair-reversal fractions.
- `analysis/scoring_reference_disagreement.png` and `.pdf`: all seed-level LOO/data-size allocation disagreements with Shapley.
- `analysis/primary_per_seed.csv`, `primary_summary.csv`: primary values and uncertainty.
- `analysis/secondary_per_seed.csv`, `score_diagnostics_per_seed.csv`, `score_diagnostics_summary.csv`.
- `analysis/scoring_reference_disagreement_per_seed.csv`.
- `analysis/test_descriptive_per_seed.csv`, `test_descriptive_summary.csv`.
- `analysis/cohort_counts.csv`, `runtime_storage.csv`.
- `analysis/all_client_scores.csv`, `all_reward_shares.csv`, `all_rank_comparisons.csv`.
- `analysis/bootstrap_indices.npy`, `analysis_specification.json`, `input_hashes.json`.

New-run execution, verification, and full cross-seed audit took 292.7 seconds.
All ten raw run directories occupy 32.48 MiB (file bytes,
excluding filesystem allocation overhead). Timing includes local execution conditions and is
not a benchmark claim. The existing pilot was audited, not retrained, in this batch.

## Limits and approval boundary

Only ten seeds and one dataset; fixed five-subject validation set; four-client games; one-round
fixed updates rather than retrained coalitions or lifetime client contribution. CE utility,
normalization, clipping, and ties are explicit design choices. Exact Shapley removes coalition
sampling approximation for this game; it does not establish a uniquely correct contribution
value. No behavioral incentives, fairness guarantees, privacy mechanisms, or deployment
properties were tested. No result-driven experiment changes were made.

Execution and descriptive analysis are complete. The results remain subject to the user's
review; no paper drafting or significance, superiority, fairness, or incentive claim is authorized.

## Reproduction

From the package root, with the frozen project dependencies and the recorded analysis
versions available:

```bash
# Read-only full audit for a run (repeat for the ten listed directories):
cd har-evaluator-sensitivity-v2
PYTHONPATH=src python -m har_eval.audit results/seed_4102_study_v1_1 data/uci_har.zip
cd ..
# Regenerate descriptive outputs into a NEW directory; no training:
python evaluator-ten-seed-v1/analyze.py --repository har-evaluator-sensitivity-v2 \
  --audit evaluator-ten-seed-v1/cross_seed_audit.json --out regenerated-analysis
```

The analysis rejects missing completion markers, hash-invalid artifacts, incorrect source/seed,
and an incomplete cross-seed audit. The analysis script and bootstrap draws are saved.
Do not rerun the execution driver: these seeds have already completed and new training needs approval.
