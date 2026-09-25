# Submission companion

**Evaluator Sensitivity of Client Contribution Scoring in Federated Wearable Learning**

Status: complete 10-page IEEE two-column draft, including references. This is an author-review draft, not a submission approval. No further experiments were run and no validated study files were modified during drafting.

## Abstract

Client contribution scores depend on the evaluation objective used to assign utility to a coalition. We study how this choice affects allocation while holding client updates fixed in a wearable activity-recognition setting. Using UCI HAR, we construct validation data exclusively from official training subjects, retain the official test split, and prespecify four cohorts of four clients. For each of ten independent training seeds, one-round client-update snapshots define all sixteen coalitions in each cohort. We compute exact Shapley values, leave-one-out scores, and data-size scores under sample-frequency, equal-class, and equal-subject evaluators. Allocation is defined by normalizing positive scores, with an explicit uniform fallback. Relative to sample-frequency evaluation, the mean within-seed, across-cohort total-variation distance between Shapley allocations is 0.12087 for equal-class evaluation and 0.00947 for equal-subject evaluation. Seed-level percentile bootstrap intervals at nominal 97.5% are [0.10462, 0.13912] and [0.00801, 0.01100], respectively. The audit reconstructs 1,920 coalition utilities, 1,440 scores, and 240 ranking records and verifies 1,020 artifact hashes. The evidence supports a narrow conclusion: evaluator composition can materially change Shapley-based allocation in this fixed UCI HAR one-round setting. We make no claim of statistical significance, method superiority, general fairness, behavioral incentives, privacy guarantees, or generalization beyond this design.

## Result interpretation and scope

The independent unit is the training seed (4101–4110), conditional on the fixed dataset, validation subjects, four cohorts, architecture, and update protocol. Coalitions, clients, evaluators, and checkpoints are not independent replicates.

For a signed score vector z, allocation is r_i = max(z_i,0) / sum_j max(z_j,0); if the denominator is zero, r_i = 1/4. Total variation between two allocations is TV(r,r′) = (1/2) sum_i |r_i−r′_i|. It ranges from zero to one and is the mass that must be reassigned to transform one unit allocation into the other. This is not TV between raw scores, labels, or predictive distributions.

The mean TV is first calculated over four cohorts within each seed, then averaged over ten seeds. Equal-class versus sample-frequency: 0.1208657023, nominal 97.5% seed-bootstrap interval [0.1046234724, 0.1391175411]. Equal-subject versus sample-frequency: 0.0094661950, interval [0.0080061197, 0.0109981939]. Intervals use the saved 50,000 seed-resampling draws; no new resampling or statistical tests were conducted for this draft.

The narrow conclusion is that evaluator composition can materially change Shapley-based allocation in this fixed UCI HAR one-round setting. “Materially” describes a nontrivial fraction of unit allocation, not an externally validated practical threshold. Neither the magnitude nor the intervals establish statistical significance, superiority, general fairness, behavioral incentives, privacy, or generalization. Exact coalition enumeration is exact for this fixed-update game, not for coalition retraining or lifetime data value.

## Figures and tables

All figures are copied unchanged from the audited descriptive-analysis package. No new experimental data or illustrative synthetic results are plotted.

| Item | Content | Evidence/source |
|---|---|---|
| Figure 1 | Ten seed-level Shapley-allocation TV values and mean bootstrap intervals | primary_seed_intervals.pdf; primary_per_seed.csv; primary_summary.csv |
| Figure 2 | Exploratory rank reversals by fixed cohort, evaluator contrast, and scoring reference | rank_reversal_descriptive.pdf; all_rank_comparisons.csv |
| Figure 3 | Exploratory LOO/data-size allocation disagreement with exact Shapley | scoring_reference_disagreement.pdf; scoring_reference_disagreement_per_seed.csv |
| Table I | Scope relative to prior contribution-valuation work | Verified references [5], [7]–[10]; full-text flag for [10] |
| Table II | Four fixed cohorts, subject identifiers, and training counts | cohort_counts.csv; frozen split/configuration |
| Table III | Frozen data/model/update/uncertainty protocol | study-v1.1 configuration and source; analysis_specification.json |
| Table IV | Complete ten-seed audit counts and maximum efficiency residual | cross_seed_audit.json; seed audit records |
| Table V | Primary means, medians, SDs, and mean intervals | primary_summary.csv |
| Table VI | All ten primary seed observations | primary_per_seed.csv |
| Table VII | Negative-score fractions and zero fallback frequencies | score_diagnostics_summary.csv |

## Citation verification ledger

Verification performed September 25, 2026. “Verified” describes the stated scope, not an assertion that every cited article was exhaustively reviewed. No unverified bibliographic fields were invented.

| Paper reference | Verified source | Scope/status |
|---|---|---|
| [1] McMahan et al., FedAvg, AISTATS/PMLR 54:1273–1282 (2017) | https://proceedings.mlr.press/v54/mcmahan17a.html | Publisher metadata/abstract; standard model-averaging context |
| [2] Shapley, A Value for N-Person Games, RAND P-295 (1952), DOI 10.7249/P0295 | https://www.rand.org/pubs/papers/P295.html | RAND record; deliberately cites the 1952 report, not the 1953 book chapter |
| [3] Anguita et al., public-domain HAR dataset, ESANN (2013), pp. 437–442 | https://www.esann.org/sites/default/files/proceedings/legacy/es2013-84.pdf | Full publisher-hosted paper, authors/title/pages/dataset description |
| [4] Reyes-Ortiz et al., UCI HAR dataset (2013), DOI 10.24432/C54S4K | https://archive.ics.uci.edu/dataset/240/ | Official citation, dataset/split description and license listing; citation year differs from donation date |
| [5] Ghorbani and Zou, Data Shapley, PMLR 97:2242–2251 (2019) | https://proceedings.mlr.press/v97/ghorbani19c.html | Publisher metadata/abstract |
| [6] Ghorbani, Kim and Zou, distributional valuation, PMLR 119:3535–3544 (2020) | https://proceedings.mlr.press/v119/ghorbani20a.html | Publisher metadata/abstract; no transfer of its guarantees |
| [7] Liu et al., GTG-Shapley, arXiv:2109.02053 (2021) | https://arxiv.org/abs/2109.02053 | Verified arXiv metadata/abstract; source reconstruction/sampling context. Optional author action: verify final publisher version before replacing this valid preprint citation |
| [8] Chen et al., contribution-estimation evaluation, PVLDB 17(8):2077–2090 (2024), DOI 10.14778/3659437.3659459 | https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/FLCE_VLDB2024.pdf | Full author-hosted article; title/authors/venue/DOI and evaluation framing |
| [9] Tastan et al., Redefining Contributions, IJCAI (2024), pp. 5009–5017, DOI 10.24963/ijcai.2024/554 | https://www.ijcai.org/proceedings/2024/554 | Official publisher metadata/abstract; class-aware valuation/aggregation context |
| [10] Yang, Buyukates and Markopoulou, Rewarding the Rare, TMLR (Nov. 2025) | https://openreview.net/forum?id=JtybGfTUdq ; institutional verification: https://research.birmingham.ac.uk/en/publications/rewarding-the-rare-maverick-aware-shapley-valuation-in-federated-/ | **MANUAL FULL-TEXT CONFIRMATION REQUIRED.** Authors, title, venue, date, and validation-reweighting abstract verified through institutional record. OpenReview full text was not accessible. Detailed novelty comparison must be checked manually; draft makes only abstract-supported scope statements |
| [11] Efron, Bootstrap Methods, Annals of Statistics 7(1):1–26 (1979), DOI 10.1214/aos/1176344552 | https://projecteuclid.org/journals/annals-of-statistics/volume-7/issue-1/Bootstrap-Methods-Another-Look-at-the-Jackknife/10.1214/aos/1176344552.full | Publisher/search metadata verified; cited for resampling principle, not finite-sample coverage guarantee |

## Final submission checklist

### Completed in this draft

- [x] Exact requested title; full abstract and complete manuscript.
- [x] Ten pages, IEEEtran conference format, US Letter, two columns, references included, no appendix.
- [x] Three figures and seven tables, based on audited evidence or verified protocol/literature.
- [x] Precise game, evaluators, Shapley/LOO/data-size scores, positive-part allocation and TV definitions.
- [x] Seed is the independent unit; every primary seed value is reported.
- [x] Reported bootstrap intervals retain their original nominal level and conditional interpretation.
- [x] Audit counts: 1,920 utilities, 1,440 scores, 240 ranking records, 1,020 verified artifact hashes.
- [x] Fixed-cohort, one-dataset, one-round and fixed-update limitations; warmup influence and normalization caveats.
- [x] No significance, superiority, general fairness, incentive, privacy or broad-generalization claims.
- [x] All ten rendered pages visually checked; no LaTeX overflow or unresolved-reference warnings.
- [x] Validated source and existing study artifacts preserved; no new experiments.

### Author actions before submission

- [ ] Replace author placeholder with approved names, order, affiliations and contact information. The general conference CFP specifies single-blind review.
- [ ] Obtain all coauthors’ approval of the interpretation and complete manuscript.
- [ ] Read reference [10] in full and confirm the detailed overlap; retain conservative scope/novelty claims. Remove the editorial manual-review note only after this is done.
- [ ] Decide whether to replace [7] with a verified final publisher citation; the current arXiv citation is valid.
- [ ] Publish the approved reproducibility package at a stable public archival URL/DOI, respecting data licenses; replace the explicit pending-link sentence. Do not invent a URL or imply public availability before publication.
- [ ] Confirm dataset redistribution/license obligations and any institution-required ethics statement; no new human-subject collection is part of this study.
- [ ] Confirm conference/publisher rules on AI-assisted writing and any required disclosure; no policy text has been presumed.
- [ ] Recompile and recheck page count, all references and embedded fonts after author/URL edits. Do not reduce margins or fonts to fit changes.
- [ ] Verify portal-specific PDF compliance and any PDF eXpress requirement if instructed by the venue.
- [ ] Select “Special Session on Federated Learning on Big Data” in the submission system. Verify September 30, 2026 deadline and exact time zone directly in the portal.
- [ ] Confirm presenter registration/attendance obligations before submission.

Verified venue pages: https://bigdataieee.org/BigData2026/calls/papers/ (general format: up to ten pages including references, no appendix, single blind); https://bigdataieee.org/BigData2026/calls/special-federated-learning/ (special-session deadline September 30, 2026). The general-track deadline differs; do not substitute it for the special-session deadline.

## Editable package and build

The source package contains main.tex, IEEEtran.cls, the three unchanged figure PDFs, this companion, manuscript PDF, selected existing evidence tables, and drafting-verification records. It contains no newly executed study runs. Bibliography entries are in main.tex.

Build from the package directory with a TeX distribution containing the standard packages:

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Frozen study provenance: study-v1.1; commit 84ed97a818c82c4472a540f00e3be1c60e265e8b; source-archive SHA-256 21dc02c1b0491c9ae7e763d188107f392da85266e4a59c97a4958a172d043a0a. The paper sources are a separate drafting artifact, not a new study release.
