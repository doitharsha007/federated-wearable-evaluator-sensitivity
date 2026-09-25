# UCI HAR evaluator sensitivity — frozen study v1
Independent local PyTorch fixed-update experiment. No prior experimental artifacts imported.
Flower is intentionally unnecessary for a local four-client coalition game.

## Setup and commands
```
python -m venv .venv
.venv/bin/pip install torch==2.14.0+cpu --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e .
mkdir -p data
curl --fail -L 'https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip' -o data/uci_har.zip
.venv/bin/python -m pytest -q
.venv/bin/python -m har_eval.run --seed 4101 --out results/pilot_4101_v1
.venv/bin/python -m har_eval.audit results/pilot_4101_v1 data/uci_har.zip
```
Run command requires a clean committed source tree and refuses existing output directories.
Only seed 4101 is currently authorized; other configured seeds need approval.

## Protocol
Official training subjects sorted: validation positions 1,5,9,13,17 = 1,7,15,21,26.
Remaining subjects form four prespecified cohorts in configs/study_v1.yaml.
Whole-subject split; train-only channel normalization; official test unchanged.
Labels 0–5: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING.
Channels ordered body_acc xyz, body_gyro xyz, total_acc xyz; 128-sample publisher windows.
No test-based model selection. Twenty full-participation FedAvg warmup rounds;
SGD .03, batch64, five minibatch steps, no momentum/decay. Each snapshot starts from the same
final base. Independent SHA256-derived streams for initialization and each client invocation.
Mini-batches visit a shuffled permutation and restart only on exhaustion; short last batches retained.
Conv9→16→32 (kernel5,pad2,ReLU), mean/max pooling, linear64→6.

Coalition model = base + data-size weighted mean of fixed updates in coalition.
Empty model = base. Each of 16 masks uses ascending cohort subject bit order.
Utility = evaluator-weighted base CE minus coalition CE, float64 from saved float32 logits.
Evaluator weights: 1/M; 1/(6 M_class); 1/(5 M_subject).
Exact Shapley uses subset factorial weights; verifier independently averages 24 permutations.
LOO = grand utility minus utility without client; data-size = client count / cohort count.
Rewards = positive part normalized, uniform if all nonpositive (flagged); signed scores retained.
Ranks preserve ties within 1e-8; tau-b undefined for all ties.

Test split is loaded only after immutable valuation_frozen.json is persisted. Data preparation
unit tests may inspect test overlap/metadata; no test performance informs training or valuation.
Five descriptive test models: base and four full-cohort aggregates.

## Analysis contract
Independent unit: seed (planned 4101–4110). Cohorts are repeated fixed blocks, not replicates.
Primary endpoints: mean-over-four-cohorts Shapley allocation TV, equal-class vs frequency and
 equal-subject vs frequency. Ten per-seed values, mean, median; 50,000 seed bootstrap draws,
seed91001, nominal97.5% intervals per contrast. No confirmatory p-values. Pilot not inferential.
All rank, LOO, data-size, test-performance comparisons exploratory. No seed/checkpoint selection.
No bootstrap/inferential analysis is run in this one-seed pilot.

## Integrity
Atomic immutable artifacts, fsync files and directories, finalized-file manifest, fresh-process
independent verifier, COMPLETE last. Failures preserved; no automatic retries. Manifest covers
raw outputs; completion authenticates manifest and audit. External stdout/stderr are packaged
separately since the process owns them until exit. Empty-coalition logits repeated per cohort
intentionally; 64 coalition artifacts,192 utilities,144 client scores per seed.
