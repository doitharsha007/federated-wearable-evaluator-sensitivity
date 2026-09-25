# Frozen evaluator-sensitivity protocol, September 25, 2026
The complete fixed protocol is in README.md and configs/study_v1.yaml.
Pilot authorization is limited to seed4101. No previous principal artifacts may be used.
Primary contrast is evaluator choice conditional on identical models and validation windows.
This is an empirical evaluation, not an algorithmic novelty claim.
All sixteen coalitions, all four cohorts, all three evaluators, all three scoring references
must be logged regardless of outcome. No checkpoint, client, seed or score selection.
Unit of future statistical replication is independently trained seed. The pilot supports no
confidence interval, significance or superiority claim. Exact Shapley is exact for the
specified one-round fixed-update game, not for retrained coalitions or lifetime data value.
No subject-level generalization beyond this fixed split is inferred.
A failed pilot is preserved and requires disclosure and approval before another experiment.

Implementation details fixed before pilot: CPU deterministic operations, one thread;
normalization in float64 then model tensors float32; CE calculations for valuation float64;
no batch normalization/dropout; local optimizer freshly initialized each invocation.
Data are retrieved fresh from UCI; both container ZIP and source commit hashed.
Actual source tree and dependency lock committed before pilot execution.
