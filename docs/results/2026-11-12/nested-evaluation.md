# Nested evaluation: conditional model-based estimates

Questions: 500; outer folds: 5; inner folds: 3; B=0 cap: 2048.
Features: available.

Policies, layers and lambda are selected without outer-test outcomes. Fixed-cost policies freeze the lambda selected at the reference load; load-aware policies select it separately for each background load. Both use the same training-estimated token costs. This isolates load-dependent operating-point selection, not a learned nonlinear per-action GPU cost.

**These are offline expected accuracies and predicted background latencies, not live routing gains or correct answers per GPU-second.** Mixture behavior and corrected-cap request lengths need live validation. The empirical trace pool supplies labels, not ground-truth future success probabilities.

Pooled p95 applies the curve to all outer-fold mean tokens. Fold failures remain visible: V = violation, O = outside calibration domain, U = unavailable.

| SLO | background load | policy | accuracy | mean tokens | pooled predicted p95 | status | folds V/O/U |
|---|---|---|---|---|---|---|---|
| 20.7 | 0.5 | fixed | 0.771 | 2907.171 | 18.911 | predicted_feasible | 0/0/0 |
| 20.7 | 0.5 | fixed-mix | 0.770 | 3025.922 | 19.125 | predicted_feasible | 0/0/0 |
| 20.7 | 0.5 | length-fixed-cost | 0.779 | 2285.024 | 17.901 | predicted_feasible | 0/0/0 |
| 20.7 | 0.5 | length-load-aware | 0.779 | 2285.024 | 17.901 | predicted_feasible | 0/0/0 |
| 20.7 | 0.5 | level-fixed-cost | 0.784 | 3356.528 | 19.722 | pooled_feasible_with_fold_failures | 1/0/0 |
| 20.7 | 0.5 | level-load-aware | 0.784 | 3356.528 | 19.722 | pooled_feasible_with_fold_failures | 1/0/0 |
| 20.7 | 0.5 | probe-fixed-cost | 0.782 | 3688.208 | 20.438 | pooled_feasible_with_fold_failures | 2/0/0 |
| 20.7 | 0.5 | probe-load-aware | 0.782 | 3688.208 | 20.438 | pooled_feasible_with_fold_failures | 2/0/0 |
| 20.7 | 0.8 | fixed | — | — | — | unavailable_in_some_folds | 0/0/5 |
| 20.7 | 0.8 | fixed-mix | — | — | — | unavailable_in_some_folds | 0/0/5 |
| 20.7 | 0.8 | length-fixed-cost | 0.779 | 2285.024 | 23.909 | predicted_slo_violation | 5/0/0 |
| 20.7 | 0.8 | length-load-aware | — | — | — | unavailable_in_some_folds | 0/0/5 |
| 20.7 | 0.8 | level-fixed-cost | 0.784 | 3356.528 | 27.362 | predicted_slo_violation | 5/0/0 |
| 20.7 | 0.8 | level-load-aware | — | — | — | unavailable_in_some_folds | 0/0/5 |
| 20.7 | 0.8 | probe-fixed-cost | 0.782 | 3688.208 | 27.532 | predicted_slo_violation | 5/0/0 |
| 20.7 | 0.8 | probe-load-aware | — | — | — | unavailable_in_some_folds | 0/0/5 |
| 20.7 | 1.3 | fixed | — | — | — | unavailable_in_some_folds | 0/0/5 |
| 20.7 | 1.3 | fixed-mix | — | — | — | unavailable_in_some_folds | 0/0/5 |
| 20.7 | 1.3 | length-fixed-cost | 0.779 | 2285.024 | 61.523 | predicted_slo_violation | 5/0/0 |
| 20.7 | 1.3 | length-load-aware | — | — | — | unavailable_in_some_folds | 0/0/5 |
| 20.7 | 1.3 | level-fixed-cost | 0.784 | 3356.528 | 77.276 | predicted_slo_violation | 4/1/0 |
| 20.7 | 1.3 | level-load-aware | — | — | — | unavailable_in_some_folds | 0/0/5 |
| 20.7 | 1.3 | probe-fixed-cost | 0.782 | 3688.208 | 78.052 | predicted_slo_violation | 4/1/0 |
| 20.7 | 1.3 | probe-load-aware | — | — | — | unavailable_in_some_folds | 0/0/5 |
| 20.7 | 1.0 | fixed | — | — | — | unavailable_in_some_folds | 0/0/5 |
| 20.7 | 1.0 | fixed-mix | — | — | — | unavailable_in_some_folds | 0/0/5 |
| 20.7 | 1.0 | length-fixed-cost | 0.779 | 2285.024 | 29.502 | predicted_slo_violation | 5/0/0 |
| 20.7 | 1.0 | length-load-aware | — | — | — | unavailable_in_some_folds | 0/0/5 |
| 20.7 | 1.0 | level-fixed-cost | 0.784 | 3356.528 | 50.164 | predicted_slo_violation | 5/0/0 |
| 20.7 | 1.0 | level-load-aware | — | — | — | unavailable_in_some_folds | 0/0/5 |
| 20.7 | 1.0 | probe-fixed-cost | 0.782 | 3688.208 | 50.164 | predicted_slo_violation | 5/0/0 |
| 20.7 | 1.0 | probe-load-aware | — | — | — | unavailable_in_some_folds | 0/0/5 |
| 41.4 | 0.5 | fixed | 0.791 | 6844.725 | 22.991 | predicted_feasible | 0/0/0 |
| 41.4 | 0.5 | fixed-mix | 0.791 | 6844.725 | 22.991 | predicted_feasible | 0/0/0 |
| 41.4 | 0.5 | length-fixed-cost | 0.796 | 6406.738 | 21.206 | predicted_feasible | 0/0/0 |
| 41.4 | 0.5 | length-load-aware | 0.796 | 6406.738 | 21.206 | predicted_feasible | 0/0/0 |
| 41.4 | 0.5 | level-fixed-cost | 0.791 | 6038.002 | 21.178 | predicted_feasible | 0/0/0 |
| 41.4 | 0.5 | level-load-aware | 0.791 | 6038.002 | 21.178 | predicted_feasible | 0/0/0 |
| 41.4 | 0.5 | probe-fixed-cost | 0.787 | 4527.912 | 21.178 | predicted_feasible | 0/0/0 |
| 41.4 | 0.5 | probe-load-aware | 0.787 | 4527.912 | 21.178 | predicted_feasible | 0/0/0 |
| 41.4 | 0.8 | fixed | 0.771 | 2907.171 | 25.825 | predicted_feasible | 0/0/0 |
| 41.4 | 0.8 | fixed-mix | 0.786 | 5860.336 | 28.887 | predicted_feasible | 0/0/0 |
| 41.4 | 0.8 | length-fixed-cost | 0.796 | 6406.738 | 31.299 | pooled_feasible_with_fold_failures | 1/0/0 |
| 41.4 | 0.8 | length-load-aware | 0.796 | 6406.738 | 31.299 | pooled_feasible_with_fold_failures | 1/0/0 |
| 41.4 | 0.8 | level-fixed-cost | 0.791 | 6038.002 | 29.273 | pooled_feasible_with_fold_failures | 1/0/0 |
| 41.4 | 0.8 | level-load-aware | 0.791 | 6038.002 | 29.273 | pooled_feasible_with_fold_failures | 1/0/0 |
| 41.4 | 0.8 | probe-fixed-cost | 0.787 | 4527.912 | 28.022 | predicted_feasible | 0/0/0 |
| 41.4 | 0.8 | probe-load-aware | 0.787 | 4527.912 | 28.022 | predicted_feasible | 0/0/0 |
| 41.4 | 1.3 | fixed | — | — | — | unavailable_in_some_folds | 0/0/5 |
| 41.4 | 1.3 | fixed-mix | — | — | — | unavailable_in_some_folds | 0/0/5 |
| 41.4 | 1.3 | length-fixed-cost | 0.796 | 6406.738 | — | out_of_calibration_domain | 0/5/0 |
| 41.4 | 1.3 | length-load-aware | — | — | — | unavailable_in_some_folds | 0/0/5 |
| 41.4 | 1.3 | level-fixed-cost | 0.791 | 6038.002 | — | out_of_calibration_domain | 0/5/0 |
| 41.4 | 1.3 | level-load-aware | — | — | — | unavailable_in_some_folds | 0/0/5 |
| 41.4 | 1.3 | probe-fixed-cost | 0.787 | 4527.912 | — | out_of_calibration_domain | 2/3/0 |
| 41.4 | 1.3 | probe-load-aware | — | — | — | unavailable_in_some_folds | 0/0/5 |
| 41.4 | 1.0 | fixed | 0.766 | 1711.181 | 28.604 | predicted_feasible | 0/0/0 |
| 41.4 | 1.0 | fixed-mix | 0.769 | 2448.175 | 34.234 | pooled_feasible_with_fold_failures | 2/0/0 |
| 41.4 | 1.0 | length-fixed-cost | 0.796 | 6406.738 | 58.137 | predicted_slo_violation | 5/0/0 |
| 41.4 | 1.0 | length-load-aware | 0.777 | 2076.660 | 29.324 | predicted_feasible | 0/0/0 |
| 41.4 | 1.0 | level-fixed-cost | 0.791 | 6038.002 | 58.002 | predicted_slo_violation | 5/0/0 |
| 41.4 | 1.0 | level-load-aware | 0.775 | 2040.673 | 29.294 | predicted_feasible | 0/0/0 |
| 41.4 | 1.0 | probe-fixed-cost | 0.787 | 4527.912 | 52.137 | predicted_slo_violation | 5/0/0 |
| 41.4 | 1.0 | probe-load-aware | 0.777 | 2354.873 | 29.895 | pooled_feasible_with_fold_failures | 1/0/0 |

## Paired question bootstrap: prespecified baseline comparisons

Conditional on fitted folds, finite trace labels and the fitted cost curve. Does not cover model retraining, trace-generation or cost-calibration uncertainty. A positive accuracy difference is not an SLO-qualified win when either policy violates the constraint.

Each baseline is reported separately; none is chosen as best using test outcomes. Intervals are pointwise, not adjusted for multiple comparisons.

| SLO | load | policy | baseline | delta | 95% interval |
|---|---|---|---|---|---|
| 20.7 | 0.5 | length-load-aware | length-fixed-cost | +0.0000 | [+0.0000, +0.0000] |
| 20.7 | 0.5 | length-load-aware | fixed | +0.0083 | [-0.0101, +0.0271] |
| 20.7 | 0.5 | length-load-aware | fixed-mix | +0.0092 | [-0.0023, +0.0206] |
| 20.7 | 0.5 | level-load-aware | level-fixed-cost | +0.0000 | [+0.0000, +0.0000] |
| 20.7 | 0.5 | level-load-aware | fixed | +0.0131 | [-0.0038, +0.0306] |
| 20.7 | 0.5 | level-load-aware | fixed-mix | +0.0140 | [-0.0030, +0.0310] |
| 20.7 | 0.5 | probe-load-aware | probe-fixed-cost | +0.0000 | [+0.0000, +0.0000] |
| 20.7 | 0.5 | probe-load-aware | fixed | +0.0108 | [-0.0069, +0.0283] |
| 20.7 | 0.5 | probe-load-aware | fixed-mix | +0.0117 | [-0.0044, +0.0275] |
| 20.7 | 0.5 | probe-load-aware | level-load-aware | -0.0023 | [-0.0196, +0.0162] |
| 20.7 | 0.5 | probe-load-aware | length-load-aware | +0.0025 | [-0.0115, +0.0165] |
| 41.4 | 0.5 | length-load-aware | length-fixed-cost | +0.0000 | [+0.0000, +0.0000] |
| 41.4 | 0.5 | length-load-aware | fixed | +0.0052 | [-0.0071, +0.0167] |
| 41.4 | 0.5 | length-load-aware | fixed-mix | +0.0052 | [-0.0071, +0.0167] |
| 41.4 | 0.5 | level-load-aware | level-fixed-cost | +0.0000 | [+0.0000, +0.0000] |
| 41.4 | 0.5 | level-load-aware | fixed | +0.0005 | [-0.0105, +0.0122] |
| 41.4 | 0.5 | level-load-aware | fixed-mix | +0.0005 | [-0.0105, +0.0122] |
| 41.4 | 0.5 | probe-load-aware | probe-fixed-cost | +0.0000 | [+0.0000, +0.0000] |
| 41.4 | 0.5 | probe-load-aware | fixed | -0.0036 | [-0.0182, +0.0108] |
| 41.4 | 0.5 | probe-load-aware | fixed-mix | -0.0036 | [-0.0182, +0.0108] |
| 41.4 | 0.5 | probe-load-aware | level-load-aware | -0.0040 | [-0.0189, +0.0118] |
| 41.4 | 0.5 | probe-load-aware | length-load-aware | -0.0087 | [-0.0213, +0.0035] |
| 41.4 | 0.8 | length-load-aware | length-fixed-cost | +0.0000 | [+0.0000, +0.0000] |
| 41.4 | 0.8 | length-load-aware | fixed | +0.0250 | [+0.0047, +0.0458] |
| 41.4 | 0.8 | length-load-aware | fixed-mix | +0.0101 | [+0.0000, +0.0202] |
| 41.4 | 0.8 | level-load-aware | level-fixed-cost | +0.0000 | [+0.0000, +0.0000] |
| 41.4 | 0.8 | level-load-aware | fixed | +0.0203 | [+0.0004, +0.0417] |
| 41.4 | 0.8 | level-load-aware | fixed-mix | +0.0054 | [-0.0043, +0.0159] |
| 41.4 | 0.8 | probe-load-aware | probe-fixed-cost | +0.0000 | [+0.0000, +0.0000] |
| 41.4 | 0.8 | probe-load-aware | fixed | +0.0163 | [-0.0029, +0.0355] |
| 41.4 | 0.8 | probe-load-aware | fixed-mix | +0.0014 | [-0.0105, +0.0134] |
| 41.4 | 0.8 | probe-load-aware | level-load-aware | -0.0040 | [-0.0189, +0.0118] |
| 41.4 | 0.8 | probe-load-aware | length-load-aware | -0.0087 | [-0.0213, +0.0035] |
| 41.4 | 1.0 | length-load-aware | length-fixed-cost | -0.0189 | [-0.0283, -0.0093] |
| 41.4 | 1.0 | length-load-aware | fixed | +0.0110 | [-0.0004, +0.0228] |
| 41.4 | 1.0 | length-load-aware | fixed-mix | +0.0084 | [-0.0034, +0.0202] |
| 41.4 | 1.0 | level-load-aware | level-fixed-cost | -0.0165 | [-0.0287, -0.0049] |
| 41.4 | 1.0 | level-load-aware | fixed | +0.0087 | [-0.0065, +0.0237] |
| 41.4 | 1.0 | level-load-aware | fixed-mix | +0.0061 | [-0.0059, +0.0182] |
| 41.4 | 1.0 | probe-load-aware | probe-fixed-cost | -0.0101 | [-0.0189, -0.0011] |
| 41.4 | 1.0 | probe-load-aware | fixed | +0.0111 | [-0.0049, +0.0274] |
| 41.4 | 1.0 | probe-load-aware | fixed-mix | +0.0084 | [-0.0031, +0.0197] |
| 41.4 | 1.0 | probe-load-aware | level-load-aware | +0.0023 | [-0.0142, +0.0196] |
| 41.4 | 1.0 | probe-load-aware | length-load-aware | +0.0000 | [-0.0126, +0.0133] |

## Calibration limits

- Token-only surrogate; not validated for policy mixtures or new N values.
- Aggregate source logs cannot recover arrival drops or censored tail latencies.
- Excluding censored cells may bias high-load fits downward; real replay required.
- 3 censored/error cells excluded; calibrated offered-load range [0.5, 1.5].
- Per-fold choices/status and file hashes are in results.json. No infeasible policy is silently replaced.
- Widths beyond N=4, if explicitly requested, are extrapolations in action shape even within token-load range.
- Existing full-dataset explorations remain prior knowledge; nested evaluation removes computational selection leakage but does not make this a previously untouched benchmark.

## Empirical hindsight reference (not a deployable policy)

Uses test outcomes to choose actions and lambda on a finite grid. Not a certified upper bound; excluded from policy-selection and paired baseline comparisons.

| SLO | load | accuracy | status |
|---|---|---|---|
| 20.7 | 0.5 | 0.848 | hindsight_reference |
| 20.7 | 0.8 | — | no_feasible_hindsight_point |
| 20.7 | 1.0 | — | no_feasible_hindsight_point |
| 20.7 | 1.3 | — | no_feasible_hindsight_point |
| 41.4 | 0.5 | 0.848 | hindsight_reference |
| 41.4 | 0.8 | 0.848 | hindsight_reference |
| 41.4 | 1.0 | 0.848 | hindsight_reference |
| 41.4 | 1.3 | — | no_feasible_hindsight_point |
