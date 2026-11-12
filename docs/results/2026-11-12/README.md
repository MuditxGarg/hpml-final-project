# November 12: nested evaluation results (A)

## What was actually run

- Read committed 500-question/16-trace labels and cap-check labels from source revision `6a8ecec`.
- Regenerated all 500 questions' hidden states (8 layers × 2048 features) on a free Colab Tesla T4, FP16 eager attention. The original feature archive was not shared in Discord or committed; the lean archive's log confirms it existed on the author's machine. New features are a distinct extraction, not a reproduction of BF16 numerical outputs. See `feature-manifest.json`.
- Ran outer five-fold / inner three-fold evaluation using N={1,2,4}, B={0,1024,2048}, corrected B=0 cap2048, seed20261112. `nested-evaluation.md` and `nested-summary.json` contain results. Full question-level audit is included as `nested-results.json.gz`.
- A-specific tests cover held-out invariance, policy selection, voting, cost-domain limits, paired uncertainty and fold feasibility.

## Main observation

At the original 41.4-second background SLO:

| Background load | Selected fixed accuracy | Probe load-aware accuracy | Probe minus fixed, percentage points (pointwise 95% CI) |
|---|---:|---:|---:|
| 0.5 | 79.10% | 78.74% | -0.36 [-1.82, +1.08] |
| 0.8 | 77.11% | 78.74% | +1.63 [-0.29, +3.55] |
| 1.0 | 76.63% | 77.73% | +1.11 [-0.49, +2.74] |

All three intervals overlap zero. At load1.0 the adaptive probe also has one outer fold with a predicted SLO violation, despite a feasible pooled estimate. The fixed-cost probe (lambda frozen from load0.5) keeps 78.74% accuracy but predicts p95=52.14s, violating41.4s. The adaptive probe sacrifices some expected accuracy to reduce predicted p95; this is not proof of a superior live service.

The fixed-mixture baseline is stronger than a single fixed action: at load0.8 its accuracy is78.60%, only0.14 percentage points below the probe, with interval[-1.05,+1.34]. Do not compare only to the weaker fixed-action baseline.

## Cost-model check

Leave-one-action-out checking of the token-load isotonic curve gives MAE3.55s over37 in-domain aggregate cells; two cells have no in-domain prediction. The consequential worst case is action(4,2048) at load0.8: observed p95=61.77s versus predicted35.75s when that action is withheld (26.02s error). Average error alone obscures the failure near the proposed SLO. Details: `cost-diagnostics.json`.

This is a separate diagnostic, conditional on the old sweep and excluding censored/error cells. It does not validate transferred T4 latency or policy mixtures. The new replay must calibrate on its own hardware.

## Reproduction and review

The full fold/question audit is committed as `nested-results.json.gz` (gzip-compressed JSON). It includes source/code hashes, software versions, train/test IDs, selected parameters and per-question outcomes. The report and summary are readable without decompressing it.

See [evaluation instructions](../../evaluation.md) for commands. The original labels and sweep remain in the repository. The regenerated hidden-state array is not bundled in Git; `feature-manifest.json` records its hash, and `colab_recover_features.py` provides the extraction script. Reproduction requires this array or a new extraction explicitly identified as such.

This PR contains A only. Real-server experiments and online routing remain separate work. Before claiming live gains, validate the cost model near the SLO boundary and compare policies with actual answers, router overhead, GPU wall time and repeated matched traffic.
