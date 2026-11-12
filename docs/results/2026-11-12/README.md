# November 12 progress: evaluation repair and matched replay

## What was actually run

- Read committed 500-question/16-trace labels and cap-check labels from source revision `6a8ecec`.
- Regenerated all 500 questions' hidden states (8 layers × 2048 features) on a free Colab Tesla T4, FP16 eager attention. The original feature archive was not shared in Discord or committed; the lean archive's log confirms it existed on the author's machine. New features are a distinct extraction, not a reproduction of BF16 numerical outputs. See `feature-manifest.json`.
- Ran outer five-fold / inner three-fold evaluation using N={1,2,4}, B={0,1024,2048}, corrected B=0 cap2048, seed20261112. `nested-evaluation.md` and `nested-summary.json` contain results. Full question-level audit is local at `evaluation-results/probe-t4-cap2048-reviewed/results.json`.
- CPU and fake-HTTP test suite: 20 tests passed. Fake-server tests establish software plumbing, never GPU performance.

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

## Real T4 matched pilot (completed)

See `t4-matched.md` and `t4-matched-summary.json`. One seed, two background rates, three actions plus a background-only control: all eight cells completed. Rates were 0.5/0.8 times a local capacity proxy of 0.292 reference requests/s. The locally calibrated 92-second SLO is a pilot heuristic, not the original 41.4-second SLO or a user requirement. Timeout/drain were corrected to97s before the completed comparison; the earlier interrupted run remains diagnostic evidence.

- Each action had only **one measured target job** at each load. Background cohorts had9 and13 requests; nearest-rank p95 is therefore the maximum, not a stable tail estimate.
- All background requests completed within92s. At load0.8, background p95 increased from41.52s(control) to45.77s(1,2048),47.98s(4,0),59.42s(4,2048).
- The single (4,2048) target took79.73s at low load and93.72s at high load, crossing the92s threshold. The other two targets remained below92s. This is a pilot observation, not population-level feasibility evidence.
- Measured-window waiting-queue samples were zero in every cell. These results do not establish queue saturation or a KV-cache explanation. The load proxy is not a proven sustained capacity boundary.
- Candidate order was randomized within blocks, but control was always first. One seed and small cohorts leave sampling and temporal drift unresolved. Synthetic forced-length completions do not measure answer quality or routing gains.

Raw archives are retained locally under `evaluation-resources/replay-runs/`; hashes, calibration, hardware and exact server command are included beside this report. Features use the separate T4 extraction manifest. No paid compute was used. Colab was released at2026-11-12 21:28:45UTC (16:28:45 New York), about49minutes after allocation, and its Reconnect T4 state was verified.

## Next experiment

Use longer arrival windows and several paired seeds to obtain enough target jobs and stable background tails, retaining the same controls and counting all arrivals. Then validate a policy comparison with actual generated answers and aggregation, router overhead, measured GPU wall time and SLO attainment. Compare against fixed action, fixed mixture and the same predictor with a frozen cost policy. The present offline token curve's worst-case error must be addressed before treating its SLO predictions as reliable.
