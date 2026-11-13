# Real T4 matched externality pilot

Run status: partial_deadline.
Completed blocks: 3; skipped blocks: 1.

Synthetic token-count replay; this does not measure final answer accuracy or routing gains. All scheduled cohort requests, including drops/errors/timeouts, count in SLO attainment. The p95 column is completed-only; it must not be read as an unconditional tail when failures exist.

Calibration: `{"reference_capacity_jobs_s": 0.39491025497464244, "slo_s": 87, "target_rate": 0.03949102549746425, "capacity_definition": "maximum error-free completed reference requests / total trial wall time including drain", "slo_definition": "ceil(4 * nearest-rank p95 of 5 isolated local B1024 requests); pilot heuristic, not SLA", "target_fraction": 0.1, "rho_values": [0.5, 0.8]}`

Protocol and correction provenance: `{"protocol_phase": "fresh_local_calibration", "corrective_reason": null, "request_timeout_s": 92, "drain_s": 92, "reused_source": null, "selected_actions": [[4, 0], [4, 2048]], "omitted_default_actions": [[1, 0], [1, 2048]], "measured_target_arrivals": {"definition": "conditional Poisson fixed count", "count": 6, "expected_rate": 0.05, "warmup_rate": 0.03949102549746425}, "configuration": {"url": "http://127.0.0.1:8001", "model": "m", "lengths": "lengths.json", "prompts": "prompts.json", "output": "/content/t4-b-repeat-run1", "reuse_calibration": null, "seeds": [20261113, 20261114], "loads": [0.5, 0.8], "actions": ["4:0", "4:2048"], "target_count": 6, "warmup": 30.0, "duration": 120.0, "idle_timeout": 15, "randomize_control": true, "plan_only": false}, "planned_schedule": {"blocks": [{"seed": 20261113, "rho": 0.5, "conservative_block_s": 831.0, "expected_target_jobs": 6, "expected_background_jobs": 23.694615298478546}, {"seed": 20261113, "rho": 0.8, "conservative_block_s": 831.0, "expected_target_jobs": 6, "expected_background_jobs": 37.911384477565676}, {"seed": 20261114, "rho": 0.8, "conservative_block_s": 831.0, "expected_target_jobs": 6, "expected_background_jobs": 37.911384477565676}, {"seed": 20261114, "rho": 0.5, "conservative_block_s": 831.0, "expected_target_jobs": 6, "expected_background_jobs": 23.694615298478546}], "conservative_total_s": 3384.0, "warmup_s": 30.0, "duration_s": 120.0, "horizon_s": 92, "note": "expected counts are not guarantees; calibration/setup time excluded"}}`

Recovered partial blocks: 0. Partial cells are retained as incomplete evidence.

| seed | load | N | B | BG count | BG failures | BG p95 completed (s) | delta vs control (s) | BG SLO fraction | target count | target SLO fraction | target p95 completed (s) | max waiting |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261113 | 0.5 | 4 | 0 | 26 | 0 | 54.866 | 15.458 | 1.000 | 6 | 0.833 | 50.847 | 0.000 |
| 20261113 | 0.5 | 0 | 0 | 26 | 0 | 39.407 | 0.000 | 1.000 | 0 | — | — | 0.000 |
| 20261113 | 0.5 | 4 | 2048 | 26 | 0 | 85.000 | 45.593 | 1.000 | 6 | 0.000 | — | 0.000 |
| 20261113 | 0.8 | 4 | 0 | 39 | 0 | 74.601 | 13.967 | 1.000 | 6 | 0.833 | 73.913 | 0.000 |
| 20261113 | 0.8 | 4 | 2048 | 39 | 9 | 91.782 | 31.148 | 0.231 | 6 | 0.000 | — | 0.000 |
| 20261113 | 0.8 | 0 | 0 | 39 | 0 | 60.634 | 0.000 | 1.000 | 0 | — | — | 0.000 |
| 20261114 | 0.8 | 4 | 0 | 30 | 0 | 74.201 | 30.074 | 1.000 | 6 | 0.333 | 90.774 | 0.000 |
| 20261114 | 0.8 | 4 | 2048 | 30 | 11 | 90.846 | 46.719 | 0.533 | 6 | 0.000 | — | 0.000 |
| 20261114 | 0.8 | 0 | 0 | 30 | 0 | 44.127 | 0.000 | 1.000 | 0 | — | — | 0.000 |

N=0 is a background-only control. All cells, including control, were randomized within blocks. These short runs and few seeds do not establish statistical significance; empty target cohorts provide no target-performance evidence. Capacity and SLO are local pilot heuristics, not imported RTX4080 values or a user SLA. Corrected B=0 cap2048 and single chat-template prefixes differ from the historical sweep. Compare actions within each paired block before comparing across loads.

Raw cohort events, telemetry and manifests remain in the downloaded run directory. An unavailable or skipped condition is not counted as a passing condition.

## Repeated paired comparisons

Independent repetitions are seeds, not individual HTTP requests. Partial blocks are excluded here. P95 differences exclude pairs with failed/censored background requests; SLO attainment still counts those misses. No confidence interval is inferred from the short pilot.

```json
[
  {
    "load": 0.5,
    "n": 4,
    "budget": 0,
    "independent_seeds": 1,
    "background_jobs": 26,
    "target_jobs": 6,
    "uncensored_p95_pairs": 1,
    "mean_p95_delta_s": 15.458104935692276,
    "p95_delta_range_s": [
      15.458104935692276,
      15.458104935692276
    ],
    "p95_delta_ci95": null,
    "mean_background_slo_delta": 0.0,
    "inference": "descriptive paired block effects only; no population CI or routing claim"
  },
  {
    "load": 0.5,
    "n": 4,
    "budget": 2048,
    "independent_seeds": 1,
    "background_jobs": 26,
    "target_jobs": 6,
    "uncensored_p95_pairs": 1,
    "mean_p95_delta_s": 45.59273277892254,
    "p95_delta_range_s": [
      45.59273277892254,
      45.59273277892254
    ],
    "p95_delta_ci95": null,
    "mean_background_slo_delta": 0.0,
    "inference": "descriptive paired block effects only; no population CI or routing claim"
  },
  {
    "load": 0.8,
    "n": 4,
    "budget": 0,
    "independent_seeds": 2,
    "background_jobs": 69,
    "target_jobs": 12,
    "uncensored_p95_pairs": 2,
    "mean_p95_delta_s": 22.020438005909853,
    "p95_delta_range_s": [
      13.966789429929719,
      30.074086581889986
    ],
    "p95_delta_ci95": null,
    "mean_background_slo_delta": 0.0,
    "inference": "descriptive paired block effects only; no population CI or routing claim"
  },
  {
    "load": 0.8,
    "n": 4,
    "budget": 2048,
    "independent_seeds": 2,
    "background_jobs": 69,
    "target_jobs": 12,
    "uncensored_p95_pairs": 0,
    "mean_p95_delta_s": null,
    "p95_delta_range_s": null,
    "p95_delta_ci95": null,
    "mean_background_slo_delta": -0.617948717948718,
    "inference": "descriptive paired block effects only; no population CI or routing claim"
  }
]
```
