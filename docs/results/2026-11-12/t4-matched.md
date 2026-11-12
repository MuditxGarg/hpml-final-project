# Real T4 matched externality pilot

Run status: complete.
Completed blocks: 2; skipped blocks: 0.

Synthetic token-count replay; this does not measure final answer accuracy or routing gains. All scheduled cohort requests, including drops/errors/timeouts, count in SLO attainment. The p95 column is completed-only; it must not be read as an unconditional tail when failures exist.

Calibration: `{"reference_capacity_jobs_s": 0.29162730995343983, "slo_s": 92, "target_rate": 0.029162730995343986, "capacity_definition": "maximum error-free completed reference requests / total trial wall time including drain", "slo_definition": "ceil(4 * nearest-rank p95 of 5 isolated local B1024 requests); pilot heuristic, not SLA", "target_fraction": 0.1, "rho_values": [0.5, 0.8]}`

Protocol and correction provenance: `{"protocol_phase": "phase2_corrective_timeout_horizon", "corrective_reason": "timeout/drain must exceed frozen SLO; prior phase retained, not best-result selection", "request_timeout_s": 97, "drain_s": 97, "reused_source": {"path": "/content/t4-pilot-run1/calibration.json", "sha256": "6f8533243c0ba1a5b790b1ec0e5116b1b297cdf4aa42f5195d40e4570e531704", "reused_epoch": 1794518134.7179482, "selected_values_frozen": true}, "selected_actions": [[4, 0], [1, 2048], [4, 2048]], "omitted_default_actions": [[1, 0]]}`

Recovered partial blocks: 0. Partial cells are retained as incomplete evidence.

| seed | load | N | B | BG count | BG failures | BG p95 completed (s) | delta vs control (s) | BG SLO fraction | target count | target SLO fraction | target p95 completed (s) | max waiting |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 731 | 0.5 | 0 | 0 | 9 | 0 | 28.872 | 0.000 | 1.000 | 0 | — | — | 0.000 |
| 731 | 0.5 | 4 | 0 | 9 | 0 | 34.458 | 5.587 | 1.000 | 1 | 1.000 | 50.393 | 0.000 |
| 731 | 0.5 | 1 | 2048 | 9 | 0 | 32.605 | 3.734 | 1.000 | 1 | 1.000 | 53.548 | 0.000 |
| 731 | 0.5 | 4 | 2048 | 9 | 0 | 45.794 | 16.922 | 1.000 | 1 | 1.000 | 79.729 | 0.000 |
| 731 | 0.8 | 0 | 0 | 13 | 0 | 41.519 | 0.000 | 1.000 | 0 | — | — | 0.000 |
| 731 | 0.8 | 1 | 2048 | 13 | 0 | 45.774 | 4.254 | 1.000 | 1 | 1.000 | 68.253 | 0.000 |
| 731 | 0.8 | 4 | 2048 | 13 | 0 | 59.423 | 17.904 | 1.000 | 1 | 0.000 | 93.715 | 0.000 |
| 731 | 0.8 | 4 | 0 | 13 | 0 | 47.982 | 6.463 | 1.000 | 1 | 1.000 | 64.743 | 0.000 |

N=0 is a background-only control. Candidate order is randomized, but control runs first. These short runs and few seeds do not establish statistical significance; empty target cohorts provide no target-performance evidence. Capacity and SLO are local pilot heuristics, not imported RTX4080 values or a user SLA. Corrected B=0 cap2048 and single chat-template prefixes differ from the historical sweep. Compare actions within each paired block before comparing across loads.

Raw cohort events, telemetry and manifests remain in the downloaded run directory. An unavailable or skipped condition is not counted as a passing condition.
