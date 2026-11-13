# Matched Serving Interference Study — November 12, 2026

The conditional-count T4 pilot completed **three paired blocks / nine cells**: one low-load block and two high-load blocks. The second low-load block was skipped by the deadline guard. This is E1 synthetic serving externality evidence, not an online router or answer-quality experiment.

## Observed result

The background-only control and shallow action `(N=4, B=0)` met the background SLO in all three observed blocks. The deep action `(4,2048)` met it in the low-load block but failed it in both high-load blocks.

| Seed | Load fraction | Background jobs per cell | Control background SLO attainment | `(4,0)` background SLO attainment | `(4,2048)` background SLO attainment | Deep background timeouts |
|---|---|---|---|---|---|---|
| 20261113 | 0.5 | 26 | 100% | 100% | 100% | 0 |
| 20261113 | 0.8 | 39 | 100% | 100% | 23.1% | 9 |
| 20261114 | 0.8 | 30 | 100% | 100% | 53.3% | 11 |

The common background/target SLO was **87 seconds**, with a requested 95% attainment threshold. Low-load background p95 was39.41s for control,54.87s for shallow and85.00s for deep. High-load deep completed-only p95 was91.78s /90.85s, but these figures exclude9 /11 timeouts and must not be reported as unconditional tail latency.

This is an observed change in the **background-SLO feasible set** across the first seed's load pair, with a second high-load replication. It is not two complete replications of the load interaction: the second low-load condition was not run. No statistical-significance claim or confidence interval is made from one/two seeds.

## Target failures matter

Each action cell had6 measured target jobs, each requiring all4 samples to complete. Shallow target attainment was5/6,5/6 and2/6. The last shallow cell had3 completions, including one after the87s SLO, and3 timeouts. Deep target jobs timed out in **all18 cases across the three blocks**.

Therefore, neither action satisfied the runner's joint requirement that both background and target streams attain95% within87s. The background-only result above must not be relabeled as joint service feasibility. Real aggregation could potentially use partial samples; this forced-length replay does not evaluate such a policy or answer correctness.

## Protocol and provenance

- Fresh free Tesla T4 allocation; pinned Qwen3-1.7B revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, FP16, vLLM0.32.0. Exact commands and software versions are included.
- New local calibration: reference capacity proxy0.3949103 jobs/s, about398.87 tokens/s at32 concurrent reference requests. The proxy is a finite closed-loop calibration, not proof of sustained open-loop saturation. SLO87s was fixed before comparisons from4× the p95 of5 isolated reference requests; it is a pilot heuristic, not a user SLA.
- Background Poisson rates were0.197455 /0.315928 jobs/s. Each cell used30s warmup plus120s measured arrivals, with92s request timeout and drain. All scheduled requests remain in SLO denominators.
- Measured targets use **conditional Poisson arrivals: exactly6 jobs at uniformly random times within120s**, or0.05 jobs/s. Warmup targets retain the calibration Poisson rate0.039491 jobs/s. This design was fixed before comparison results: preflight showed a short unconditional Poisson cohort could be empty. Seeds were not selected by performance. Do not pool this workload with the earlier unconditional-Poisson pilot.
- All cells, including control, were randomized within each block. Prefix caching was disabled; idle checks separate cells. All9 raw event trajectories exactly match their persisted schedules, prompts and sampled lengths. See `trajectory-audit.json`.
- Corrected B0-cap2048 empirical length pools are reconstructible with `12_prepare_replay_lengths.py`; all24000 ordered values were verified against the run inputs. Prompts and lengths are independently sampled from marginal pools, so their question-level correlation is not preserved.
- No waiting requests were observed in the sampled telemetry. Maximum measured-window KV usage across cells was40.41%. These data do not demonstrate KV-cache exhaustion or establish a queuing mechanism.

## Artifacts and reproduction

- `report.md`, `summary.json`: generated per-cell and descriptive seed-level comparisons; partial/deadline status is explicit.
- `raw-run.zip`: full run state, traces, raw jobs, metrics and manifests; hash in `collection-status.json`.
- `calibration.json`, `hardware.json`, `server_command.json`, `driver_command.json`, `protocol.json`: actual configuration and provenance.
- `lengths-provenance.json`: source hashes and reconstruction method.
- `first-block-*`: interim evidence collected earlier; the final archive supersedes it for the complete run.
- [Runner instructions](../../matched-serving.md) cover generation, execution and reporting. The final driver hash matches the local tested driver. Full local suite:26 tests passed; independent code review found no blocker.

The free GPU was allocated at22:11:32UTC and released at23:02:26UTC (about51minutes), with disconnected UI verified. No paid compute was used. A brief Mac lock interrupted artifact retrieval but not the remote run; all final results were recovered.

## Next decision

Use this as evidence of load-sensitive background SLO impact, with target failures reported alongside it. Before E2, choose explicit background and target latency objectives, preserve timeout observations when fitting cost models, and allocate a longer run for several complete low/high seed pairs and the missing `(1,2048)` action. A live routing comparison must generate and score actual answers, include aggregation/router overhead, and measure GPU wall time against fixed-action, fixed-mixture and frozen-cost policies. This pilot does not demonstrate routing gains.
