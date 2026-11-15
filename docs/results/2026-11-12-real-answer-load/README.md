# Matched Real-Answer Load Comparison

All four paired blocks and all16 cells completed on November12,2026, on ins091 / RTX A6000. The experiment produced60 target decisions using the same five questions across two seeds and two nominal loads. **The higher arrival rate increased latency, but neither setting produced observed queueing. This run does not demonstrate a load-dependent strategy crossover or a routing benefit.**

## Target results

Every target returned a parseable answer within the fixed87-second target. Thus correct counts and correct-within-target counts are identical. Counts below pool two repetitions of the same five questions, not ten independent questions.

| Nominal load | Action (N, thinking tokens) | Correct within87s | Median target latency | Maximum target latency |
|---|---|---:|---:|---:|
| 0.5 | (1,0) | 5/10 | 5.69s | 13.61s |
| 0.5 | (4,0) | 7/10 | 8.29s | 19.69s |
| 0.5 | (1,2048) | 5/10 | 16.80s | 19.12s |
| 0.9 | (1,0) | 3/10 | 7.59s | 17.35s |
| 0.9 | (4,0) | 5/10 | 11.03s | 23.71s |
| 0.9 | (1,2048) | 5/10 | 20.48s | 22.66s |

The numbers do not support a causal accuracy-loss claim from higher load. Input seeds were matched, but vLLM batching can change sampled outputs; there are only two repetitions and five development questions. The nominal low blocks also happened to run before both high blocks in the pre-randomized order, leaving temporal drift as a limitation.

## Background behavior and actual contention

The finite-burst throughput proxy was3.16249 requests/s, giving arrival rates1.58125 and2.84624 requests/s. The three64-request calibration bursts ranged from1.83572 to3.33787 requests/s. This variability and the absence of queues demonstrate why the proxy must not be called steady-state capacity or the loads50%/90% utilization.

Across all16 measured windows, all1,600 telemetry samples were successful and **waiting requests were always zero**. Peak sampled KV-cache usage was9.27%. Brief bursts between one-second samples are possible, but there is no evidence of sustained queue pressure. All transported requests, including unparseable outputs and warmup requests, finished within24 seconds; there were no drops, transport errors, timeouts or censoring.

Of3,416 measured background request executions across all cells,232 (6.79%) lacked a parsed boxed answer. **Every one hit the2048-token shallow-output cap.** These are answer-delivery misses, not87-second latency misses. Transport completion within87 seconds was100%, while the predeclared parseable-answer-within87s metric was lower.

| Seed | Load | Control: parsed within87s | (1,0): change vs control | (4,0): change | (1,2048): change |
|---|---|---:|---:|---:|---:|
| 20261116 | 0.5 | 151/165 (91.52%) | +0.61pp | +2.42pp | −1.21pp |
| 20261117 | 0.5 | 144/157 (91.72%) | +1.27pp | +0.64pp | +0.64pp |
| 20261116 | 0.9 | 249/263 (94.68%) | −1.52pp | +0.38pp | −1.52pp |
| 20261117 | 0.9 | 252/269 (93.68%) | +0.37pp | −0.74pp | +0.37pp |

All four controls were below the95% answer-delivery criterion. Only one action cell, (4,0) at seed20261116/load0.9, crossed95% (250/263), while its matched control was249/263. This one-request threshold crossing is not evidence that extra sampling improves background service, and there is no robust feasible-strategy comparison here.

Completed-answer-only background p95 ranged from11.67 to17.44 seconds. The machine-readable report deliberately omits paired p95 deltas because every background cell contains answer-delivery failures; it keeps those misses in unconditional statistics. Conditional p95 should not be used to conceal the missing answers. Paired SLO differences above remain reported for every complete block.

## What to do next

1. **Separate serving completion from answer validity in the next predeclared report.** Keep the current parseable-answer metric and failures intact, but also report transport/EOS completion within the target and the fraction hitting the output cap. Resolve or explicitly retain the shallow-answer cap limitation before treating95% answer delivery as a serving feasibility condition. Do not change this run's definition retrospectively.
2. **Calibrate using sustained open-loop background-only traffic on the same hardware.** Predefine a rate ladder and use only queue growth, completion rate and latency to select a light and genuinely contended setting. Freeze them before new target comparisons. The current burst proxy undershot the conditions needed to test load-aware allocation.
3. Repeat matched actions with more prespecified questions/seeds at those frozen settings. Only then compare load-aware routing against fixed-cost routing and fixed actions if the measurements show a useful tradeoff. Any new GPU run needs authorization; no new allocation was launched after this result.

## Audit and provenance

- Four complete paired blocks,16 cells,60 target decisions; no skipped blocks. The control offered no target requests. All matched cells used identical background event IDs, question IDs, arrival times, seeds and first-stage request bodies, verified from raw evidence. All sample transports completed.
- Target answers were scored on the cluster with the project math-verify grader. An independent normalized-gold check agreed except for one correct arithmetic expression ending in14; its sum was checked directly.
- Same node and model as the idle diagnostic, but that run did not record GPU UUID, so exact physical-card continuity cannot be verified retrospectively. This run records UUID GPU-3e8517f2-b616-f4bf-c4c5-42f8ef24890e.
- Fixed Qwen3-1.7B revision70d244cc86ccca08cf5af4e1e306ecf908b1ad5e; FP16; vLLM0.32.0+cu129; prefix cache disabled. Exact environment and server command included. Source code e5f0706; executed bundle is inside the archive.
- Successful job13889524:22:55:13–23:35:53 America/New_York, elapsed40m40s, exit0. Failed startup attempt13889319 consumed3m30s. **Cumulative allocation44m10s**, including the recorded batch steps; GPU released and user queue empty. The retry's effective Slurm limit was115min, reduced after submission to allow cleanup margin within the cumulative120min authorization. The archived submitted script retains its original116m30s request; the recorded scheduler override governed execution.

[summary.json](summary.json) contains per-cell metrics and paired differences; [raw-evidence.zip](raw-evidence.zip) contains every downloaded raw/scored response, telemetry trace, logs and the executed bundle; [accounting.txt](accounting.txt) records allocation release. [artifact-hashes.json](artifact-hashes.json) covers the saved artifacts. This is descriptive diagnostic evidence, not a held-out benchmark or significance test.
