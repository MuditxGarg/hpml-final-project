# Matched Real-Answer Load Comparison

This follows the completed [idle real-answer diagnostic](results/2026-11-12-real-answer-validation/README.md). It tests fixed answering configurations under matched background traffic. It is not a router evaluation.

## Frozen protocol

- Same school node `ins091`, RTX A6000 model, pinned Qwen3-1.7B and offline environment as the idle run. Record GPU UUID for future comparisons; the earlier run did not record UUID, so physical-card identity cannot be verified retrospectively.
- Target cohort: the same five questions, one per level. Actions `(1,0)`, `(4,0)`, `(1,2048)`, plus a background-only control. Background: single shallow real answers from the other495 MATH-500 prompts. Both streams use EOS-aware actual generation, not synthetic forced-token lengths.
- Three64-request simultaneous shallow bursts estimate a finite-burst throughput proxy. It is not a proven sustainable arrival capacity. Freeze low/high arrival rates at0.5/0.9 times its median; inspect queue telemetry afterward to determine whether meaningful contention occurred. Do not tune rates using target accuracy.
- Two seeds20261116/20261117, two load levels, four randomized cells per block. Twenty seconds background warmup and100 seconds measured arrivals per cell. Five target requests at measured times10/30/50/70/90 seconds; question order shuffled by seed. Background arrivals Poisson. No target warmup.
- A block reuses exact background arrival times/prompts/seeds across its control and actions. Target schedules/seeds match across both loads. Matched inputs do not guarantee identical outputs under different batching. No prefix cache.
- Fixed87s delivery target for both streams;180s observation limit from scheduled arrival, including dispatch delay and all deep stages. Missing boxes, transport errors, local drops and censoring stay in offered denominators. Target correctness graded offline; background grading is not part of the objective.
- AllN samples must return, then normalized-box plurality, ties by sample index. Keep the prior pilot's tie sensitivity visible. Controls do not send target requests and have no target accuracy denominator.

## Running

`evaluation-results/real-answer-load-bundle/` is a prepared local bundle (ignored by Git) containing the five-question plan/gold,495 background prompts, exact code, lock file and batch wrapper. Its manifest hashes every input. On the cluster, create a fresh execution directory with this bundle and symlink `model/` and `wheelhouse/` to the verified cache from the previous experiment. Submit only after a new explicit GPU authorization.

```sh
sbatch run_real_answer_load.sbatch
```

The batch uses `edu/short/free_short`, node `ins091`, one GPU, no requeue, a120-minute hard limit, and a117-minute internal deadline. It refuses a different GPU model, records UUID/hardware/environment, and saves artifacts on exit. It stops admitting new blocks unless four worst-case cells and a180-second reserve fit. Calibration takes at most12 minutes including idle checks; comparisons at most96 minutes; setup expected~3 minutes from cache. The hard limit overrides these estimates. A preempted or partial run is retained, not silently rerun within an expired authorization.

For an already authorized dedicated server:

```sh
python code/real_answer_load.py run --plan input/plan.json \
  --background input/background.json --out run --deadline-epoch UNIX_TIMESTAMP
python code/real_answer_load.py score --out run --gold input/gold.json
```

If generation stops early, offline scoring can process saved completed cells after downloading and installing the pinned grading dependencies. Every block is marked paired-complete only if all four cells and their post-run idle checks finished. Review `manifest.json` before interpreting a partial result.

## Analysis

Inspect `calibration.json`, per-block trace/order files, raw sample responses and telemetry. Report target correct-and-on-time counts together with background SLO attainment and failure counts. A completed-only p95 does not include failed requests; preserve that distinction. Queue evidence determines whether the nominal high setting really produced load. Summarize paired control-to-action differences separately for each seed and load; with just two seeds, do not present request-level bootstraps as independent repetitions.

The result may show no feasible crossover or no high contention. Either is useful diagnostic evidence, not a reason to claim routing gains. A later router comparison requires more questions/seeds and independent training/operating-point selection.
