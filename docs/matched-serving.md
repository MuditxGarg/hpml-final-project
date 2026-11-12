# Matched E1 serving replay

`10_matched_serving.py` generates persisted Poisson traffic, then sends the same arrival/prompt/empirical-length trace to an existing vLLM server for a background-only control and an action grid. It never launches a GPU process or provisions infrastructure. Python's standard library is sufficient.

This is **E1 synthetic token-count externality**, not answer-quality or end-to-end routing validation. Every completion forces the sampled length with `ignore_eos=true`. Real answers, correctness, reasoning behavior and dynamic routing are not evaluated. The HTTP response must report the requested completion token count; a mismatch is an error, not a successful replay.

## Inputs and commands

Run these from the repository root. Supply a real prompt file and hardware record before running. The prompt JSON is a list such as:

```json
[
  {"thinking": [100, 101, 102], "nothinking": [100, 103]},
  {"thinking": "fully formatted thinking prompt", "nothinking": "fully formatted nonthinking prompt"}
]
```

Token ID lists should come from the exact deployed model's tokenizer and chat template. Strings are also accepted; the server tokenizes those. Use real task prompts and record tokenizer/model revisions in the hardware JSON. Do not use the illustrative IDs above for a real measurement.

```sh
python3 tcr_pipeline_full/tcr_pipeline/10_matched_serving.py generate \
  --lengths tcr_pipeline_full/tcr_pipeline/tcr_results/main/lengths.json \
  --prompts /absolute/path/prompts.json \
  --output /tmp/matched-seed42.json \
  --seed 42 --background-rate 1.0 --target-rate 0.2 \
  --warmup 30 --duration 120 --max-n 4

python3 tcr_pipeline_full/tcr_pipeline/10_matched_serving.py run \
  --trace /tmp/matched-seed42.json \
  --url http://127.0.0.1:8000 --model YOUR_SERVED_MODEL \
  --hardware-json /absolute/path/hardware.json \
  --output /tmp/matched-run-seed42 \
  --reference-budget 1024 --actions 1:0 1:1024 2:1024 4:1024 \
  --cache-policy disabled --slo 60 --attainment 0.95 \
  --timeout 120 --drain 180 --idle-timeout 180 --max-inflight 128
```

Budgets must exist in the supplied lengths file. Its schema is `{"0": [length, ...], "1024": [length, ...], ...}`. `N` must not exceed `--max-n`. The control is `(N=0, B=0)`, automatically first: it offers **background traffic only**. Its target events remain in the raw shared trace with status `not_offered_control`, but trigger no HTTP request and are excluded from its target cohort denominator. The background stream always uses `(N=1, B=reference-budget)`. Action cells add their target traffic to the identical scheduled background trace; every action cell uses the same target schedule. Control background SLO is checked alone, while action feasibility checks both streams.

The committed `lengths.json` reflects its original measurement provenance; notably its B=0 sample is capped at 1024, rather than the corrected 2048-cap labels used by the new evaluation path. Do not silently conflate these datasets. For a corrected-cap live run, supply a separately derived empirical length JSON and identify that provenance in the hardware/config record. The trace records the exact length-source hash and pathname. To use corrected lengths, substitute `--lengths /absolute/path/lengths_v2.json` in the generate command and choose a fresh trace output; the replay command then consumes that trace unchanged. This CLI consumes empirical per-sample lengths, not mean-only tables. Rates are independent Poisson job rates, not HTTP request rates; a target action with N samples issues N simultaneous HTTP requests per job.

Hardware JSON is user-supplied evidence, for example:

```json
{
  "gpu_model": "REPLACE",
  "gpu_count": 1,
  "driver": "REPLACE",
  "vllm_version": "REPLACE",
  "model_revision": "REPLACE",
  "tokenizer_revision": "REPLACE",
  "server_command": "REPLACE with exact launch flags",
  "prefix_caching_enabled": false,
  "other_clients": "none",
  "length_provenance": "committed original lengths; B=0 cap 1024"
}
```

## Matching, isolation and artifacts

Arrival, prompt and budget-specific length generators have independent deterministic streams for background and target. All budgets and up to max-N sample lengths are drawn **before** choosing an action; selecting an action consumes no randomness. The first sample is identical when N changes. Each persisted event has its stream, schedule, warmup/cohort flag, prompt index and lengths. Preserve and reuse the same trace for a seed; generate separate traces for additional seeds. The full trace and its SHA-256 accompany each run.

Before every cell and after draining every cell, both `vllm:num_requests_running` and `vllm:num_requests_waiting` must be present and zero for one second. Missing metrics or failure to become idle aborts the run. This requires an otherwise exclusive server; another client's later arrivals cannot be ruled out by an idle snapshot.

Cache policy is mandatory and recorded:

- `disabled`: user asserts the server was launched with prefix caching disabled; include the launch flags. The runner cannot verify that setting from request metrics.
- `reset-prefix`: POST `/reset_prefix_cache` while idle before each cell; a failing endpoint aborts. This only resets the prefix cache, not CUDA/kernel/model warm state.
- `retain`: explicitly retain cache state; cell-order effects remain and must be acknowledged in results.

Every cell uses the same configured warmup duration. The control runs first, then supplied actions in order (duplicates removed); randomize the supplied action order across independent repetitions when studying ordering effects. This runner does not claim full thermal/cache isolation. All output directories and generated trace files must be new: existing paths fail rather than resuming or overwriting stale experiments. A failed run retains a failed manifest and any finished cells. Use a fresh directory for retry.

Outputs include `manifest.json` (arguments, user-supplied hardware, client platform, hashes, order, cache policy and completion status), a copy of `trace.json`, raw `cell-*-N*-B*.jsonl` jobs with per-sample HTTP timing/results, `cell-*-metrics.jsonl` telemetry, and `summary.json`. The metrics sampler runs during the arrival window and draining, every `--metrics-interval` seconds (default 1). It records running/waiting request gauges, available KV-cache usage and token counters, timestamps and sampling duration; failed scrapes appear as error records. Gauges/counters are summed across labeled replicas, so cache fractions are not a normalized cluster utilization measure. Sampling is diagnostic and may miss short bursts; a missing sample is not zero load. Inspect a failed manifest before consuming its partial summaries. Credentials in server URLs should be avoided because the supplied URL is recorded.

## Cohort and metrics

The arrival window is `[warmup, warmup+duration)`. Warmup events exercise the server but are excluded from summaries. All measured offered jobs stay in the denominator, including local admission drops, HTTP errors, token-count mismatches, timeouts and censoring. The sole deliberate exclusion is the control target stream: those events are recorded as `not_offered_control` and separately counted, because no target work is offered in that cell. The open-loop scheduler never waits for a previous job before offering the next one. `max-inflight` caps admitted **jobs**, so potential concurrent HTTP calls scale with N; saturated admissions are recorded as drops.

A job completes when all N samples succeed. Its latency is last completion minus **scheduled arrival**, including scheduler/thread dispatch lag. Successful completions after the arrival window count. The absolute observation deadline is end-of-arrival-window plus `drain`; unfinished requests are explicitly censored. Socket timeout bounds stalled I/O; the cohort deadline also stops waiting for daemon workers. Server work may persist after a client timeout, which is why idle verification is required before another cell. Raw records include per-sample transport time and dispatch lag when dispatch occurred.

Both streams report scheduled counts, outcome counts, SLO attainment and finite completed-only nearest-rank p95. The unconditional p95 interval treats drops/errors/timeouts as infinite service misses and censored jobs as `[observed duration, infinity)`. A null bound means infinite/unbounded or an empty cohort; the scheduled count distinguishes empty cohorts. Completed-only p95 must not be read as unconditional p95. All censoring counts as a miss in conservative SLO attainment, including jobs censored before their SLO deadline.

`observed_slo_feasible` requires nonempty cohorts and the requested attainment on both streams for action cells; the background-only control requires only its background cohort. `any_observed_feasible=false` explicitly reports that no tested **action** cell met its criterion (the control cannot make this true); it never substitutes a cheap action. This is an observed-cohort check, without confidence intervals or a population-level feasibility claim. Use repeated matched seeds for uncertainty analysis. These measurements do not justify N extrapolation or an answer-quality conclusion.

## Local verification

```sh
python3 -m unittest discover -s tcr_pipeline_full/tcr_pipeline/tests -p test_matched_serving.py -v
```

The fake HTTP integration checks trace matching, late completion, parallel sample requests, identical background traffic and absent target HTTP calls in the control, overload/censor accounting, live queue telemetry, the full control/action CLI artifact path, and stale-output refusal. It requires loopback binding permission. A fake server validates plumbing only; GPU evidence is reported separately under `docs/results/`.

## Bounded free-T4 driver

`colab_run_replay.py` consumes `matched_serving.py`, empirical `lengths.json` and `prompts.json` in the same directory, with an already-running server. Set `LATTE_GPU_DEADLINE` to an explicit UTC timestamp. It recalibrates capacity and SLO on T4, uses target fraction0.10 (not the original0.05), two seeds, loads0.5/0.8, and shuffled candidate order. Control remains first in each block; temporal drift is a limitation. It archives partial results on failure and skips blocks that cannot fit the conservative deadline bound.

Start vLLM with explicit `--no-enable-prefix-caching` and check its startup configuration; omitting an enable flag may not disable caching. The driver does not disconnect Colab or stop the server: those lifecycle actions belong to the session operator and must respect the approved runtime budget.

`--seeds`, `--loads` and `--actions` can prospectively reduce the matrix to fit a session. `--reuse-calibration PATH` freezes a previous calibration and records its hash; it does not retune capacity or SLO to the new outcomes. Request timeout and drain are at least five seconds beyond the calibrated SLO (and at least 60 seconds). The November 12 corrected pilot uses a 92-second SLO and 97-second timeout/drain. Its earlier interrupted run is retained as calibration/diagnostic evidence, not a completed comparison.

After downloading an archive, summarize its completed and partial blocks with:

```sh
python3 tcr_pipeline_full/tcr_pipeline/11_report_matched.py \
  --run /absolute/path/extracted-run --output /absolute/path/new-report
```

The report keeps failures alongside completed-only p95, reports empty target cohorts and skipped blocks, and limits queue telemetry to the measured arrival window. It does not infer statistical significance or answer quality from the replay.
