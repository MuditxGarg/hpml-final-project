# Sustained calibration

`13_sustained_calibration.py` measures how one machine behaves under background traffic alone and freezes two background rates for later experiments: a **light** rate and a **contended** rate. It is step 1 of the mid-point baseline. It talks to an already running vLLM server, never launches one, and needs only the Python standard library. The logic lives in `sustained_calibration.py` and reuses the trace generator, open-loop runner, idle check and telemetry sampler of `matched_serving.py`.

This is a synthetic-length measurement. Every request replays an empirical completion length for the reference budget with `ignore_eos=true`. No answers are scored and no action other than the background request `(N=1, B=reference budget)` is offered.

## What a run does

1. **Idle reference.** Sequential reference requests on an idle server (the first is a discarded warm-up). The idle p95 defines the latency targets as multiples of it (default 2x and 4x), the same definition as `04_serve_measure.py`.
2. **Ladder placement.** By default the ladder is a list of fractions (`0.3 0.5 0.7 0.85 1.0 1.15 1.3`) of an anchor: the throughput of one simultaneous burst of reference requests. The anchor is a proxy that only places the ladder on this machine; it is not reported as capacity. `--rates` gives the ladder in absolute requests/s and skips the anchor.
3. **Rungs, in ascending order.** Each rung is a persisted Poisson trace at one rate: warm-up, a sustained measurement window (default 180 s), then a drain. The server must be idle (running and waiting both zero) before and after every rung. Requests that arrive in the window form the measured cohort.
4. **Early stop.** The ladder stops after the first unstable rung (`--stop-after-unstable`). Skipped rungs are listed in the manifest.

## Per-rung record

- `completion_rate`: completed requests over all scheduled requests in the window. Drops, timeouts, errors and censored requests stay in the denominator and are counted in `statuses`.
- `throughput_rps` against `offered_rps`: completions inside the window against arrivals inside the window.
- `latency`: mean, p50, p90, p95, p99 and max over completed requests, plus unconditional p95 bounds that treat every failure as an infinite miss.
- `queue`: mean and max of `vllm:num_requests_waiting` and `vllm:num_requests_running`, the fraction of samples with a non-empty waiting queue, KV-cache usage, token throughput, and backlog growth.
- **Backlog growth** is the least-squares slope of running + waiting over the window, divided by the offered rate. It is the fraction of offered requests that pile up instead of being served: about zero for a sustainable rate, about `1 - capacity/rate` past saturation.

## Frozen-rate rule

The rule is fixed before the run and written into `calibration.json` with the thresholds used.

- **Stable rung:** completion rate at least 0.99, no dropped requests, backlog growth at most 0.05, and at least 5 telemetry samples in the window. A rung whose telemetry is missing is unverified and is never treated as stable.
- **Contended rate:** the highest stable rung whose unconditional p95 is inside the loose target and that shows contention. Contention means p95 above the tight target, or a non-empty waiting queue in at least half of the samples.
- **Light rate:** the highest stable rung at no more than 0.6 times the contended rate, with p95 inside the tight target and a waiting queue in at most 5% of samples.

If the ladder has no rung for one of them, the status says which is missing, both raw rungs remain in the output, and the script exits with code 2. Nothing is interpolated.

Contention is not defined by the waiting count alone because vLLM admits requests into the running batch until `max_num_seqs` is reached, so a busy server slows every sequence down long before anything waits. The background-only control cells of the committed RTX 4080 sweep (`tcr_results/main/load/ext_cells.jsonl`) show a mean waiting count of 0 at loads 0.5, 0.8 and 1.0 while background p95 rises well above the idle p95; a waiting queue appears only at load 1.3.

The 0.6 ratio keeps the two rates clearly apart, since the go/no-go test compares the best action at light and at heavy load.

## Command

Run from `tcr_pipeline_full/tcr_pipeline/` against a dedicated server. The prompt and hardware JSON formats are those of [matched-serving.md](matched-serving.md).

```sh
python 13_sustained_calibration.py \
  --url http://127.0.0.1:8001 --model m \
  --lengths tcr_results/main/lengths_v2.json \
  --prompts /absolute/path/prompts.json \
  --hardware-json /absolute/path/hardware.json \
  --reference-budget 1024 \
  --output /absolute/path/new-calibration-directory
```

The output directory must not exist. A failed run keeps a failed manifest and every finished rung; retry into a fresh directory.

## Outputs

| File | Content |
| --- | --- |
| `manifest.json` | arguments, seed, user-supplied hardware, git commit and dirty flag, client platform, `nvidia-smi` line, server version and models when exposed, input hashes, ladder, rungs run and skipped, status |
| `idle.json` | every idle sample and the idle p50 and p95 |
| `anchor.json`, `anchor.jsonl` | the burst anchor and its raw requests (absent with `--rates`) |
| `rung-NN-trace.json` | the exact arrival, prompt and length trace of the rung |
| `rung-NN.jsonl` | one row per scheduled request with status and timing |
| `rung-NN-metrics.jsonl` | timestamped `/metrics` samples, failed scrapes included |
| `calibration.json` | latency targets, the rule, every rung with its flags, and the frozen light and contended rates |

## Limitations

- One pass per rung and an ascending order, so thermal or cache drift is confounded with rate. Repeat the whole calibration with another `--seed` before relying on a rung close to a threshold.
- A rung near saturation is ambiguous in a finite window. Lengthen `--duration` rather than loosening the thresholds.
- The frozen rates hold for one machine, one server configuration and one reference budget. Do not reuse them elsewhere.
- Background-only: this run says nothing about what an extra action stream costs. That is the profiling pass and the matched comparison.
- The server must be exclusive. Another client's traffic cannot be ruled out by the idle check.

## Tests

`tests/test_sustained_calibration.py` covers the rung summary and the selection rule with fixed inputs, and runs the full CLI against `test_fake_vllm_server.py` started as a subprocess with a small capacity, on a ladder with a quiet rung, a busy but bounded rung and an overloaded rung. The fake server's numbers are meaningless; the test checks the bookkeeping and the rule.
