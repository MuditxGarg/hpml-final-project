# Profiling pass

`14_profiling_pass.py` measures what each action `(N, B)` does to a shared server at the two background rates frozen by the [sustained calibration](sustained-calibration.md). It is step 2 of the mid-point baseline, and its purpose is to explain why `(4,2048)` collapses near saturation. The logic is in `profiling_pass.py`. It reuses the trace generator, cell order, open-loop runner and idle check of `matched_serving.py`, talks to an already running vLLM server, and needs only the standard library.

This is a synthetic-length replay with `ignore_eos=true`. No answers are scored.

## What a run does

1. Reads `calibration.json` and refuses to start unless its status is `complete`. The light and contended rates, the reference budget and the latency targets all come from that file.
2. For each rate, generates one matched trace: Poisson background at the frozen rate and a Poisson target stream at `--target-fraction` (default 0.05) of the contended rate. Traces and the cell order are written before any cell runs.
3. For each rate, runs a background-only control and every action in a seeded random order. Every cell of a rate offers the identical background requests and the identical target arrivals. The server must be idle before and after each cell.

A target job with `N` samples sends `N` parallel requests for the same prompt; the job finishes when the slowest one does.

## What is logged

| Source | Interval | Fields |
| --- | --- | --- |
| vLLM `/metrics` | `--metrics-interval` (1 s) | running and waiting counts, waiting split by reason, KV-cache usage, preemption counter, token counters, prefix-cache counters, and the sum and count of the queue-time, time-to-first-token, prefill, decode, inter-token-latency and end-to-end histograms |
| `nvidia-smi` (NVML) | same | GPU utilisation, memory used, power draw, temperature |
| Target requests | per request | dispatch time, first-token time, a progress mark every 256 tokens, last-token time, longest gap between tokens |
| Every job | per job | status, latency, parts |

Hardware performance counters are not used; they are unreliable under WSL2.

**Only target requests are streamed.** Background requests are sent exactly as in the calibration and the matched replay, so the profile does not change the load it measures. Queue wait, time to first token and decode time for all requests together come from the server's own histograms; the streamed target requests add the per-action client view.

## Per-cell summary (`profile.json`)

- `streams.background` and `streams.target`: scheduled count, statuses, completion rate, latency mean, p50, p95 and max, unconditional p95, and attainment at each latency target. Failures stay in the denominator.
- `streams.target.streamed_requests`: time to first token, decode milliseconds per token and longest stall, over completed streamed requests.
- `server`: running, waiting and KV-cache mean and max, waiting by reason, counter increases over the window (preemptions, tokens, finished requests), token throughput, and histogram means over the window.
- `server.decode_by_concurrency`: for each pair of consecutive samples, the mean inter-token latency of the tokens produced in between, grouped by the running count in bins of 16. This is the per-token decode time against the number of concurrent sequences.
- `gpu`: mean and max of each sampled field.

Histogram means cover whatever the server observed inside the window, for all streams together. A metric the server does not expose is reported as null, not zero.

## Command

```sh
python 14_profiling_pass.py \
  --url http://127.0.0.1:8001 --model m \
  --lengths tcr_results/main/lengths_v2.json \
  --prompts /absolute/path/prompts.json \
  --hardware-json /absolute/path/hardware.json \
  --calibration /absolute/path/calibration-directory/calibration.json \
  --output /absolute/path/new-profile-directory
```

The output directory must not exist. Outputs: `manifest.json`, `light-trace.json`, `contended-trace.json`, `profile.json`, and per cell `<rate>-cell-<i>-N<n>-B<b>.jsonl` (jobs), `-server.jsonl` and `-gpu.jsonl`.

## Limitations

- One pass per cell. Differences between cells of a few seconds are within run-to-run noise until the pass is repeated with other seeds.
- Sampling once a second can miss short bursts, and a missing sample is not zero load.
- Sample timestamps and job timestamps use the same clock, but the sampler starts a few milliseconds before the replay clock.
- Streaming target requests adds a small amount of server work that the non-streamed matched replay does not have.
- The server must be exclusive, and the calibration must come from the same machine and server configuration.

## Tests

`tests/test_profiling_pass.py` covers metric parsing, the decode-by-concurrency table and the cell summary with fixed inputs, and runs the CLI against `test_fake_vllm_server.py`, which now also streams and exposes queue-time and inter-token histograms.
