# Profiling pass on the RTX 4080 laptop, 2026-11-15

One pass of `14_profiling_pass.py` (seed 42) on the HP Omen laptop: RTX 4080 Laptop GPU (12 GB), WSL2, vLLM 0.32.0, Qwen3-1.7B in BF16, same server process as the [sustained calibration](../2026-11-15-sustained-calibration-4080/README.md). Protocol: [docs/profiling-pass.md](../../profiling-pass.md). Code at commit `5b3431a`, clean working tree.

**Finding: `(4,2048)` collapses because the KV cache fills, not because the GPU or the sequence limit saturates.** When KV-cache usage reaches 1.0 the server starts preempting running sequences, a waiting queue forms within a second or two, and streamed requests stall for 10 to 25 s. The same mechanism appears for `(1,2048)` and `(4,0)` at the contended rate.

## What was measured

- Background: synthetic single requests at budget 1024 at the frozen rates, light 1.231 req/s and contended 2.463 req/s.
- Target stream: Poisson at 0.123 jobs/s (5% of the contended rate), synthetic lengths with `ignore_eos`, streamed. An action `(N, B)` sends `N` parallel requests with lengths drawn from the budget-`B` pool of `lengths_v2.json`.
- Per rate: a background-only control and four actions in a seeded random order, 60 s warm-up and 180 s window per cell, the same background trace in all five cells of a rate. Latency targets from the calibration: 20.9 s (tight) and 41.8 s (loose).
- Every request in every cell completed (no drops, timeouts or errors). All 10 cells had 170 to 172 GPU samples and no failed samples.

## What was not measured

- No real answers and no accuracy. Lengths are replayed.
- One pass per cell. Differences of a few seconds between cells are within run-to-run noise (the two calibration passes differed by up to 3 s in background p95 at similar rates).
- The target trace differs between the two rates (29 target jobs at light, 21 at contended), so target latency is comparable within a rate, not across rates.
- No hardware performance counters.

## Results

### Latency

| Rate | Action | Background p50 (s) | Background p95 (s) | Within 20.9 s | Within 41.8 s | Target jobs | Target p50 (s) | Target p95 (s) | Target within 41.8 s |
|---|---|---|---|---|---|---|---|---|---|
| light | control | 14.2 | 15.8 | 1.00 | 1.00 | 0 | - | - | - |
| light | (1,0) | 14.6 | 16.8 | 1.00 | 1.00 | 29 | 8.5 | 31.8 | 1.00 |
| light | (1,2048) | 16.2 | 19.2 | 1.00 | 1.00 | 29 | 30.9 | 37.6 | 1.00 |
| light | (4,0) | 16.7 | 20.5 | 0.99 | 1.00 | 29 | 22.1 | 39.4 | 1.00 |
| light | (4,2048) | 28.1 | 45.7 | 0.11 | 0.87 | 29 | 53.1 | 73.4 | 0.03 |
| contended | control | 23.3 | 26.9 | 0.31 | 1.00 | 0 | - | - | - |
| contended | (1,0) | 25.1 | 28.2 | 0.16 | 1.00 | 21 | 14.0 | 48.0 | 0.86 |
| contended | (1,2048) | 31.6 | 40.2 | 0.04 | 0.99 | 21 | 54.1 | 66.4 | 0.29 |
| contended | (4,0) | 32.9 | 43.0 | 0.04 | 0.94 | 21 | 35.4 | 64.0 | 0.57 |
| contended | (4,2048) | 67.1 | 84.9 | 0.00 | 0.15 | 21 | 98.9 | 112.4 | 0.00 |

"Within" columns are the fraction of scheduled requests that completed inside that target.

### Server state

| Rate | Action | Running mean / max | Waiting mean / max | KV cache mean / max | Preemptions | Preemptions per finished request | Mean queue wait (s) | Mean time to first token (s) | Mean inter-token latency (ms) | Tokens/s |
|---|---|---|---|---|---|---|---|---|---|---|
| light | control | 18.9 / 29 | 0.0 / 0 | 0.20 / 0.31 | 0 | 0.00 | 0.00 | 0.04 | 13.8 | 1357 |
| light | (1,0) | 21.5 / 31 | 0.0 / 0 | 0.23 / 0.37 | 0 | 0.00 | 0.00 | 0.04 | 14.5 | 1477 |
| light | (1,2048) | 26.2 / 40 | 0.0 / 0 | 0.31 / 0.49 | 0 | 0.00 | 0.00 | 0.05 | 16.3 | 1608 |
| light | (4,0) | 30.9 / 50 | 0.0 / 0 | 0.34 / 0.60 | 0 | 0.00 | 0.00 | 0.05 | 16.8 | 1832 |
| light | (4,2048) | 59.8 / 91 | 6.1 / 53 | 0.79 / 1.00 | 441 | 1.28 | 1.26 | 1.21 | 27.3 | 2323 |
| contended | control | 58.3 / 78 | 0.0 / 0 | 0.63 / 0.88 | 0 | 0.00 | 0.00 | 0.07 | 22.4 | 2604 |
| contended | (1,0) | 63.9 / 84 | 0.0 / 0 | 0.69 / 0.93 | 0 | 0.00 | 0.00 | 0.07 | 24.0 | 2671 |
| contended | (1,2048) | 74.5 / 101 | 7.6 / 55 | 0.84 / 1.00 | 651 | 1.18 | 0.87 | 1.20 | 29.4 | 2680 |
| contended | (4,0) | 77.9 / 116 | 11.2 / 63 | 0.83 / 1.00 | 891 | 1.71 | 1.38 | 1.74 | 30.2 | 2776 |
| contended | (4,2048) | 83.8 / 128 | 78.3 / 180 | 0.98 / 1.00 | 1243 | 3.07 | 10.84 | 17.91 | 39.6 | 2606 |

All columns cover the 180 s window. Queue wait, time to first token and inter-token latency are means of the server's own histograms over all requests in the window. Every waiting request was waiting for the reason `capacity`; none was `deferred`.

### Streamed target requests (client side)

| Rate | Action | Streamed requests | Time to first token p50 / p95 (s) | Decode ms per token p50 / p95 | Longest stall p95 / max (s) |
|---|---|---|---|---|---|
| light | (1,0) | 29 of 29 | 0.05 / 0.05 | 14.6 / 16.3 | 0.06 / 0.06 |
| light | (1,2048) | 29 of 29 | 0.05 / 0.06 | 16.7 / 18.5 | 0.07 / 0.07 |
| light | (4,0) | 116 of 116 | 0.06 / 0.07 | 17.0 / 19.9 | 0.07 / 0.07 |
| light | (4,2048) | 116 of 116 | 0.09 / 6.32 | 26.0 / 38.0 | 9.48 / 13.76 |
| contended | (1,0) | 21 of 21 | 0.07 / 0.09 | 23.4 / 27.7 | 0.08 / 0.08 |
| contended | (1,2048) | 21 of 21 | 0.09 / 6.05 | 29.0 / 33.5 | 11.28 / 12.03 |
| contended | (4,0) | 84 of 84 | 0.09 / 5.76 | 29.4 / 55.8 | 15.02 / 15.26 |
| contended | (4,2048) | 84 of 84 | 28.91 / 51.80 | 34.1 / 47.0 | 21.80 / 25.29 |

"Longest stall" is the largest gap between two consecutive tokens of one request. Decode time per token includes stalls.

### Per-token decode time against concurrent sequences

Mean inter-token latency between consecutive one-second samples, grouped by the running count, taking each cell's median for a bin and keeping bins with at least 10 intervals.

| Running sequences | Cells with at least 10 intervals | Inter-token latency, lowest to highest cell median (ms) | Cells with no preemptions (ms) | Cells with preemptions (ms) |
|---|---|---|---|---|
| 0 to 15 | 2 | 12.6 to 12.6 | 12.6 to 12.6 | - |
| 16 to 31 | 4 | 13.8 to 15.3 | 13.8 to 15.3 | - |
| 32 to 47 | 7 | 17.9 to 20.2 | 17.9 to 18.6 | 19.2 to 20.2 |
| 48 to 63 | 6 | 21.7 to 33.3 | 21.7 to 22.1 | 22.8 to 33.3 |
| 64 to 79 | 6 | 25.2 to 32.5 | 25.2 to 25.4 | 28.1 to 32.5 |
| 80 to 95 | 5 | 27.5 to 31.4 | 27.5 | 28.2 to 31.4 |
| 96 to 111 | 2 | 28.3 to 30.9 | - | 28.3 to 30.9 |
| 112 to 127 | 1 | 32.2 | - | 32.2 |

### Onset of the collapse

Read from the per-second server samples of the four cells that had preemptions (times are seconds from the start of the cell, including the 60 s warm-up; the last column counts the whole cell including the drain).

| Rate | Action | KV cache first at 0.99 or more (s) | First preemption (s) | Running count at first preemption | First waiting request (s) | Preemptions over the whole cell |
|---|---|---|---|---|---|---|
| light | (4,2048) | 104.7 | 106.8 | 75 | 106.8 | 480 |
| contended | (1,2048) | 114.7 | 115.7 | 88 | 116.8 | 685 |
| contended | (4,0) | 111.7 | 111.7 | 90 | 111.7 | 947 |
| contended | (4,2048) | 40.3 | 40.3 | 72 | 40.3 | 1879 |

### GPU

Mean GPU utilisation was 97 to 99% in every cell, including the light control. Mean temperature was 86 C in every cell (maximum 87 to 88 C) and mean power draw 117 to 126 W. Memory stayed at the server's fixed allocation.

## Reading the results

1. **Below the KV limit, cost is smooth.** With no preemptions, per-token decode time rises steadily with the number of running sequences, from 12.6 ms under 16 sequences to about 25 ms at 64 to 79 and 27.5 ms at 80 to 95. Nothing waits, time to first token stays under 0.1 s, and background p95 grows by a few seconds per action. This is the regime a token-load cost model can describe.
2. **The knee is the KV cache.** In all four collapsing cells the first preemption and the first waiting request came within about two seconds of KV-cache usage reaching 0.99. The running count at that moment was 72 to 90, well under the 128-sequence limit, and only the worst cell ever reached 128. The cache holds 55,664 tokens on this machine, so what runs out is tokens held in memory, not sequence slots.
3. **Preemption is what makes it a collapse.** Once preemptions start, requests are evicted and resumed: streamed requests show single stalls of 9 to 25 s, mean queue wait goes from 0 to 1 s and then 11 s, and in the worst cell every finished request had been preempted 3 times on average. Inside a bin of equal concurrency, cells with preemptions are slower per token than cells without (for example 22.8 to 33.3 ms against 21.7 to 22.1 ms at 48 to 63 sequences), so concurrency alone does not explain the slowdown.
4. **Token throughput does not rise in the collapsed cell.** `(4,2048)` at the contended rate delivered 2,606 tokens/s, the same as the control (2,604) and below `(4,0)` (2,776), while offering far more work. The extra demand turned into queue and repeated work, not output.
5. **GPU utilisation cannot see any of this.** It reads 97 to 99% from the light control to the collapsed cell. It is not a usable load signal for a controller on this setup; KV-cache usage and the preemption counter are.
6. **This explains the cost-model failure.** The isotonic token-load model predicted 35.75 s for `(4,2048)` at 0.8 load against 61.77 s measured (cost-model check of PR #1, [results of 2026-11-12](../2026-11-12/README.md)). Tokens per second describe point 1 but carry no information about how many tokens are held in the cache at once, which is what triggers points 2 and 3.
7. **Width and depth both reach the limit at the contended rate, depth with fewer samples.** `(1,2048)` and `(4,0)` both filled the cache and produced similar background p95 (40.2 s and 43.0 s). This is one pass on synthetic lengths; whether the two differ at equal accuracy is the question of the matched real-answer comparison, not something this table settles.
8. **The laptop runs hot.** 86 C in every cell means thermal limits may be shaping absolute latencies. Final latency numbers should come from the cluster, as planned.

## Implications for the next steps

- A cost model should take KV-cache occupancy (tokens held, which depends on how long sequences live and how long they are) as an input, with the preemption onset as the feasibility boundary.
- A controller can use KV-cache usage as its early signal: it crossed 0.99 one to two seconds before latency broke.
- `--max-num-seqs 128` is not the binding limit here. Lowering it so that admission stops before the cache fills would trade preemptions for queueing; that is a server setting worth one controlled comparison.

## Files

- `profile.json`: every cell's summary. `manifest.json`: arguments, git commit, environment, cell order, hashes. `hardware.json`: user-supplied hardware and server record.
- Per-job rows, per-second server samples, GPU samples and the two traces (12 MB) are in the git-ignored `evaluation-results/profiling-4080-20261115-seed42/` on the laptop. The onset table above was read from the `*-server.jsonl` files there.
