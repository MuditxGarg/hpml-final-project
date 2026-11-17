# Sustained calibration on the RTX 4080 laptop, 2026-11-15

Two full passes of `13_sustained_calibration.py` (seeds 42 and 1042) on the HP Omen laptop: RTX 4080 Laptop GPU (12 GB), WSL2, vLLM 0.32.0, Qwen3-1.7B in BF16. Protocol: [docs/sustained-calibration.md](../../sustained-calibration.md). Code at commit `25d6f83`, clean working tree.

**Frozen rates for this machine and server configuration: light 1.23 req/s, contended 2.46 req/s.** Both come from the seed 1042 pass. The reason for preferring it over seed 42 is given under "Choosing the contended rate".

## What was measured

- Background traffic only: single requests at the reference budget `B=1024`, with completion lengths replayed from the budget-1024 pool of `tcr_results/main/lengths_v2.json` using `ignore_eos`. Prompts are the 500 MATH-500 questions in the Qwen3 chat template.
- Per pass: 24 sequential idle requests, one 64-request burst to place the ladder, then one rung per rate with a 60 s warm-up and a 180 s measurement window, in ascending order. The server was idle before and after every rung.
- Server command and versions are in `hardware.json`. The server was started once and served both passes. Its KV cache held 55,664 tokens.

## What was not measured

- No answers were scored. Lengths are synthetic replays.
- No action other than the background request was offered. The cost of an extra stream is the job of the profiling pass.
- One pass per rung per seed. Two seeds are not enough for confidence intervals.
- WSL2 on a laptop: timing is noisier than bare Linux, and one manual `nvidia-smi` reading early in the second pass showed 86 C (temperature is not logged by this script).

## Results

Every scheduled request completed in every rung of both passes (no drops, timeouts or errors), so completion rate does not separate the rungs. Latency, KV-cache usage and backlog growth do.

### Seed 42

Idle p95 10.43 s, latency targets 20.85 s and 41.70 s, burst anchor 2.394 req/s.

| Rate (req/s) | Requests | Completed | p50 (s) | p95 (s) | p95 / idle | Running mean / max | Waiting mean / max | KV cache mean / max | Backlog growth | Tokens/s | Stable | Role |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.718 | 109 | 109 | 11.6 | 12.5 | 1.20 | 6.8 / 11 | 0.0 / 0 | 0.08 / 0.13 | -0.013 | 602 | yes |  |
| 1.197 | 209 | 209 | 13.1 | 14.4 | 1.38 | 14.8 / 23 | 0.0 / 0 | 0.16 / 0.28 | +0.004 | 1148 | yes | light |
| 1.676 | 320 | 320 | 16.3 | 18.8 | 1.81 | 29.4 / 44 | 0.0 / 0 | 0.32 / 0.53 | +0.012 | 1809 | yes |  |
| 2.035 | 369 | 369 | 17.5 | 20.5 | 1.97 | 37.1 / 54 | 0.0 / 0 | 0.40 / 0.64 | -0.016 | 2106 | yes |  |
| 2.394 | 424 | 424 | 20.1 | 22.1 | 2.12 | 45.7 / 62 | 0.0 / 0 | 0.49 / 0.68 | +0.033 | 2358 | yes |  |
| 2.753 | 491 | 491 | 23.4 | 26.2 | 2.51 | 61.8 / 78 | 0.0 / 0 | 0.66 / 0.85 | +0.030 | 2700 | yes | contended |
| 3.112 | 553 | 553 | 32.7 | 45.5 | 4.36 | 86.9 / 116 | 9.0 / 78 | 0.92 / 1.00 | +0.077 | 2927 | no |  |

### Seed 1042

Idle p95 10.46 s, latency targets 20.92 s and 41.84 s, burst anchor 2.463 req/s. The ladder stopped after the first unstable rung, so the seventh rung (1.3 times the anchor) was not run.

| Rate (req/s) | Requests | Completed | p50 (s) | p95 (s) | p95 / idle | Running mean / max | Waiting mean / max | KV cache mean / max | Backlog growth | Tokens/s | Stable | Role |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.739 | 137 | 137 | 12.2 | 13.1 | 1.25 | 9.3 / 17 | 0.0 / 0 | 0.10 / 0.19 | -0.006 | 785 | yes |  |
| 1.231 | 204 | 204 | 13.0 | 14.3 | 1.37 | 14.6 / 23 | 0.0 / 0 | 0.16 / 0.25 | +0.008 | 1133 | yes | light |
| 1.724 | 312 | 312 | 16.1 | 17.9 | 1.71 | 27.7 / 40 | 0.0 / 0 | 0.30 / 0.45 | +0.012 | 1753 | yes |  |
| 2.094 | 363 | 363 | 17.7 | 19.3 | 1.84 | 35.7 / 48 | 0.0 / 0 | 0.39 / 0.52 | -0.000 | 2064 | yes |  |
| 2.463 | 462 | 462 | 21.1 | 25.2 | 2.41 | 54.2 / 72 | 0.0 / 0 | 0.58 / 0.83 | +0.004 | 2578 | yes | contended |
| 2.832 | 527 | 527 | 25.9 | 31.9 | 3.05 | 68.9 / 96 | 0.5 / 16 | 0.72 / 1.00 | +0.103 | 2813 | no |  |

Columns: p95 is over completed requests (all of them here). Backlog growth is the slope of running + waiting over the window divided by the offered rate. "Stable" and "Role" are the flags the pre-specified rule assigned.

## Reading the results

- **Idle latency reproduces.** Idle p95 was 10.43 s and 10.46 s, against 10.35 s in the original sweep of 2026-11-05 (`tcr_results/main/load/calibration.json`).
- **Slowdown is gradual, then there is a knee.** Background p95 rises smoothly from 1.2 to about 2.5 times idle while nothing waits. No request waited in any rung at or below 2.75 req/s in either pass.
- **The knee coincides with a full KV cache.** In both passes the first rung with a waiting queue is the first rung where KV-cache usage reached 1.00 (3.112 req/s in seed 42, 2.832 req/s in seed 1042). The running count peaked at 116 and 96 there, below the 128-sequence limit, so the queue formed because cache blocks ran out, not because sequence slots did. This run did not log preemptions; the profiling pass does.
- **Token throughput flattens near 2,900 tokens/s** (2,927 and 2,813 in the two unstable rungs), consistent with the 2,794 tokens/s closed-loop saturation of the original sweep.
- **The stability edge lies between 2.75 and 2.83 req/s.** 2.753 req/s was classed stable in seed 42 with backlog growth +0.030; 2.832 req/s was classed unstable in seed 1042 with +0.103. The rule threshold is 0.05.

## Choosing the contended rate

The two passes froze different contended rates: 2.753 req/s (seed 42) and 2.463 req/s (seed 1042). The light rates agree (1.197 and 1.231 req/s).

2.753 req/s sits right at the knee: KV-cache usage peaked at 0.85, backlog growth was more than half of the threshold, and a rate 3% higher was unstable in the other pass. A contended rate that close to saturation would make every later comparison depend on run-to-run luck. The neighbouring rates 2.394 and 2.463 req/s were stable in both passes with p95 above the tight target (2.12 and 2.41 times idle), which is the contention evidence the rule asks for.

The frozen pair is therefore the seed 1042 output: **light 1.231 req/s, contended 2.463 req/s**. This choice applies the rule's own flags across both passes (a rate must be stable in every pass) and was made before any profiling or action comparison was run. As a fraction of the original sweep's capacity estimate of 2.78 req/s, these are about 0.44 and 0.89.

## Files

- `seed-42/`, `seed-1042/`: `calibration.json` (rule, every rung, frozen rates), `manifest.json` (arguments, git commit, environment, hashes), `idle.json`, `anchor.json`.
- `hardware.json`: user-supplied hardware and server record.
- Per-request rows, per-second telemetry and the exact traces (about 6 MB per pass) are kept in the git-ignored `evaluation-results/sustained-calibration-4080-20261115-seed{42,1042}/` on the laptop. Each rung's trace hash is recorded in `calibration.json`.
