# Real Answer Completion Validation

The fixed five-question pilot completed on November 12, 2026, on one school-cluster RTX A6000 (48 GiB). All 15 planned actions ran, with 30 successful sample transports and no HTTP timeouts. One action returned no parseable answer. This validates the real-answer execution path; it does not establish a routing benefit.

## Results

Latency includes generation, extraction and voting, and both stages for deep reasoning. Medians and ranges below include the no-answer action. The fixed diagnostic target was 87 seconds, with a 180-second observation limit.

| Action (N, thinking budget) | Parsed answers | Correct | Correct within 87s | Median latency | Observed latency range | Total completion tokens |
|---|---:|---:|---:|---:|---:|---:|
| (1, 0) | 4/5 | 1/5 | 1/5 | 3.65 s | 3.46–13.23 s | 4,323 |
| (4, 0) | 5/5 | 3/5 | 3/5 | 8.87 s | 4.92–16.00 s | 16,994 |
| (1, 2048) | 5/5 | 3/5 | 3/5 | 13.38 s | 13.31–13.63 s | 10,346 |

Tokens are summed across all samples and stages, not GPU cost. The machine-generated summary also contains completed-only sample p95; with five trials, use the observed ranges here rather than interpreting it as a reliable tail-latency estimate.

| Question | Gold | (1, 0) | (4, 0) | (1, 2048) |
|---|---|---|---|---|
| geometry/226 | 180 degrees | 90 ✗ | 180 ✓ | 180 ✓ |
| algebra/1349 | Evelyn | Evelyn ✓ | Evelyn ✓ | Evelyn ✓ |
| number_theory/183 | 14 | 9 ✗ | 20 ✗ | 16 ✗ |
| counting_and_probability/430 | 1/3 | 1/6 ✗ | 1/6 ✗ | 0 ✗ |
| intermediate_algebra/966 | 16 | no answer ✗ | 16 ✓ | 16 ✓ |

## What the raw outputs show

- Single shallow reasoning hit its 2048-token cap on the last question without a boxed answer. This counts as a delivery and accuracy miss despite finishing in 13.23 seconds. None of the 20 wide shallow samples hit that cap.
- Four of five deep traces hit the 2048-token thinking cap and used the forced-answer suffix. Only the algebra question stopped naturally. Two final-answer continuations hit the 48-token cap, but both contained a parseable box; one was correct and one incorrect. A cap hit alone is not a delivery failure.
- The final question's four shallow answers were `16`, `12 sqrt(2)`, `21`, and `12`. All tied; the predeclared lowest-sample-index tie rule selected `16`. That success is sensitive to tie-breaking, not majority agreement.
- On the number-theory question, one of four samples was correct (`14`), but plurality selected the incorrect `20`. Report selected-answer accuracy, not whether any sample succeeded.
- The same question/sample seed was used across strategies, but batching does not guarantee identical outputs: the geometry sample at index zero differed between N=1 and N=4. These are paired question comparisons, not a nested set of identical generated traces.

## Interpretation and next experiment

The two-stage deep-answer path works, and all parsed answers met the diagnostic target on this idle A6000. This does not resolve the earlier T4 deep-request timeouts: the hardware and EOS-aware real-answer workload differ. Do not pool those latency measurements.

The five questions were selected before generation, one per level, using seed 20261115. They are development diagnostics with one seed per sample, not a clean held-out accuracy benchmark. Both richer actions corrected the same two single-shallow failures here; this is not evidence of a load-dependent strategy crossover. No background traffic or router was evaluated.

Next, use the cached environment to run the same real-answer actions against matched low/high background arrival traces on the same GPU. Calibrate load on that GPU first; freeze the arrival traces, action order and evaluation target before comparing actions. Measure target correct-and-on-time counts together with background SLO attainment, retaining missing/late answers as misses. Add more preselected questions and independent seeds before claiming a strategy ranking. Only after finding a useful feasible tradeoff should we compare a load-aware router with a fixed-cost router and fixed actions under identical traffic. Keep router fitting and operating-point selection separate from final evaluation questions.

## Reproducibility and resources

- Qwen/Qwen3-1.7B revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; FP16; vLLM `0.32.0+cu129`; prefix caching disabled; exact command and installed environment included.
- Execution code commit: `7c23a36`. The archive includes the exact runner, scoring helpers, inputs, raw requests/responses/token IDs, scored outputs and server/install logs.
- Scoring: math-verify 0.9.0 plus the existing project's normalized-string fallback. The five selected predictions were also checked against saved gold values in the table above.
- Successful job `13888804` completed in 6m05s, including setup. Slurm confirmed completion with exit 0 and no remaining job in the user queue.
- All six setup/execution allocations total **56m45s** by parent-job accounting, or **58m17s** conservatively including cancellation cleanup (maximum elapsed job step per allocation). This is below the authorized 120-minute cumulative cap. No further GPU allocation was started.

| Job | Outcome | Parent elapsed | Including cleanup |
|---|---|---:|---:|
| 13886133 | Cancelled: default CUDA 13 dependency mismatch | 1m46s | 2m17s |
| 13886164 | Cancelled: slow serial dependency download | 29m10s | 29m41s |
| 13887170 | Failed: NVIDIA package download timeout | 2m55s | 2m55s |
| 13887790 | Cancelled: move downloads off GPU allocation | 16m08s | 16m38s |
| 13888699 | Failed: optional torchcodec required CUDA 13 | 41s | 41s |
| 13888804 | Completed all planned answer trials | 6m05s | 6m05s |

The final run used a verified offline wheel/model cache. The official CPU torchcodec build avoids an unused audio/video dependency's CUDA 13 import; text inference still runs on CUDA 12.9. The first five attempts did not produce scored answer trials.

See [trial-results.json](trial-results.json) for compact per-action results, [raw-evidence.zip](raw-evidence.zip) for the full evidence, [manifest.json](manifest.json) for artifact hashes, and [accounting.txt](accounting.txt) for Slurm records.
