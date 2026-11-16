# Where depth stops being affordable, RTX 4080 laptop, 2026-11-15

This file is written by hand. Every number is copied from the generated [README.md](README.md) in this folder, which is built only from logged runs. Hardware: HP Omen laptop, RTX 4080 Laptop GPU (12 GB), WSL2, vLLM 0.32.0, Qwen3-1.7B in BF16. Do not pool with another machine.

## What was run

The [frozen-rate comparison](../2026-11-15-baseline-4080/INTERPRETATION.md) measured four fixed actions at 1.231 and 2.463 req/s. This run adds two background rates between them, 1.724 and 2.094 req/s, with the same 50 questions, the same three seeds, the same target schedule and the same latency targets (loose 41.84 s). Actions: `(1,0)`, `(4,0)`, `(1,2048)` and a background-only control. `(4,2048)` was left out because it already failed at both ends.

- The two rates were chosen by hand from the calibration ladder (both were stable there). They are not frozen rates of the calibration rule.
- 24 cells, all complete. No target job timed out, errored or was dropped. 16 of 900 target jobs gave no boxed answer and count as wrong.
- Code at commit `86ee512`. Evidence in `ladder/`.

## Result

Correct answers within the loose target, as a fraction of target jobs:

| Background rate (req/s) | `(1,0)` | `(4,0)` | `(1,2048)` | `(1,2048)` minus `(4,0)`, 95% interval |
|---|---|---|---|---|
| 1.231 | 0.833 | 0.867 | 0.847 | -0.020 [-0.087, +0.047] |
| 1.724 | 0.827 | 0.867 | 0.833 | -0.033 [-0.093, +0.020] |
| 2.094 | 0.827 | 0.813 | 0.560 | -0.253 [-0.347, -0.173] |
| 2.463 | 0.840 | 0.813 | 0.427 | -0.387 [-0.493, -0.280] |

- **Depth stops being affordable between 1.72 and 2.09 req/s.** `(1,2048)` matches the other actions up to 1.72 req/s and loses a third of its on-time answers at 2.09 req/s.
- **Width and the single answer stay flat** across the whole range (0.81 to 0.87).
- **Per attributed GPU-second, depth beats width at the two lower rates** (+0.192 [+0.069, +0.283] and +0.249 [+0.082, +0.359]), **ties at 2.09 req/s** (-0.029 [-0.261, +0.142]) and **loses at 2.46 req/s** (-0.246 [-0.497, -0.031]).
- `(1,0)` remains the best action per GPU-second at every rate, and accuracy ignoring latency is the same for all actions (0.807 to 0.867, overlapping intervals).

## Depth fails before the KV cache fills

At 2.09 req/s, `(1,2048)` missed the target in all three seeds (33, 30 and 21 of 50 jobs correct on time), but only the third seed had preemptions (111) or a waiting queue. In the first two seeds KV-cache usage peaked at 0.91 and 0.82 with no preemption, and the target p95 was still 50.5 s and 48.8 s.

So a long thinking trace misses the target through slower decoding with many running sequences, before the cache-exhaustion collapse seen in the profiling pass. A controller should stop choosing depth well before the cache is full.

## Cautions

- **Correction, 2026-11-16: the last six cells ran at reduced GPU power.** Mean power was 84 to 91 W in the whole 2.09 req/s block of seed 20261118 and in the last two cells of its 1.72 req/s block (control and `(1,0)`), against 117 to 132 W in the other eighteen cells (`power.draw` in the `cells.json` of the raw run). That is why this block was slower even without target jobs (control background p95 26.9 s against 20.4 s and 20.5 s), and why it is the only one with preemptions. The cause is outside the experiment. The pooled 2.09 req/s values above include this block; the two normal-power seeds alone give 33 and 30 of 50 correct on time for `(1,2048)` against 40 and 41 for `(4,0)`, so the conclusion stands.
- Run-to-run variation of the server is not covered by the intervals, which resample questions.
- Per-GPU-second values are comparable between actions at one rate only (see the frozen-rate interpretation).
- The load curve joins measured points with straight lines. `(4,2048)` was measured at the two outer rates only.
- 50 questions with equal numbers per MATH level; synthetic background; evenly spaced target jobs; WSL2 laptop.
