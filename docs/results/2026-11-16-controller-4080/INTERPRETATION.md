# Reading the live controller run, RTX 4080 laptop, 2026-11-16

This file is written by hand. Every number is copied from the generated [README.md](README.md) in this folder, which is built only from logged runs. Hardware: HP Omen laptop, RTX 4080 Laptop GPU (12 GB), WSL2, vLLM 0.32.0, Qwen3-1.7B in BF16. Do not pool with another machine.

## What was run

- The same 50 questions (ten per MATH level) as the [frozen-rate comparison](../2026-11-15-baseline-4080/INTERPRETATION.md) and the [load ladder](../2026-11-15-load-ladder-4080/INTERPRETATION.md), on three new seeds (20261120 to 20261122), at four background rates: 1.231, 1.724, 2.094 and 2.463 req/s. 60 cells, all complete.
- Fixed actions `(1,0)`, `(4,0)`, `(1,2048)`, a background-only control, and one per-job policy, **`kv-ladder`**: when a target job arrives, read the KV-cache usage of the server (scraped every second, at most 5 s old) and choose `(1,2048)` below 0.5, `(4,0)` below 0.8, otherwise `(1,0)`. If no fresh reading exists, `(1,0)`.
- The thresholds were frozen before the run from the on-time rate of each fixed action by KV usage at arrival, over the earlier runs on seeds 20261116 to 20261118 (`policy/freeze.json`). They were not tuned on this run.
- Code at commit `86f8ba9`. Evidence in `controller/`.

## What the controller did

Actions chosen, out of 150 jobs per rate:

| Rate (req/s) | `(1,2048)` | `(4,0)` | `(1,0)` |
|---|---|---|---|
| 1.231 | 150 | 0 | 0 |
| 1.724 | 135 | 15 | 0 |
| 2.094 | 62 | 83 | 5 |
| 2.463 | 10 | 99 | 41 |

It moved from depth to width to the single answer as load rose, which is what the ladder said it should do. The five and 41 `(1,0)` choices at the two upper rates were all fallbacks (no fresh KV reading), not threshold decisions.

## Result: the controller matches the best fixed action at every rate, and never beats it

Correct answers within the loose target (41.8 s), 150 jobs per entry:

| Rate (req/s) | `(1,0)` | `(4,0)` | `(1,2048)` | `kv-ladder` | Best fixed minus controller, 95% interval |
|---|---|---|---|---|---|
| 1.231 | 0.827 | 0.860 | 0.840 | 0.847 | `(4,0)`: +0.013 [-0.053, +0.073] |
| 1.724 | 0.813 | 0.867 | 0.827 | 0.840 | `(4,0)`: +0.027 [-0.020, +0.073] |
| 2.094 | 0.820 | 0.833 | 0.660 | 0.827 | `(4,0)`: +0.007 [-0.033, +0.047] |
| 2.463 | 0.827 | 0.813 | 0.373 | 0.807 | `(1,0)`: +0.020 [-0.027, +0.073] |

- **Against the best fixed action, every interval includes zero.** The controller is not a gain by this metric; it tracks the best fixed action within noise.
- **It avoids the collapse of depth.** At 2.094 and 2.463 req/s, `(1,2048)` minus controller is -0.167 [-0.253, -0.093] and -0.433 [-0.533, -0.327]. That is the feasibility result the project predicted: the controller stops choosing depth once the server is loaded.
- **Under the tight target (20.9 s)** the controller is between the two: 0.580 and 0.667 correct within tight at the two upper rates, against 0.220 and 0.140 for depth and 0.733 and 0.633 for width.
- **Accuracy ignoring latency is the same for everything** (0.813 to 0.873, all intervals overlap). As in every run on this 50-question cohort, extra compute bought no measurable accuracy, so there was nothing for a controller to gain in accuracy. The [150-question cohort](../2026-11-16-representative-cohort-4080/INTERPRETATION.md) shows about ten points for extra compute; the controller has not been run there.

Per attributed GPU-second (comparable within a rate only):

- `(1,0)` beats the controller at every rate, with intervals that exclude zero (+1.08 to +2.54 correct on-time answers per GPU-second).
- The controller beats `(4,0)` at every rate, with intervals that exclude zero (+0.09 to +0.23), because it spends less.
- The controller ties `(1,2048)` at the three lower rates and beats it at 2.463 req/s (+0.547 [+0.271, +0.936]).

So on this cohort the controller is a safe way to spend more compute than `(1,0)`, but there is no reason to spend it: `(1,0)` delivers the same number of correct on-time answers at a third of the cost.

## Cost to everyone else

At 2.463 req/s, per seed (preemptions; background requests within the loose target out of the block's total):

| Seed | Control | `(1,0)` | `(4,0)` | `(1,2048)` | `kv-ladder` |
|---|---|---|---|---|---|
| 20261120 | 0; 990 of 990 | 0; 990 | 16; 990 | 0; 990 | 39; 990 |
| 20261121 | 0; 1073 of 1073 | 0; 1073 | 1382; 929 | 1338; 605 | 701; 1073 |
| 20261122 | 0; 991 of 991 | 0; 991 | 30; 991 | 114; 991 | 0; 991 |

In the one seed where the server was pushed over the edge, the controller caused half the preemptions of width and kept every background request inside the target, where width lost 144 and depth lost 468. At the three lower rates no cell of any action had a preemption or a waiting request.

## Checks on the run itself

- **GPU power.** `nvidia-smi` timed out in 13 of 60 cells, so their power is not recorded; the other 47 cells averaged 105 to 115 W, the normal range. The background-only controls give the same answer at every rate: p95 14.7 / 15.6 / 15.2 s (light), 17.1 / 18.6 / 17.8 s (1.724), 19.7 / 21.9 / 20.9 s (2.094), 22.4 / 28.0 / 24.7 s (2.463) across the three seeds, in line with the normal-power cells of the earlier runs. Seed 20261121 ran a little slower at the top rate.
- **Fallbacks.** 46 of 600 controller jobs found no KV reading under 5 s old and fell back to `(1,0)`, all at the two upper rates. The metrics scrape slows down when the server is loaded. A production controller would keep the last reading instead.
- **Thresholds.** The freeze used the six reduced-power ladder cells of seed 20261118 among its inputs, which makes the depth threshold (0.5) more cautious than normal-power data alone would give.

## What this does and does not establish

Established on this machine, with real answers and real queueing:

- A controller reading one server signal at arrival follows the affordable action set as load changes, and matches the best fixed action at each of four rates within noise, without being told the rate.
- It avoids the on-time collapse of depth and loads the server less than always choosing width.

Not established:

- Any gain over the best fixed action, or over a load-blind mixture. The **fixed-mixture baseline** with the shares the controller used (7.7% `(1,0)`, 59.5% `(1,2048)`, 32.8% `(4,0)`, pooled over the four rates) is running on the same seeds and rates and must be reported next to these numbers before any of them is quoted as a controller result.
- Anything about accuracy. On this cohort no action is more accurate than `(1,0)`.

## Cautions

- 50 questions with equal numbers per MATH level; intervals resample questions, not seeds; synthetic background; evenly spaced target jobs; WSL2 laptop.
- Per-GPU-second values are comparable between actions at one rate only.
- One policy with hand-chosen rule structure (two KV thresholds) and a hand-chosen freezing rule (first bin under 0.90 on time). Neither was varied.
