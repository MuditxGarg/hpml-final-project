# Reading the 150-question representative run, RTX 4080 laptop, 2026-11-16

This file is written by hand. Every number is copied from the generated [README.md](README.md) in this folder, which is built only from logged runs, unless it is marked as arithmetic on two of its columns. Hardware: HP Omen laptop, RTX 4080 Laptop GPU (12 GB), WSL2, vLLM 0.32.0, Qwen3-1.7B in BF16. Do not pool with another machine.

## What was run

- 150 questions drawn with the MATH level mix of the full 500 (13, 27, 32, 38 and 40 questions for levels 1 to 5), fixed before the run. The earlier 50-question cohort had ten per level and could not detect accuracy differences.
- 3 seeds, the two frozen background rates (light 1.231 req/s, contended 2.463 req/s), actions `(1,0)`, `(4,0)`, `(1,2048)` plus a background-only control. 24 cells, all complete.
- The target stream has the same rate as before, so the measurement window is 1,218 s per cell instead of 406 s.
- Code at commit `38a0f4e`. Evidence in `cohort/`.

## The GPU ran at reduced power for half of the run

Mean GPU power per cell is in the per-cell table of the README.

| Cells, in run order | Mean GPU power | Background p95 of the control |
|---|---|---|
| seed 20261116, contended (4 cells) | 87.5 to 88.6 W | 39.1 s |
| seed 20261116, light (4 cells) | 71.2 to 76.3 W | 21.1 s |
| seed 20261117, light (4 cells) | 70.6 to 74.9 W | 23.5 s |
| seed 20261117, contended (4 cells) | 110.2 W for the control, then 114.6 to 114.9 W | 39.6 s |
| seed 20261118, light (4 cells) | 114.9 W | 15.5 s |
| seed 20261118, contended (4 cells) | 114.8 to 114.9 W | 23.5 s |

- The first twelve cells ran at about two thirds of the usual power and were slower for every request, including the controls. The cause is outside the experiment (laptop power state); it is not recorded.
- **Latency results pooled over the three seeds are therefore not usable**: correct within target, per-GPU-second values and their intervals in the README mix a slow machine and a normal one.
- The contended control of seed 20261117 ran while power was still recovering.
- Only the two blocks of seed 20261118 ran entirely at normal power.

## Accuracy: extra compute is worth about ten points on a representative cohort

Accuracy ignoring latency does not depend on machine speed as long as jobs complete. At the light rate every target job completed in all three seeds.

| Action | Accuracy, light rate, 450 jobs | No boxed answer |
|---|---|---|
| `(1,0)` | 0.680 [0.613, 0.744] | 38 |
| `(4,0)` | 0.778 [0.718, 0.833] | 14 |
| `(1,2048)` | 0.776 [0.711, 0.836] | 0 |

| Paired difference, light rate | Value, 95% interval |
|---|---|
| `(1,0)` minus `(4,0)` | -0.098 [-0.136, -0.060] |
| `(1,0)` minus `(1,2048)` | -0.096 [-0.151, -0.044] |
| `(1,2048)` minus `(4,0)` | -0.002 [-0.053, +0.051] |

- **Both ways of spending extra compute raise accuracy by about ten points over a single non-thinking answer, and they are indistinguishable from each other.** The 50-question cohort showed no difference; it was too small and had a different level mix.
- At the contended rate the accuracy of `(1,2048)` is 0.678 because 45 jobs timed out and 1 was dropped, all in the reduced-power block of seed 20261116. Those are counted as wrong.

## Latency: one block per rate at normal power

Seed 20261118 only, 150 jobs per cell. These are single cells, so they carry no interval.

| Rate | Action | Correct | Correct within loose | Target GPU-s | Background within loose | Preemptions |
|---|---|---|---|---|---|---|
| light | `(1,0)` | 97 | 97 | 85 | 1506 of 1506 | 0 |
| light | `(4,0)` | 120 | 120 | 288 | 1506 of 1506 | 0 |
| light | `(1,2048)` | 119 | 119 | 182 | 1506 of 1506 | 0 |
| contended | `(1,0)` | 106 | 106 | 44 | 3028 of 3028 | 0 |
| contended | `(4,0)` | 114 | 90 | 164 | 1811 of 3028 | 3384 |
| contended | `(1,2048)` | 114 | 40 | 99 | 3016 of 3028 | 1320 |

- **Light rate:** every correct answer arrived within the loose target, so the extra-compute actions delivered 120 and 119 correct on-time answers against 97. `(1,0)` is still the cheapest per correct on-time answer (97 / 85 against 119 / 182 and 120 / 288, arithmetic on the two columns).
- **Contended rate:** `(1,0)` is the only action that stayed inside the target and left the background untouched. The background-only control of this block was stable over the whole window (p95 23.5 s, no preemption, KV cache at most 0.83).
- **Width is not affordable at the contended rate over a long window.** In the 406 s runs `(4,0)` kept 0.813 of its jobs correct on time at this rate; here it kept 90 of 150 and pushed 1,217 background requests past the target. The second seed's contended block, whose action cells also ran at normal power, agrees: 58 of 150 for `(4,0)`, 10 of 150 for `(1,2048)`, 99 of 150 for `(1,0)`.
- This is the first sign of a real trade-off on this machine: at light load extra compute buys correct answers at no latency cost, at the contended rate it does not.

## What this changes

- The 406 s windows of the earlier runs understate the cost of width near capacity. Backlog keeps growing for as long as the extra stream lasts.
- The go/no-go question stays open. By correct on-time answers the best action differs between the two rates in the normal-power block (`(4,0)` or `(1,2048)` when light, `(1,0)` when contended). By the per-GPU-second headline `(1,0)` is best at both. One block per rate is not evidence beyond noise.

## Cautions

- One block per rate at normal power. The twelve reduced-power cells need a rerun under a new output name before any pooled latency result is quoted.
- The harness did not detect the power drop. Mean GPU power per cell is now written to `summary.json` and shown in the report; a run should be checked for it before its latency numbers are used.
- The last six cells of the [load ladder](../2026-11-15-load-ladder-4080/INTERPRETATION.md) were affected by the same drop.
- Per-GPU-second values are comparable between actions at one rate only.
- Synthetic background; evenly spaced target jobs; WSL2 laptop.
