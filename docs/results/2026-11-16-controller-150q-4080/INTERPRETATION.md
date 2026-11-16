# Reading the controller on the 150-question cohort, RTX 4080 laptop, 2026-11-16

This file is written by hand. Every number is copied from the generated [README.md](README.md) in this folder or from `fixed-vs-policies.json`, both built only from logged runs. Hardware: HP Omen laptop, RTX 4080 Laptop GPU (12 GB), WSL2, vLLM 0.32.0, Qwen3-1.7B in BF16. Do not pool with another machine.

## What was run

- The 150-question cohort of the [representative-cohort run](../2026-11-16-representative-cohort-4080/INTERPRETATION.md) (count per MATH level proportional to MATH-500), one seed (20261116), both frozen rates (light 1.231 req/s, contended 2.463 req/s), window 1,218 s per cell.
- **Fixed actions** (`fixed/`, commit `c758d09`): `(1,0)`, `(4,0)`, `(1,2048)` and a background-only control at each rate. This is the normal-power rerun of seed 20261116; the original block of that seed ran at reduced GPU power and is not used.
- **Policies** (`policies/`, commit `7f0fb86`, same seed, same background traces, same target schedule): a control and the two per-job policies of the [50-question controller run](../2026-11-16-controller-4080/INTERPRETATION.md), unchanged (`policy/freeze.json`):
  - `kv-ladder`: read the server's KV-cache usage at arrival (scraped every second, at most 5 s old); `(1,2048)` below 0.5, `(4,0)` below 0.8, otherwise `(1,0)`; `(1,0)` when no fresh reading exists.
  - `mix-kv-ladder`: a load-blind hash of question and seed with the shares 7.7% `(1,0)`, 59.5% `(1,2048)`, 32.8% `(4,0)`, the same assignment at both rates.
- All 14 cells complete. The paired differences between the two runs are in `fixed-vs-policies.json` (same questions, seed and traces).

## What the controller did

| Rate | `(1,2048)` | `(4,0)` | `(1,0)` |
|---|---|---|---|
| light | 150 (rule 0) | 0 | 0 |
| contended | 4 (rule 0) | 74 (rule 1) | 72 (all fallbacks: no KV reading under 5 s old) |

At the contended rate the "otherwise `(1,0)`" rule never fired on a fresh reading. Every `(1,0)` came from the fallback, because the metrics scrape goes stale exactly when the server is busiest. The staleness of the metrics endpoint acted as the load signal. That worked here, but it is not the mechanism the policy was written for.

## Result 1: within noise of the best fixed action at both rates, not better

Correct answers within the loose target (41.8 s), 150 jobs per entry:

| Rate | `(1,0)` | `(4,0)` | `(1,2048)` | `kv-ladder` | `mix-kv-ladder` |
|---|---|---|---|---|---|
| light | 0.720 | 0.740 | 0.747 | 0.760 | 0.753 |
| contended | 0.707 | 0.400 | 0.127 | 0.680 | 0.147 |

Paired differences, controller minus fixed action, 95% interval:

| Rate | minus `(1,0)` | minus `(4,0)` | minus `(1,2048)` |
|---|---|---|---|
| light | +0.040 [-0.033, +0.113] | +0.020 [-0.060, +0.093] | +0.013 [-0.040, +0.060] |
| contended | -0.027 [-0.087, +0.027] | +0.280 [+0.200, +0.367] | +0.553 [+0.473, +0.633] |

- At the light rate the controller sent every job to depth and delivered 0.760, the highest value in the table, but no interval against a fixed action excludes zero.
- At the contended rate it stayed within noise of `(1,0)` and well above width and depth, which both collapsed on this cohort.
- **Accuracy ignoring latency is the same for everything on this seed:** 0.713 to 0.760, all intervals overlap. `(1,0)` minus `(4,0)` is -0.020 [-0.073, +0.033] at the light rate. The three-seed representative run estimated about ten points for extra compute; this one seed shows two to four and cannot resolve them. The three-seed assembled run of the fixed actions, which follows tonight, is the better estimate.

Per attributed GPU-second (within a rate only): `(1,0)` beats the controller at both rates (+0.627 [+0.435, +0.847] light, +1.406 [+1.095, +1.770] contended). The controller beats `(4,0)` at both rates and `(1,2048)` at the contended rate; it ties `(1,2048)` at the light rate, where it is `(1,2048)`.

Under the tight target (20.9 s) the controller is tuned wrongly: at the light rate it chose depth for every job and delivered 0.207 against 0.700 for `(1,0)`. The thresholds were frozen against the loose target; a tight-target controller would need to exclude depth outright.

## Result 2: reacting to load beats the same blend applied blindly, by 53 points at the contended rate

| Rate | `mix-kv-ladder` | `kv-ladder` | Controller minus mixture, 95% interval | Per GPU-second, 95% interval |
|---|---|---|---|---|
| light | 0.753 | 0.760 | +0.007 [-0.047, +0.060] | +0.112 [+0.041, +0.179] |
| contended | 0.147 | 0.680 | +0.533 [+0.447, +0.613] | +0.759 [+0.576, +1.002] |

The mixture kept sending 57% of jobs to depth at the contended rate; 1 of its 85 depth jobs met the loose target (mean latency 108 s) and 16 of its 52 width jobs. Its accuracy ignoring latency was the same as the controller's (0.733 both). The gap is larger than on the 50-question cohort (+0.260 at this rate) because width is no longer affordable here either.

## Result 3: the affordable set at the contended rate is narrower on this cohort than on the 50-question one

On the 50-question cohort `(4,0)` stayed affordable at 2.463 req/s (0.813 correct within loose, 36 to 125 preemptions per cell). On this cohort, same rate and same machine, fixed `(4,0)` delivered 0.400 with 5,314 preemptions, a background p95 of 72.5 s (control 26.7 s) and only 616 of 3,031 background requests within the loose target. Two things changed: the window is three times longer (1,218 s against 406 s), so queues have time to build, and the proportional cohort has more level-4 and level-5 questions with longer answers.

The consequence for the controller: its width threshold (KV usage below 0.8) was frozen on 50-question data where width was affordable, and it admitted width 74 times here. 48 of those 74 jobs met the loose target, better than fixed width (60 of 150) because they were interleaved with cheaper fallback jobs, but 26 missed. A threshold frozen on one cohort does not transfer to another on the same machine; the controller needs a signal that reflects the current cost of width, not a fixed KV level.

## Cost to everyone else

Contended rate, seed 20261116 (control: p95 26.2 and 26.7 s in the two runs, no preemptions):

| Entry | Background p95 (s) | Preemptions | Background within loose, of 3,031 | Waiting max |
|---|---|---|---|---|
| `(1,0)` | 32.7 | 310 | 3,031 | 21 |
| `kv-ladder` | 41.8 | 2,552 | 2,881 | 59 |
| `(1,2048)` | 63.0 | 3,768 | 1,770 | 122 |
| `(4,0)` | 72.5 | 5,314 | 616 | 172 |
| `mix-kv-ladder` | 97.6 | 4,976 | 417 | 215 |

The controller loaded the server less than width, depth or the mixture, and more than `(1,0)`: 150 background requests missed the loose target in its cell, none in the `(1,0)` cell. Its background p95 sits on the loose target itself (41.8 against 41.84 s). At the light rate no cell of either run had a preemption or a waiting request.

## Checks on the runs themselves

- **GPU power** was recorded in 3 of 6 policy cells (103 to 107 W, normal) and all 8 fixed cells (102 to 115 W); `nvidia-smi` timed out in the other three. The controls agree between the two runs (15.9 and 16.0 s light, 26.2 and 26.7 s contended), so the machine was in the same state.
- **The two runs were not interleaved.** The policies ran about three hours after the fixed actions, on the same server process. The controls above are the check.
- **One seed.** Intervals resample questions only. The seed-to-seed variation seen on the 50-question cohort (for example 0 to 1,382 preemptions for `(4,0)` at this rate) is not bounded here.
- **Fallbacks.** 72 of 150 controller jobs at the contended rate had no fresh KV reading, against 46 of 600 in the 50-question run. The longer window keeps the server saturated for longer, and the scrape is slow while it is.

## What this does and does not establish

Established on this machine, one seed, with real answers and real queueing:

- On the cohort where extra compute is expected to matter, the controller again matches the best fixed action at each rate within noise and avoids the collapse of both width and depth at the contended rate.
- It beats the load-blind mixture of the same actions by 53 points of correct on-time answers at the contended rate, with a lower background cost.
- Which actions are affordable at a given rate depends on the cohort and the window, not only on the rate, so thresholds frozen on one cohort are not a general controller.

Not established:

- Any gain over the best fixed action. At the light rate the controller is 1.3 to 4.0 points above the fixed actions with intervals that include zero; at the contended rate it is 2.7 points below `(1,0)`.
- That extra compute raises accuracy on this seed. The accuracy differences are within noise; the three-seed assembled run is needed for that.
- That the controller's threshold decisions, rather than its fallbacks, are what kept it on time at the contended rate.

## Cautions

- One seed; synthetic background; evenly spaced target jobs; WSL2 laptop at its thermal limit.
- Per-GPU-second values are comparable between entries at one rate only.
- The policies were frozen on the 50-question cohort and not changed; this is the right test of transfer, and the transfer was partial.
