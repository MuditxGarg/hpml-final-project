# Reading the assembled three-seed run on the 150-question cohort, RTX 4080 laptop, 2026-11-16

This file is written by hand. Every number is copied from the generated [README.md](README.md) in this folder, which is built only from logged runs. Hardware: HP Omen laptop, RTX 4080 Laptop GPU (12 GB), WSL2, vLLM 0.32.0, Qwen3-1.7B in BF16. Do not pool with another machine.

## What was run, and which blocks were repeated

The [first run on this cohort](../2026-11-16-representative-cohort-4080/INTERPRETATION.md) had twelve of its 24 cells at about two thirds of the usual GPU power, so only one block per rate was usable for latency. The affected blocks were repeated under new output names at normal power, and the six clean blocks were put together with `assemble`:

| Block | Taken from | Commit | GPU power of its cells |
|---|---|---|---|
| seed 20261118, light | original run `matched-real-answers-4080-150q-20261115` | `38a0f4e` | 114.9 W |
| seed 20261118, contended | original run | `38a0f4e` | 114.8 to 114.9 W |
| seed 20261116, light | rerun `…-rerun-seed20261116` | `c758d09` | 102.5 to 112.8 W |
| seed 20261116, contended | rerun `…-rerun-seed20261116` | `c758d09` | 111.6 to 114.6 W |
| seed 20261117, light | rerun `…-rerun-seed20261117-light` | `c758d09` | 104.6 to 112.8 W, one cell unrecorded |
| seed 20261117, contended | rerun `…-rerun-seed20261117-contended` | `2473f46` | 112.2 W for the control, three cells unrecorded |

Blocks were repeated because of the machine's power state, not because of their results; the reduced-power cells are kept in the original run directory and are not used here. The reruns reuse the plan, seeds, traces and schedules of the original run; the manifest in `cohort/manifest.json` lists the source of every block. Where `nvidia-smi` timed out, the background-only control of the block is the speed check: control p95 15.5 / 16.0 / 16.4 s at the light rate and 23.5 / 26.7 / 26.8 s at the contended rate, all in the normal-power range.

150 questions with the MATH level mix of the full 500, 3 seeds, the two frozen rates (light 1.231 req/s, contended 2.463 req/s), actions `(1,0)`, `(4,0)`, `(1,2048)` and a control per block; 24 cells, 450 target jobs per rate and action. Latency targets tight 20.92 s and loose 41.84 s. No target job timed out, errored or was dropped; 41 and 43 `(1,0)` jobs and 13 and 14 `(4,0)` jobs at the two rates produced no boxed answer and count as wrong.

## Accuracy: extra compute is worth six to eight points on this cohort

Accuracy ignoring latency, 450 jobs per entry:

| Rate | `(1,0)` | `(4,0)` | `(1,2048)` |
|---|---|---|---|
| light | 0.693 [0.627, 0.756] | 0.758 [0.696, 0.816] | 0.776 [0.711, 0.836] |
| contended | 0.693 [0.629, 0.753] | 0.756 [0.691, 0.813] | 0.751 [0.687, 0.811] |

| Paired difference, light rate | Value, 95% interval |
|---|---|
| `(1,0)` minus `(1,2048)` | -0.082 [-0.136, -0.031] |
| `(1,0)` minus `(4,0)` | -0.064 [-0.100, -0.031] |
| `(1,2048)` minus `(4,0)` | +0.018 [-0.031, +0.071] |

Both ways of spending extra compute raise accuracy, intervals exclude zero, and they are indistinguishable from each other. The first run's estimate of about ten points came from blocks at mixed power; this one is the estimate to quote.

## Go/no-go: the best fixed action changes with load, by more than noise

Correct answers within the loose target (41.8 s):

| Rate | `(1,0)` | `(4,0)` | `(1,2048)` |
|---|---|---|---|
| light | 0.693 [0.633, 0.753] | 0.758 [0.693, 0.813] | 0.776 [0.716, 0.833] |
| contended | 0.687 [0.622, 0.751] | 0.456 [0.396, 0.516] | 0.144 [0.107, 0.187] |

| Paired difference, correct within loose | light | contended |
|---|---|---|
| `(1,0)` minus `(1,2048)` | -0.082 [-0.133, -0.033] | +0.542 [+0.482, +0.602] |
| `(1,0)` minus `(4,0)` | -0.064 [-0.100, -0.031] | +0.231 [+0.184, +0.276] |
| `(1,2048)` minus `(4,0)` | +0.018 [-0.033, +0.069] | -0.311 [-0.358, -0.262] |

- **At the light rate, every correct answer arrives in time, so the extra-compute actions deliver six to eight more correct on-time answers per hundred than `(1,0)`.** Depth and width tie.
- **At the contended rate, `(1,0)` is the best action by 23 points over width and 54 over depth.** Both extra-compute actions lose more to the latency target than they gain in accuracy.
- Every interval in the table excludes zero except depth against width at the light rate. **By correct on-time answers, the best fixed action changes between the two rates by more than noise.** This is the go/no-go test of the project, answered yes on this machine.
- **By the per-GPU-second headline, `(1,0)` is best at both rates**, with every interval excluding zero (light: +0.598 [+0.417, +0.807] against depth, +0.840 [+0.684, +1.034] against width; contended: +2.117 and +1.923). Six to eight points of accuracy do not pay for two to three times the attributed GPU time. The trade-off is between correct on-time answers and compute, and which side wins depends on which the operator is short of.
- **Under the tight target (20.9 s)** `(1,0)` is best or tied at both rates: 0.684 against 0.671 for width and 0.216 for depth at the light rate; 0.587 against 0.202 and 0.016 at the contended rate. A 2048-token thinking trace does not fit in twice the idle p95 even on an idle server.

## Cost to everyone else, contended rate

Per seed (preemptions; background requests within the loose target out of the block's total):

| Seed | Control | `(1,0)` | `(4,0)` | `(1,2048)` |
|---|---|---|---|---|
| 20261118 | 0; 3028 of 3028 | 0; 3028 | 3384; 1811 | 1320; 3016 |
| 20261116 | 0; 3031 of 3031 | 310; 3031 | 5314; 616 | 3768; 1770 |
| 20261117 | 0; 3098 of 3098 | 2089; 2917 | 4187; 935 | 4974; 678 |

- Width and depth both fill the KV cache in every seed and push between 12 and 2,420 background requests per cell past the loose target. Background p95 rose from 23.5 to 26.8 s (controls) to 65 to 104 s under width and 36 to 130 s under depth.
- `(1,0)` left the background inside the target in two seeds and pushed 181 requests past it in seed 20261117, which was the heaviest seed for every action.
- Seed-to-seed variation is large: `(1,2048)` kept 40, 19 and 6 of 150 jobs on time; `(4,0)` 90, 60 and 55. Intervals here resample questions, not seeds, and three seeds do not bound this variation.
- At the light rate no cell had a preemption or a waiting request; every background request met the loose target.

## What this changes against the 50-question runs

- On the 50-question cohort (ten per level, 406 s windows) `(4,0)` stayed affordable at the contended rate (0.813 correct within loose) and extra compute bought no measurable accuracy. Here, with the full level mix and 1,218 s windows, width collapses at the contended rate and extra compute buys six to eight points at the light rate. Both the affordability of an action and the value of extra compute depend on the cohort and the window length, not only on the rate.
- The [controller run on this cohort](../2026-11-16-controller-150q-4080/INTERPRETATION.md) (one seed, on the controller branch) should be read against these three-seed fixed-action numbers.

## Cautions

- Blocks from four run directories, assembled after the fact. Each block is a complete seed at one rate with its own control, so nothing within a block mixes machine states, but the blocks ran on different days and the server process was restarted between the original run and the reruns.
- GPU power is unrecorded in four cells (`nvidia-smi` timed out under load); their controls are the check.
- Per-GPU-second values are comparable between actions at one rate only.
- Synthetic background; evenly spaced target jobs; WSL2 laptop at its thermal limit; one model and one dataset.
