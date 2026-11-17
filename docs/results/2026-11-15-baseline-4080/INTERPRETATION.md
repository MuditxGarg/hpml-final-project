# Reading the mid-point baseline, RTX 4080 laptop, 2026-11-15

This file is written by hand. Every number in it is copied from the generated [README.md](README.md) in this folder, which is built only from logged runs. Hardware: HP Omen laptop, RTX 4080 Laptop GPU (12 GB), WSL2, vLLM 0.32.0, Qwen3-1.7B in BF16. Nothing here may be pooled with results from another machine.

## The matched real-answer run

- 50 questions (ten per MATH level, fixed before the run), 3 seeds, 2 background rates (light 1.231 req/s, contended 2.463 req/s), actions `(1,0)`, `(4,0)`, `(1,2048)`, `(4,2048)` plus a background-only control per block. 30 cells, all complete; no block skipped.
- Latency targets: tight 20.92 s and loose 41.84 s (2x and 4x idle p95).
- No target job timed out, errored or was dropped. 25 of 1,200 target jobs produced no boxed answer (12 and 9 for `(1,0)` at the two rates, 1 and 3 for `(4,0)`); they are counted as wrong.
- Code at commit `abae828`. Per-job records are in `matched/scored.jsonl`, the summary in `matched/summary.json`.

## Go/no-go: does the best fixed action change with load?

**By the headline metric, no.** `(1,0)` gives the most correct answers within the loose target per attributed GPU-second at both rates: 1.73 [1.27, 2.36] at the light rate and 3.52 [2.61, 4.82] at the contended rate. Every paired difference between `(1,0)` and another action on that metric has an interval that excludes zero, at both rates.

**The reason is that extra compute bought no measurable accuracy on this cohort.** Accuracy ignoring latency was between 0.820 and 0.867 for all eight rate and action combinations, with intervals about 0.17 wide that all overlap. With equal accuracy, the cheapest action wins on any per-compute metric.

**Among the two ways of spending extra compute, the ranking does flip with load.** `(4,0)` (width) and `(1,2048)` (depth) have the same accuracy here.

| Rate | Metric | `(1,2048)` | `(4,0)` | `(1,2048)` minus `(4,0)`, 95% interval |
|---|---|---|---|---|
| light | correct within loose | 0.847 | 0.867 | -0.020 [-0.087, +0.047] |
| light | correct within loose per GPU-s | 0.724 | 0.532 | +0.192 [+0.069, +0.283] |
| contended | correct within loose | 0.427 | 0.813 | -0.387 [-0.493, -0.280] |
| contended | correct within loose per GPU-s | 0.670 | 0.916 | -0.246 [-0.497, -0.031] |

At the light rate, depth is cheaper per correct on-time answer. At the contended rate, depth misses the loose target on about half of its jobs (target p95 54 to 76 s across seeds, against 46 to 51 s for `(4,0)`), and width becomes the cheaper one. Both intervals exclude zero, the contended one only just.

**The set of affordable actions shrinks with load.** Correct within the loose target, light to contended: `(1,0)` 0.833 to 0.840, `(4,0)` 0.867 to 0.813, `(1,2048)` 0.847 to 0.427, `(4,2048)` 0.640 to 0.027.

## Cost to everyone else

- `(4,2048)` at the contended rate pushed background p95 to 144 to 162 s (control: 23 to 26 s), with 1,974 to 2,496 preemptions and up to 338 waiting requests per cell. Only 153 to 188 of about 1,000 background requests met the loose target.
- `(1,2048)` at the contended rate filled the KV cache in all three seeds (52 to 801 preemptions) and raised background p95 to 29 to 41 s. In one seed 49 of 1,021 background requests missed the loose target.
- `(4,0)` at the contended rate filled the KV cache in two of three seeds (36 and 125 preemptions); all background requests still met the loose target.
- At the light rate no cell had a waiting request or a preemption. `(4,2048)` still raised background p95 from about 15 s to 23 to 24 s.
- This agrees with the profiling pass: the collapse is KV-cache exhaustion followed by preemptions, not GPU saturation.

## What this does and does not establish

Established on this machine:

- Depth and width at equal accuracy load a shared server differently, and which one is cheaper per on-time correct answer depends on load.
- The affordable action set shrinks with load, and `(4,2048)` is not affordable at either rate under the loose target.

Not established:

- Any routing gain. Only fixed actions were run.
- That extra compute is worth buying at all on this model and cohort. `(1,0)` was best at both rates. A load-adaptive controller has something to exploit only where extra compute raises accuracy, which this run could not detect with 50 questions.

## Cautions

- **Per-GPU-second values are not comparable across rates.** GPU time is attributed by token share of a cell's wall time, and the server generates about twice as many tokens per second at the contended rate, so every token is charged about half as much. The attributed GPU seconds of `(1,0)` are 72 at the light rate and 36 at the contended rate for the same questions. Compare actions within a rate only.
- **The `(1,2048)` versus `(4,0)` flip on the per-GPU-second metric comes from latency, not from cost.** `(4,0)` is attributed about 1.39 times the GPU seconds of `(1,2048)` at both rates (244 against 175, and 133 against 95). The flip happens because `(1,2048)` stops meeting the target.
- **The tight target is nearly out of reach for thinking actions even at the light rate** (`(1,2048)` 0.360 correct within tight). A 2048-token thinking trace takes longer than twice the idle p95 of a 1024-token reference request.
- **Intervals resample questions, not server runs.** Seed-to-seed variation is visible: `(1,2048)` at the contended rate had 18, 19 and 27 of 50 jobs correct within the loose target, with 801, 460 and 52 preemptions. Three seeds do not bound that variation.
- **The cohort has equal numbers per MATH level**, so its accuracy is not the MATH-500 average and should not be compared with the 500-question label table.
- Background traffic is synthetic replay. Target jobs are evenly spaced, not Poisson. WSL2 on a laptop at 86 C; final latency numbers should come from the cluster.

## Suggested next steps (for discussion)

1. Decide whether "width versus depth under load" is the paper core, given that `(1,0)` dominates here. The flip is real on this machine but sits between two actions that are both dominated.
2. Repeat on questions where extra compute changes the answer (for example the questions the 500-question labels mark as wrong at `(1,0)` and right at `(1,2048)` or `(4,0)`, chosen from labels and not from this run), so that accuracy differences are detectable.
3. Add a rung between the light and contended rates to locate where `(1,2048)` stops being affordable.
4. Rerun the frozen protocol on the cluster for latency numbers free of WSL2 noise.
