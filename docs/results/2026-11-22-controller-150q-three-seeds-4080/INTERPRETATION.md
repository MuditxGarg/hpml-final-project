# Reading the three-seed controller run, RTX 4080 laptop, 2026-11-22

This file is written by hand. Every number is copied from the generated [README.md](README.md), `fixed-vs-policies.json`, `machine-state.json` or `policies/policy-signals-summary.json`, all built from logged runs. Hardware: HP Omen laptop, RTX 4080 Laptop GPU (12 GB), WSL2, vLLM 0.32.0, Qwen3-1.7B in BF16. Do not pool with another machine.

## What was run

- The 150-question cohort, seeds 20261116, 20261117 and 20261118, at the frozen light (1.231 req/s) and contended (2.463 req/s) background rates. Each cell has a 1,218 s window. Every block uses the same background trace and target schedule for all of its entries.
- **Policies** (`policies/`): a control, `kv-ladder` and `mix-kv-ladder`, unchanged from the [50-question controller run](../2026-11-16-controller-4080/INTERPRETATION.md) (`policy/`). `kv-ladder` reads KV-cache usage at arrival and picks `(1,2048)` below 0.5, `(4,0)` below 0.8, otherwise `(1,0)`. `mix-kv-ladder` assigns actions by a load-blind hash.
- **Fixed actions** (`fixed/`): a control, `(1,0)`, `(4,0)` and `(1,2048)`. Seed 20261116 uses its existing blocks. Seeds 20261117 and 20261118 use blocks rerun on 2026-11-21 and 2026-11-22.
- **Why the fixed actions were rerun.** The policy cells of the two later seeds ran at about 122 to 127 W. The earlier fixed cells recorded about 103 to 115 W, so the comparison could favour the policies near the KV-cache limit. The rerun and its analysis rule were written down before the rerun started (`protocol/`). The earlier comparison is kept in `earlier-fixed-vs-policies.json`.
- No target job timed out, errored or was dropped. Jobs with no boxed answer are counted as wrong.

## Result 1: the best fixed action changes with load

Correct answers within the loose target (41.8 s), 450 jobs per entry:

| Rate | `(1,0)` | `(4,0)` | `(1,2048)` | `kv-ladder` | `mix-kv-ladder` |
|---|---|---|---|---|---|
| light | 0.700 | 0.742 | 0.767 | 0.756 | 0.744 |
| contended | 0.684 | 0.520 | 0.191 | 0.693 | 0.309 |

- At the light rate, `(1,0)` is below depth by 0.067 [0.013, 0.120] and below width by 0.042 [0.007, 0.080].
- At the contended rate, `(1,0)` is above depth by 0.493 [0.442, 0.549] and above width by 0.164 [0.120, 0.209].

## Result 2: the controller stays within noise of the best fixed action at each rate

Controller minus each entry, correct within the loose target, 95% interval over questions:

| Rate | minus `(1,0)` | minus `(4,0)` | minus `(1,2048)` | minus `mix-kv-ladder` |
|---|---|---|---|---|
| light | +0.056 [+0.009, +0.104] | +0.013 [-0.040, +0.064] | -0.011 [-0.042, +0.020] | +0.011 [-0.022, +0.044] |
| contended | +0.009 [-0.027, +0.042] | +0.173 [+0.127, +0.220] | +0.502 [+0.442, +0.558] | +0.384 [+0.324, +0.442] |

- No single fixed action matches the controller at both rates. `(1,0)` is below it at the light rate, and width and depth are below it at the contended rate.
- The controller does not beat the best fixed action at either rate. Its interval against `(1,2048)` at the light rate and against `(1,0)` at the contended rate includes zero.
- Against the load-blind mixture, the controller is ahead by 0.384 at the contended rate and level at the light rate.

## Limits of the controller

- **Per attributed GPU-second, `(1,0)` is better at both rates.** `(1,0)` minus controller is +0.605 [+0.433, +0.815] at the light rate and +1.389 [+1.115, +1.696] at the contended rate. Compare these values within a rate only.
- **The thresholds suit the loose target only.** Under the tight target (20.9 s), `(1,0)` minus controller is +0.464 [+0.400, +0.524] at the light rate and +0.069 [+0.029, +0.109] at the contended rate.
- **The controller causes more preemptions than `(1,0)` in every seed.** In seed 20261116 it also pushed 150 background requests past the loose target. See the table below.

## What the controller chose

Of 450 jobs per rate:

| Rate | `(1,2048)` | `(4,0)` | `(1,0)` |
|---|---|---|---|
| light | 450 | 0 | 0 |
| contended | 23 | 251 | 176 |

All 900 controller jobs had a KV reading at most 1.012 s old, and none lacked a signal. The 176 `(1,0)` choices are fresh readings at or above 0.80.

## Cost to everyone else

Contended rate, values for seeds 20261116, 20261117 and 20261118:

| Entry | Preemptions | Background p95 (s) | Background within loose target |
|---|---|---|---|
| control | 0, 0, 0 | 26.7, 25.5, 22.7 | 3,031 / 3,098 / 3,028 (all) |
| `(1,0)` | 310, 38, 0 | 32.7, 27.9, 26.5 | 3,031 / 3,098 / 3,028 (all) |
| `kv-ladder` | 2,552, 846, 352 | 41.8, 31.6, 29.4 | 2,881 / 3,098 / 3,028 |
| `(1,2048)` | 3,768, 3,207, 457 | 63.0, 59.9, 30.4 | 1,770 / 1,609 / 3,028 |
| `(4,0)` | 5,314, 4,745, 2,923 | 72.5, 66.3, 52.1 | 616 / 1,133 / 2,438 |
| `mix-kv-ladder` | 4,976, 3,252, 1,119 | 97.6, 48.7, 35.3 | 417 / 2,394 / 3,017 |

No light-rate cell had a preemption, and every light-rate background request met the loose target.

## Checks on the runs themselves

- **Controls.** The largest difference in control p95 between the fixed and policy runs of a block is 1.53 s, below the 2 s flag set before the rerun.
- **Contended power is matched for the two rerun seeds.** Fixed cells averaged about 121 to 125 W and policy cells about 122 to 123 W.
- **Light power is not matched.** Thermal slowdown lowered the rerun light cells to about 91 to 117 W, against about 123 to 127 W for the policy cells. On the loose target this did not change the outcome measure: at the light rate every entry's correct-within-loose value equals its accuracy. **Tight-target and per-GPU-second values at the light rate are affected and should be read with that difference in mind.**
- **Server processes.** The rerun used two server processes because the first stopped during the seed 20261118 light block. Its partial block was discarded and the whole block was repeated. Both processes had a KV cache of 55,664 tokens, the same as the earlier runs.
- **Kernel.** The WSL kernel changed from 6.18.54.1 to 6.18.58.1 between seed 20261116 and the later blocks.
- **Effect of matching.** The rerun raised contended width from 0.456 to 0.520 and depth from 0.144 to 0.191, and reduced the controller's lead over them from 0.238 to 0.173 and from 0.549 to 0.502.

## What this does and does not establish

Established on this machine, with real answers and real queueing:

- The best fixed action by correct on-time answers changes between the light and contended rates.
- A controller reading one server signal stays within noise of the best fixed action at each rate, without being told the rate.
- It clearly beats the same blend of actions applied without load information at the contended rate.

Not established:

- Any gain over the best fixed action at a known, constant rate.
- Any gain per GPU-second over `(1,0)`.
- Behaviour when load changes during a run. Each cell used one constant rate.
- That the thresholds transfer to another model, cohort, latency target or machine.

## Cautions

- Intervals resample questions and do not bound seed-to-seed variation. Contended `(4,0)` scored 0.400, 0.487 and 0.673 on the three seeds.
- Background traffic is synthetic replay, and target jobs are evenly spaced.
- The laptop runs at its thermal limit under WSL2, so final latency numbers should come from the cluster.
