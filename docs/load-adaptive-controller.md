# Load-adaptive controller

The first optimisation of the project: instead of one fixed action `(N, B)` for every question, a policy picks the action of each target job when it arrives, from the server's current load. It runs inside the [matched real-answer comparison](matched-real-answers.md), next to the fixed actions and on the same traces, so it is compared on equal terms.

The logic is in `load_policy.py`. `15_matched_real_answers.py run --policies policies.json` adds one cell per policy to every block; `score` reports each policy like an action.

## Why the signal is KV-cache usage

The [profiling pass](profiling-pass.md) showed that expensive actions hurt everyone when the KV cache fills: preemptions and a waiting queue start within seconds of usage reaching 0.99. GPU utilisation reads 97 to 99% at every load and carries no information. So the controller reads KV-cache usage and the number of waiting requests from `/metrics`.

## Policies

A policy file is a JSON list. Each policy gets its own cell.

```json
[
  {"kind": "threshold", "name": "kv-ladder", "fallback": "1:0",
   "rules": [{"action": "1:2048", "kv_below": 0.5, "waiting_at_most": 0},
             {"action": "4:0", "kv_below": 0.8}]},
  {"kind": "mixture", "name": "mix", "shares": {"1:0": 0.5, "1:2048": 0.5}}
]
```

- **`threshold` (load-adaptive).** Rules are tried in order; the first whose conditions all hold gives the action, otherwise the fallback. Conditions: `kv_below`, `waiting_at_most`, `running_below`. The signal is the newest successful server sample of the cell (sampled every `--metrics-interval`, default 1 s). With no sample, or one older than `max_signal_age_s` (default 5), the policy takes the fallback: it never assumes an expensive action is affordable.
- **`mixture` (load-blind baseline).** A fixed share of jobs per action, assigned by a hash of the policy name, block seed and question. The assignment ignores load and is identical at every background rate. A mixture with the shares the controller ended up using separates "reacting to load" from "using this blend of actions".

The numbers in the example are placeholders, not recommendations.

## Rules for a valid comparison

- Thresholds and mixture shares are frozen before the run and copied into the manifest. They must come from earlier runs with other seeds (calibration, profiling, earlier matched runs), never from the run being reported.
- A mixture matched to a controller's realised shares needs those shares first. Take them from a pilot run on other seeds, then run controller and mixture together.
- Every row of a policy cell records the chosen action, the rule that fired and the signal it saw (`policy_reason`, `policy_signal`). `summary.json` lists per policy how often each action and each reason occurred.
- Report the fixed actions, the mixture and the controller together. A controller that matches the best fixed action is not a gain.

## Limitations

- The decision uses load at arrival only. A job admitted while the cache has room can still be caught by a later burst; nothing is cancelled or downgraded mid-flight.
- The signal is up to one sampling interval old, and the controller's own jobs change it.
- The controller does not look at the question. It cannot gain accuracy over the best fixed action; it can only keep extra compute where the server can afford it.

## Tests

`tests/test_load_policy.py` covers rule order, the missing-signal fallback, mixture shares and reproducibility, and runs policy cells end to end against `test_fake_vllm_server.py`.
