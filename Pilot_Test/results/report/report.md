# LATTE go/no-go report (MAIN run, model Qwen/Qwen3-1.7B)
150 prompts x 16 traces x budgets [0, 256, 512, 1024, 2048]
single-sample accuracy by budget: B=0: 0.608, B=256: 0.219, B=512: 0.402, B=1024: 0.621, B=2048: 0.771
mean tokens per trace by budget: B=0: 633, B=256: 265, B=512: 521, B=1024: 1014, B=2048: 1762
cost models: per-token decode latency = 9.8 ms + 0.00 ms x N (fit from load test); saturated GPU-seconds/token ~ 0.35 ms (fit from load test)

## Test 2 - is there headroom for a router? (cross-fitted oracle vs best fixed action)
- cost = tok: oracle gain over best-fixed envelope: max 30.5 pts, mean 10.5 pts
- cost = idle: oracle gain over best-fixed envelope: max 33.0 pts, mean 19.0 pts
- cost = sat: oracle gain over best-fixed envelope: max 28.2 pts, mean 12.9 pts
- most accurate fixed action: N=8,B=2048 at 0.808; cheapest: N=1,B=256

## Does ignoring load cost accuracy? (regret matrix, accuracy points at equal true cost)
rows = cost model the router *assumes*, columns = regime it actually runs in
| assumed \ real | idle-latency regime | saturated-GPU regime |
|---|---|---|
| tok | max 4.3 / mean 1.0 | max 0.8 / mean -0.5 |
| idle | max 0.0 / mean 0.0 | max 24.7 / mean 15.4 |
| sat | max 4.2 / mean 1.3 | max 0.0 / mean 0.0 |
- router assuming tok cost: worst-case regret across regimes = 4.3 pts
- router assuming idle cost: worst-case regret across regimes = 24.7 pts
- router assuming sat cost: worst-case regret across regimes = 4.2 pts
=> best single fixed cost model still loses up to 4.2 pts in some regime
- fixed Pareto actions under idle cost: N=1,B=256, N=4,B=256, N=1,B=512, N=4,B=512, N=8,B=512, N=1,B=0, N=2,B=0, N=4,B=0, N=8,B=0, N=1,B=2048, N=2,B=2048, N=4,B=2048, N=8,B=2048
- fixed Pareto actions under sat cost: N=1,B=256, N=1,B=512, N=1,B=0, N=1,B=1024, N=2,B=0, N=4,B=0, N=1,B=2048, N=2,B=2048, N=4,B=2048, N=8,B=2048

## Test 1 - does GPU cost diverge from latency as load rises? (equal-token actions)
| action (N x T) | tokens | idle latency (s) | max tok/s | GPU-s per request at saturation |
|---|---|---|---|---|
| 1 x 2048 | 2048 | 20.3 | 1642 | 1.25 |
| 4 x 512 | 2048 | 6.0 | 4511 | 0.45 |
| 16 x 128 | 2048 | 2.8 | 5801 | 0.35 |
| 1 x 4096 | 4096 | 41.3 | 1721 | 2.38 |
| 4 x 1024 | 4096 | 10.6 | 2849 | 1.44 |
| 16 x 256 | 4096 | 2.9 | 5431 | 0.75 |
- 4096 tokens: wide (16, 256) vs deep (1, 4096): idle-latency ratio 0.07, saturated-cost ratio 0.32, divergence D = 4.6
- 2048 tokens: wide (16, 128) vs deep (1, 2048): idle-latency ratio 0.14, saturated-cost ratio 0.28, divergence D = 2.0

## Test 3 - can a prefill hidden state predict success?
AUROC of predicted single-sample success (mean of B=512 and B=2048; computed per fold, 5-fold CV, 150 prompts):
- baseline prompt length: 0.747
- baseline MATH level (metadata): 0.723
- probe layer 0: 0.500
- probe layer 4: 0.750
- probe layer 8: 0.764
- probe layer 12: 0.797
- probe layer 16: 0.816
- probe layer 20: 0.819
- probe layer 24: 0.830
- probe layer 28: 0.831
best probe layer 28: AUROC 0.831 (length baseline 0.747)
prefill forward pass (HF, with all hidden states): median 41 ms per prompt

router driven by predictions (gain over best fixed action at equal cost, accuracy points):
- cost=tok: probe: max 19.5 | prompt length: max 14.5 | MATH level (metadata): max 13.4 | cross-fit oracle: max 30.5
- cost=idle: probe: max 18.4 | prompt length: max 12.7 | MATH level (metadata): max 10.8 | cross-fit oracle: max 33.0
- cost=sat: probe: max 20.3 | prompt length: max 14.0 | MATH level (metadata): max 9.7 | cross-fit oracle: max 28.2

## VERDICT
| check | value | threshold | result |
|---|---|---|---|
| 1. wide-vs-deep cost diverges between idle and saturated | D = 4.6 | >= 3.0 | PASS |
| 2. no fixed cost model is right at every load | best fixed model's worst regret 4.2 pts | >= 2.0 pts | PASS |
| 3. router headroom over best fixed (tokens) | max gain 30.5 pts | >= 3.0 pts | PASS |
| 4. probe beats length baseline | 0.831 vs 0.747 | +0.03 AUROC | PASS |

**GO (full LATTE)**: headroom exists and ignoring load costs accuracy. Keep the load-aware cost model as the core contribution.

Caveats: one model and one benchmark; small sample (see CIs by re-running with more prompts); the cross-fit oracle is noisy and can even sit below a probe-driven router; load costs come from a synthetic closed-loop test, not real traffic. Thresholds were fixed before the run.