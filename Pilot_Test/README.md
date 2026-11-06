
# LATTE pilot (superseded by `../tcr_pipeline`)

A first, small test of the idea: can a probe on the prompt's hidden state pick how much test-time compute to spend, and does the cost of that compute depend on server load?

**Setup:** Qwen3-1.7B, 150 MATH-500 questions, RTX 4080 Laptop GPU (12 GB), vLLM. Actions are (N samples, B thinking budget) with N in {1,2,4,8} and B in {0,256,512,1024,2048}.

## What it did

1. `stage_rollouts.py`: 16 traces per question, cut at every budget, forced answers, correctness labels (about 50 min).
2. `stage_probe.py`: last-token hidden states for a linear success probe.
3. `stage_load.py`: vLLM concurrency sweep to see how latency and GPU cost change with load (about 30 min).
4. `summarize.py`: writes `results/report/report.md`.

## Results (preliminary: one model, 150 questions, no confidence intervals)

- Single-sample accuracy by budget: 0.608 / 0.219 / 0.402 / 0.621 / 0.771 for B = 0 / 256 / 512 / 1024 / 2048.
- The probe predicts success with AUROC 0.83, against 0.75 for prompt length alone.
- Latency and GPU cost diverge under load: 16 x 128 tokens delivers about 5.8k tok/s, 1 x 2048 about 1.6k tok/s, because long requests fill the KV cache.
- A router that knows per-question difficulty has large headroom over the best fixed action (oracle gain averaging 10 to 19 points depending on the cost model).
- Ignoring load costs up to about 4 accuracy points in the static comparison, so the live-server test (done in `../tcr_pipeline`) is what matters.

## Run

```bash
python run_local.py smoke     # plumbing check
python run_local.py main      # about 90 min on the laptop
```

Start with `results/report/report.md`.

## Known limits

This pilot used a closed-loop load test without background traffic, so it could not show the effect of one request's compute on others. `../tcr_pipeline` replaces it with an open-loop sweep with background traffic, 500 questions, and censored requests counted as misses.
