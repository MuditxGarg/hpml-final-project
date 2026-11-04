# LATTE pilot (Qwen3-1.7B, 150 MATH-500 questions, RTX 4080 laptop)

Run: `python run_local.py main` (about 90 min on the laptop). Smoke test: `python run_local.py smoke`.

- `stage_rollouts.py`: 16 traces per question, cut at every thinking budget, forced answer
- `stage_probe.py`: hidden states for the probe router
- `stage_load.py`: vLLM concurrency sweep (load vs GPU cost)
- `summarize.py`: builds `results/report/report.md`

Read `results/report/report.md` first. Preliminary: one model, one benchmark, 150 questions, no confidence intervals yet.
