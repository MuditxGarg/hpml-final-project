# Real Answer Completion Validation

This is the next diagnostic after the Matched Serving Interference Study. It generates and scores actual answers on an idle server before introducing background traffic. It does not evaluate a router.

## Fixed pilot

Five questions, one from each MATH difficulty level, selected before seeing outputs using seed 20261115. Compare `(N=1,B=0)`, `(4,0)`, `(1,2048)` on every question with randomized action order. The cohort and order live in the saved `plan.json`; `gold.json` is only read by offline scoring. These questions are development diagnostics, not a clean held-out benchmark.

- No-thinking: original saved template, EOS enabled, 2048-token output cap.
- Thinking: original labeler's explicit opening marker, maximum 2048 thinking tokens, stop at `</think>`, then its natural/forced answer suffix and greedy 48-token final answer. Both HTTP stages count toward latency. Exact returned token IDs are used for continuation.
- Plurality among parsed normalized boxes, ties by first sample index. All N samples must finish; no valid answer is a delivery failure. This deterministic tie rule differs from the legacy Monte Carlo label evaluator's random tie rule, so the small cohort's accuracy is not directly comparable to its estimates.
- Fixed 87-second diagnostic latency target, 180-second absolute observation limit for an entire action, including both stages and concurrent samples. Late correct answers remain SLO misses. Offline symbolic verification is excluded from serving latency; extraction/voting is included.
- Server must be idle before and after each action; prefix caching disabled. A short unscored warmup precedes the run. No forced `ignore_eos` replay.

## Prepare and run

From the repository root:

```sh
python tcr_pipeline_full/tcr_pipeline/real_answer_validation.py prepare \
  --prompts evaluation-resources/t4-features/prompts.json \
  --labels tcr_pipeline_full/tcr_pipeline/tcr_results/main/rollouts \
  --out evaluation-resources/real-answer-validation
```

The portable bundle has `code/`, `input/{plan,gold}.json`, and `run_real_answers.sbatch`. Submit from its directory on Insomnia with `sbatch run_real_answers.sbatch`. The wrapper requests one GPU, 8 CPUs, 32 GiB, 55 minutes for the corrected CUDA-12.9 job (after cancelling an incompatible 1m46s setup, below 60 minutes total) using `edu/short/free_short`. Installation and execution occur on the compute node in `/tmp`, matching the previously verified cluster convention. It saves raw results and installation/server logs to `results/job-ID/` even on failure. No software runs on the login node beyond file transfer and Slurm commands.

Runner CLI for an existing dedicated server:

```sh
python code/real_answer_validation.py run --plan input/plan.json --out run \
  --deadline-utc YYYY-MM-DDTHH:MM:SS+00:00
python code/real_answer_validation.py score --run run --gold input/gold.json
```

Set deadline from the actual allocation start, not from server readiness. The runner only starts a question when all three worst-case actions, idle checks and a 3-minute reserve fit. Slurm separately enforces the allocation deadline. A deadline-guarded run can be incomplete and must be reported that way. Scoring requires math-verify; no silent package-missing exact-match fallback.

## Read results

`raw.jsonl` retains requests, stage responses/text/token IDs, finish reasons, errors, timings and sample seeds. `scored.json` adds correctness and on-time correctness. `summary.json` and `report.md` separate planned/observed counts, failures, correct answers and on-time answers. `status.json` records completed and skipped blocks. Examine cap hits and missing boxes, not only accuracy. With five serial questions, do not claim reliable p95, throughput, significance or background-SLO feasibility.

School GPU latency must not be pooled with previous T4 measurements. Record actual hardware and installed versions. An 87s target is retained for a transparent diagnostic comparison, not declared a suitable production SLO on the new hardware.

API reference: [vLLM completion protocol](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/completion/protocol/) documents `return_token_ids`; [vLLM 0.32 installation](https://docs.vllm.ai/en/v0.32.0/getting_started/installation/gpu/) documents CUDA wheel compatibility. Actual integration is checked by the live run.

## Current execution record

2026-11-12: initial job 13886133 cancelled during dependency setup after 1m46s, before model execution. Driver575.57.08 supports CUDA12.9; PyPI default began resolving CUDA13 dependencies. Corrected job13886164 uses the official vLLM0.32.0+cu129 wheel and the PyTorch cu129 index, starts20:06:51 America/New_York, Slurm deadline21:01:51. Automatic requeue disabled. Actual submission script is run_real_answers_cuda129.sbatch; retain it with downloaded evidence.
