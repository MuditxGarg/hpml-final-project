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

The offline execution directory contains `code/`, `input/{plan,gold}.json`, `wheelhouse/`, `model/`, `requirements-offline.lock`, and the wrapper `cluster/run_real_answers.sbatch`. Submit from that directory on Insomnia. The wrapper requests one GPU, 8 CPUs, 32 GiB and at most 65 minutes using `edu/short/free_short`, with automatic requeue disabled. For a new run, obtain a fresh resource authorization and set the time limit to its remaining budget. Dependencies and the pinned model are copied into compute-node `/tmp`; the wrapper saves raw results and server/setup logs to `results/job-ID/` on exit. No ML workload runs on the login node.

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

## Completed execution

The fixed pilot completed in job 13888804 on an RTX A6000, with all 15 planned actions observed. See the [results and interpretation](results/2026-11-12-real-answer-validation/README.md), including per-question answers, cap hits, raw evidence, exact environment and all setup-attempt accounting. Total allocation including cancellation cleanup was 58m17s, below the user-authorized 120 minutes; the GPU has been released.

The final run uses offline package/model caches because compute-node downloads were slow. `cluster/requirements-offline.lock` pins the Linux/Python 3.11 environment, including official CUDA 12.9 builds and CPU torchcodec for optional audio/video imports. `UV_OFFLINE=1`, `HF_HUB_OFFLINE=1`, `--no-index`, and pinned model/tokenizer revisions prevent network setup during execution. Model LFS hashes and downloaded wheel hashes were checked; the official torchvision index had no published hash, so its local digest records transfer integrity only. Cache preparation and transfer used no allocated GPU.
