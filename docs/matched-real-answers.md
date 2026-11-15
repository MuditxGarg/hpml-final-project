# Matched real-answer comparison

`15_matched_real_answers.py` compares fixed actions `(N, B)` on real, scored answers while the server carries matched background traffic at the two rates frozen by the [sustained calibration](sustained-calibration.md). It is step 3 of the mid-point baseline and the go/no-go test of the project: does the best fixed action change between light and heavy load by more than noise?

The logic is in `matched_real_answers.py`. It reuses the trace generator, cell order, runner and idle check of `matched_serving.py`, the answer generation and voting of `real_answer_validation.py`, the grader of `answer_scoring.py` and the sampler of `profiling_pass.py`. It never launches a server. **No router is evaluated.**

## Design

- **Questions.** `prepare` draws a fixed cohort with an equal number of questions per MATH level (default 50, ten per level) from id-sorted pools with a seeded generator. The cohort is written to `plan.json` before any run and is never chosen from outcomes.
- **Target stream: real answers.** Each question is one target job per cell. An action with `N` samples sends `N` parallel requests with different seeds; the job's answer is the plurality of the valid boxed answers and the job finishes when the slowest sample does. `B=0` is one non-thinking completion capped at 2048 tokens. `B>0` thinks up to `B` tokens, then asks for the final answer in a second request, as in the original labelling. Generation is EOS-aware.
- **Background stream: the calibrated synthetic stream.** Poisson arrivals at the frozen rate, single requests at the reference budget with replayed lengths and `ignore_eos`, using prompts of questions outside the cohort. This is exactly the traffic the rates were calibrated on, and it is byte for byte the same in every cell of a block, which real background answers could not guarantee.
- **Blocks and cells.** A block is one seed at one rate. It holds a background-only control and every action, in a seeded random order. The order of the two rates is also shuffled per seed. Target jobs are evenly spaced over the window in a seeded question order, and that schedule is the same at both rates. The window length is the number of questions divided by the target rate (`--target-fraction`, default 0.05, of the contended rate).
- **Sample seeds** depend on the block seed, the question and the sample index only, so sample 0 of `(4,0)` uses the same seed as the single sample of `(1,0)`. Batching can still change sampled tokens.
- All traces and cell orders are written before the first cell runs. The server must be idle before and after every cell. `--deadline-epoch` stops the run before a block that could not finish; only complete blocks are scored.

## Metrics (`score`)

Scoring is offline and requires `math_verify`, the grader used for the labels.

- **Correct within target:** the job completed, its voted answer is verified correct, and its latency from scheduled arrival to the last sample is within the latency target. Every scheduled job is in the denominator: timeouts, errors, censored jobs and jobs with no boxed answer count as misses, and are also reported by status.
- **Per GPU-second (headline):** correct-within-target jobs divided by the GPU seconds attributed to the target jobs. A job is attributed the cell's wall time (one GPU) times its share of all tokens generated in the cell. Prompt processing is not attributed separately.
- **Goodput per GPU-second (per cell):** target jobs correct within target plus background requests completed within target, divided by the cell's wall seconds. This is the view that includes what the action does to everyone else.
- Both are reported at the tight and the loose target (2x and 4x idle p95 from the calibration).
- **Intervals:** percentile bootstrap over questions. All seeds of a question are resampled together. Differences between two actions resample the same questions for both.

`summary.json` holds per-cell counts, per-rate and per-action results with intervals, and all pairwise action differences per rate. `scored.jsonl` holds one record per target job.

## Commands

Run with a Python environment that has `transformers` (for the tokenizer) and, for scoring, `math_verify`.

```sh
python 15_matched_real_answers.py prepare \
  --prompts /absolute/path/prompts.json --labels tcr_results/main/rollouts_v2 \
  --out /absolute/path/new-plan-directory --questions 50

python 15_matched_real_answers.py run \
  --plan /absolute/path/plan-directory/plan.json \
  --calibration /absolute/path/calibration-directory/calibration.json \
  --lengths tcr_results/main/lengths_v2.json --prompts /absolute/path/prompts.json \
  --hardware-json /absolute/path/hardware.json --out /absolute/path/new-run-directory

python 15_matched_real_answers.py score \
  --out /absolute/path/run-directory --gold /absolute/path/plan-directory/gold.json
```

## Limitations

- One machine. Intervals come from resampling questions; with three seeds they say little about run-to-run variation of the server itself.
- The background is synthetic, so the comparison says how an action behaves next to calibrated traffic, not next to other users' real answers.
- The GPU-second attribution is by generated tokens. Two actions with the same token count but different batch shapes are charged the same.
- Fixed actions only. A result here cannot be described as a routing gain.
- Target jobs are spaced evenly, not Poisson, so target-on-target bursts are not exercised.

## Tests

`tests/test_matched_real_answers.py` covers cohort selection, the matched target schedule and the bootstrap with fixed inputs, and runs `run` and `score` against `test_fake_vllm_server.py`, which answers every real request with a fixed boxed answer.
