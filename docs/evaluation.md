# Independent evaluation (A)

Run from the repository root with Python 3.11+:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r tcr_pipeline_full/tcr_pipeline/requirements-eval.txt
.venv/bin/python -m pytest -q
.venv/bin/python tcr_pipeline_full/tcr_pipeline/09_evaluate.py \
  --output evaluation-results/baselines-cap2048
```

The committed 500-question original and cap-check labels suffice for fixed, fixed-mixture, MATH-level and prompt-length baselines. No GPU, Hugging Face download, `tcr_config.json`, or old `oof_pred.npz` is needed. Outputs refuse overwriting an existing directory. Use a new output name for every run.

## Full probe evaluation

```bash
.venv/bin/python tcr_pipeline_full/tcr_pipeline/09_evaluate.py \
  --hidden /absolute/path/to/500-question/hidden.npz --require-hidden \
  --output evaluation-results/with-probe-cap2048
```

Feature IDs must match all 500 questions exactly; feature order is aligned by ID. The 150-question pilot is not a substitute. Missing default features omit the probe explicitly. An explicitly supplied missing/mismatched feature file fails rather than silently using another model. `07_train_probe.py` now invokes this evaluation with `--require-hidden`; `08_route_full_actions.py` invokes the same corrected evaluation. Historical implementations are preserved in `legacy/` for reference, not headline reporting. The old scripts' argument-free command is intentionally replaced with an explicit output directory.

`colab_recover_features.py` regenerates features from public Qwen3-1.7B/MATH-500 on a T4. Install `datasets`, `transformers` and `torch` in Colab first. The script exports `hidden.npz`, replay prompts and a manifest; it uses FP16 eager attention, unlike the original RTX4080 BF16 run. Always report regenerated features as a new extraction and retain its manifest. Do not interpret T4 prefill times as RTX4080 router overhead.

The recovery prompt matches original `02_probe_features.py`. Original `01_label_rollouts.py` appends another `<think>\n` after the thinking chat template, making its stored prompt lengths two tokens longer for all 500 questions. This pre-existing difference is retained rather than silently changing the label experiment. The new replay prompts use the tokenizer template once; record this distinction when comparing to the historical sweep.

## Protocol

- Default actions: N={1,2,4}, B={0,1024,2048}; B=0 uses the existing cap-check labels restricted to 2048 tokens. Old and corrected labels are never overwritten. Wider N must be requested and is an action-shape extrapolation for the measured cost curve.
- Outer five-fold splits operate on sorted question IDs. Each outer training set uses inner three-fold predictions to select hidden layer/alpha and operating points. Standardization and action mean costs use training questions only. The outer test labels/tokens are read only after policy selection.
- Layer/alpha minimize inner mean squared error across action probabilities. Ridge predicts all actions jointly. Level-bin and length baselines use the same outer folds and inner selection data.
- Fixed/fixed-mixture baselines select their action or random mixture on the training side at each load. The mixture result is its expectation, not a sampled finite assignment. It may be a strong baseline.
- The fixed-cost baseline freezes each predictor's lambda selected at background load 0.5. Load-aware policy selects lambda at each load. The predictor and token costs are otherwise identical. This isolates operating-point adaptation; it is not yet a richer load-dependent per-action GPU cost model or online p95 feedback controller.
- A policy with no feasible training point is unavailable. A selected policy that violates the constraint on held-out costs is reported as violating, not repaired using held-out outcomes. Reports retain per-fold status even if aggregate mean-token prediction is feasible.
- Bootstrap resamples paired question accuracy differences. Intervals are conditional on the fitted outer folds and observed trace pool, and do not include cost-model fitting, new trace draws, or complete retraining uncertainty. They are not proof of a live service gain.
- The separately labeled empirical hindsight reference uses outcomes and a finite lambda grid. It is not deployable and not a mathematically certified upper bound.

## Cost model and measurement meaning

The curve is fit to the independent recorded externality sweep using **background fraction + target fraction × mean target tokens / reference tokens**. The target fraction defaults to 0.05, matching the main sweep. This preserves the original background-plus-extra-stream workload; it does not replace it with an unmeasured all-router workload.

Original measurements used original `lengths.json` (B=0 cap1024). Corrected evaluation predicts the effect of new token lengths through this curve, a transfer assumption requiring real replay. Censored/error cells are excluded because their stored p95 is not an uncensored latency percentile; the count and limitation are reported. This exclusion can bias high-load estimates downward. Out-of-range offered loads return unknown rather than endpoint-clipped predictions. The curve cannot establish that width, context length and cache behavior are fully explained by tokens.

Do not call these results correct answers per GPU-second: they are expected label accuracy plus predicted background p95. End-to-end goodput needs measured GPU wall time, actual answer aggregation, and a real policy-serving run. Synthetic-length replay itself would not measure answer quality.

## Audit files

`results.json` includes source hashes, code hashes, environment versions, every fold's train/test IDs, hyperparameters, selected parameters, per-question action weights, outcomes, statuses and bootstrap summaries. `report.md` is the human-readable summary. Prior exploration of all MATH-500 questions remains prior knowledge; this protocol fixes computational leakage but does not recreate a fresh untouched benchmark.
