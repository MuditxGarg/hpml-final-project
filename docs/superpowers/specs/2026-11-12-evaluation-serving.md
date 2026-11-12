# Independent evaluation and matched serving

User authorized completing A (evaluation repair) and B (matched live experiments), sequentially or in parallel. Work in the fresh task-owned clone on `peiyan/evaluation-and-matched-serving`; preserve original data. Do not publish or start paid GPU work without available access/budget.

## A: evaluation

Add an independently runnable CPU evaluation path that consumes the committed 500-question labels and optional hidden states. Question-level outer folds evaluate models; inner out-of-fold predictions select layer, regularization, fixed action/mix and routing lambda using only outer-training questions. Fold training estimates action costs. Load-aware routing selects lambda per measured load/SLO; fixed-cost routing freezes the lambda selected at a reference load. Include fixed, fixed-mix, level-bin, optional hidden probe, and a separately labeled empirical oracle. Treat unavailable feasible points as unavailable, not as successful cheap fallbacks. Report observed test violations and calibrated-domain limits. Paired question bootstrap is conditional on the fitted folds; it does not quantify cost-model or retraining uncertainty.

Use the existing background-plus-extra-stream workload for cost prediction (background fraction + target fraction * tokens/reference). Mark token-only predictions as model-based, not real GPU performance. Do not call all-policy workload replacement calibrated. Do not extrapolate N=8/16 as measured. Use corrected B=0 labels explicitly and identify their cap. Missing hidden features must omit the probe, never substitute invented features. Existing 07/08 entry points should direct users to the corrected evaluation path without silently overwriting prior reports.

## B: matched experiments

Add deterministic persisted arrival/prompt/length traces, shared across candidate actions and control, with separate background and target RNG streams. An open-loop runner consumes the trace against an explicitly supplied vLLM URL. Keep scheduled arrivals in the denominator, including client drops/errors/timeouts. Evaluate a fixed arrival cohort through completion or an explicit censor deadline; do not require completion inside the arrival window. Report background and target SLO attainment, finite completed-only p95 separately from censored/unconditional bounds, and transport/dispatch lag. Reset or record cache policy and verify server idle before each cell. A fake server verifies plumbing only. Synthetic-length replay is an E1 externality experiment, not an end-to-end answer-quality evaluation.

## Required resources

500-question hidden.npz for full probe evaluation; NVIDIA GPU/vLLM URL plus allowed runtime for real B. Complete CPU baselines, tests, documentation and runnable commands while these are unavailable.
