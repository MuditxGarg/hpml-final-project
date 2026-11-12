# B repeated replay implementation plan

> Execute inline using executing-plans; final review separately. Existing user authorization covers B implementation and a second free T4 session up to60minutes, with no purchase.

Goal: extend the existing E1 matched replay to repeatable longer blocks, preserving all scheduled outcomes and explicit limitations.

Spec: existing evaluation-serving spec, B section. Continue synthetic externality measurement; do not substitute it for E2 answer-quality routing.

Approach: extend existing runner instead of building a simulator or online router now. Those alternatives cannot resolve the short-cohort limitation in the measured evidence. Existing feature branch preserves original pilot and A; A review branch remains separate.

- [x] Add deterministic optional cell-order randomization including control to matched_serving.py; defaults retain historical ordering. Verify trace unchanged and control exactly once.
- [x] Add warmup/duration/idle-timeout options to bounded driver and a read-only planning mode showing expected request counts and conservative deadline cost. Preserve historical defaults. Test longer durations affect deadline and trace.
- [x] Make report handle interrupted blocks without a control, preserve order/protocol, and aggregate seed-level paired effects only across complete matched blocks. Do not bootstrap individual correlated requests. With fewer than5 seeds report descriptive variation, no CI.
- [x] Test existing full suite and generate a pre-run plan/bundle from corrected lengths and recovered prompts.
- [ ] Run within newly authorized T4 session; recalibrate on new allocation, record exact launch and runtime cutoff, retain partial evidence, download then disconnect before60minutes.

Prospective pilot configuration: seeds20261113/20261114, loads0.5/0.8, actions(4,0)/(4,2048) plus control, warmup30s and measurement120s. This targets replication of wide shallow vs wide deep externality, not full action optimization. At the previous target rate this is only about3.50 target jobs per block; still a pilot. The full conservative schedule may exceed the session after setup/calibration; skip whole blocks that cannot fit, never shorten a running block or select seeds by outcomes. Alternate load order across seeds. New calibration proxy is not a saturation proof. Freeze SLO before comparison.

Progress: resources authorized; GPU not allocated yet. Preserve final reviewer findings and actual completed scope here.

Ruling before GPU allocation: budget preflight gave4164s excluding setup for180s windows. Use120s windows prospectively; preserve seeds/actions, record skips. 22 tests passed.

Verification:24 full-suite tests passed, originalpilot rendered with newreport successfully. Independent reviewer found no substantive defect; countmatching remains a consistency check, not independent trajectory audit.
Session2 conservative start2026-11-12 22:11:32UTC; driver deadline23:07:32UTC; unassign watchdog23:09:32UTC. FreshT4UUID GPU-6785bd65-1459-86ba-55fb-581941b8ff74. Installation in progress.

Prospective design correction before comparisons: firstfixedseed had0 measuredtargets inoldcalibration preflight. Use --target-count6 conditionalPoissonuniform arrival times per120s; retainseeds/background/warmup. Measuredtargetrate0.05 differsfromcalibrationwarmup rate. All25tests pass; no comparisonoutcomes were observed/selected. OriginalPoissonpilot not pooled.
