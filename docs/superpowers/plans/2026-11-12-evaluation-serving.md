# Evaluation and matched serving implementation plan

> **For agentic workers:** Use executing-plans for A and a separate parallel implementer for B; review the integrated change before delivery.

**Goal:** Deliver reproducible evaluation without test-set selection and a matched open-loop serving experiment.

**Architecture:** New importable evaluation/data modules and CLI beside the numbered pipeline; independent replay module/CLI; shared outputs are JSON/CSV with explicit provenance. Preserve historical scripts as archived references and route 07/08 to the corrected CLI.

**Tech Stack:** Python, NumPy, scikit-learn, unittest; standard-library HTTP for replay.

**Spec:** docs/superpowers/specs/2026-11-12-evaluation-serving.md

## Global constraints

- Original measured data stays untouched. No invented probe or GPU results.
- Select policies on training/validation only; outer labels used only for reporting.
- All samples from a question stay in one fold.
- No-feasible/out-of-domain results must be explicit.

## Review focus

- Perturbing held-out labels must not change fitted policy choices.
- A known violating point must not be reported feasible.
- B=0 cap and load-length provenance must remain visible.
- Scheduled drops/timeouts remain misses; completions after the arrival window remain valid.
- Same trace/seed yields identical offered work across actions.

## Task 1: A evaluation

- [x] Write failing tests for held-out invariance, calibrated-domain handling, no feasible action, fixed-cost lambda freezing and paired-bootstrap pairing.
- [x] Implement `evaluation_data.py`, `evaluation.py`, `09_evaluate.py` and CPU requirements; test genuine labels with level-bin/fixed baselines when hidden states are absent.
- [x] Make 07/08 legacy behavior explicit and direct default execution to corrected evaluation.
- [x] Run unit tests and actual 500-question baseline evaluation; save report and fold/decision artifacts.

## Task 2: B matched replay (independent files)

- [x] Write failing tests for matched traces, cohort metrics and HTTP execution using a local fake.
- [x] Implement `matched_serving.py`, `10_matched_serving.py`, dedicated tests and usage documentation.
- [x] Run fake-server end-to-end test and prepare real run instructions; execute real experiments only if GPU access is supplied.

## Task 3: integration

- [x] Run full applicable test discovery, CLI smoke checks and fresh code review.
- [x] Document actual results, remaining resources and commands. Commit the reviewable local work without pushing or merging shared branches.

## Progress / rulings

- User approved A and B execution; no further design approval needed for this implementation.
- Fresh clone is dedicated to this task; use a feature branch in it rather than another checkout.
- Primary executor owns A; parallel agent owns B only. A predicts existing extra-stream externality; B measures the same workload semantics.

### Verified checkpoint

- Task1 complete: nested full 500-question probe run, cap2048; corrected fold-feasibility report.
- Task2 complete as bounded E1 pilot: two matched real-T4 blocks, eight cells, one seed. Full features recovered. Runtime released2026-11-12 21:28:45UTC, verified Reconnect T4 UI; about49minutes, within60minute authorization. No paid compute. Raw archives and notebook downloaded.
- Verification: 20 tests passed; source compile/diff checks passed. Fresh reviewer found fold failures hidden by pooling, fixed with regression test; B core no blocker.
- Ruling: original probe template and rollout template differ by appended think token; preserve labels, document pre-existing difference.
- Ruling: no strict oracle upper-bound claim; finite-grid empirical hindsight reference separately labeled.
- GPU pilot is E1 synthetic externality only; E2 actual answer quality/online p95 controller not implemented or claimed.

- Final GPU analysis: all background cohorts meet92s heuristic SLO; high-load(4,2048) single target93.72s misses. One target per action/load,9/13 background jobs, no measured waiting queue. Pilot only.
- Corrective run preserves initial calibration, raises timeout/drain60→97s beyond92s SLO, reduces action matrix prospectively, and records omitted(1,0). Initial interrupted run retained.
- Code review of final driver/report found missing corrective metadata; report now preserves protocol and partial-block status.
- Local commits use user-approved Peiyan Sun <ps3558@columbia.edu> per command; no global identity change, remote push or team messages.
