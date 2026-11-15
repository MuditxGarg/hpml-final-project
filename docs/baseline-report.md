# Baseline report generator

`16_make_baseline_report.py` builds the tables and plots of the mid-point baseline from logged run directories only. It is step 4 of the mid-point baseline. The logic is in `baseline_report.py`.

## Inputs and outputs

```sh
python 16_make_baseline_report.py \
  --calibration /absolute/path/calibration-seed-A --calibration /absolute/path/calibration-seed-B \
  --profile /absolute/path/profiling-run \
  --matched /absolute/path/scored-matched-run \
  --hardware "GPU, host, vLLM version, model" \
  --output docs/results/<date>-<name>
```

Every input is optional, but at least one is required. The output directory must not exist. It receives:

- `README.md`: what was measured, what was not, the figures, and every table.
- `figures/*.png`: the calibration ladder, per-action background latency and preemptions, per-token decode time against running sequences, the timeline of the worst profiling cell, and correct answers within the target per action.
- `sources.json`: path and SHA-256 of every file that was read.

## Rules it follows

- Every number is formatted from a logged JSON value. Missing values are printed as `-`, never as zero.
- `--hardware` is mandatory and printed at the top, because results from different machines are never pooled.
- The generated README states limitations but draws no conclusions. Interpretation goes into a separate hand-written file in the same directory, so that regenerating the report cannot overwrite it and cannot invent a claim.
- The timeline figure needs the per-second server samples of the profiling run. It is skipped when only the summary is available.
- Plots need `matplotlib` (listed in `requirements-eval.txt`). Without it, or with `--no-plots`, the tables are still written and the README says that no plots were made.

## Figures

One measure per axis; two measures of different scale are drawn as separate panels. Each action keeps one colour in every figure, with the background-only control in grey. Every figure has a caption, and the same values are in the tables.

## Tests

`tests/test_baseline_report.py` builds a report from the committed calibration and profiling summaries under `docs/results/` plus a small hand-made matched summary, and checks that table cells equal the logged values.
