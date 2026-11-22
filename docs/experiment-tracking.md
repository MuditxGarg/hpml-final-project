# Experiment tracking

`17_log_to_wandb.py` uploads finished run directories to the team's Weights & Biases project so that every reported number can be traced to a run on a dashboard. It reads what the experiment scripts already wrote; it measures nothing and changes no result. The logic is in `experiment_tracking.py`.

## What is uploaded

| Source directory | Tracked runs | Content |
|---|---|---|
| Sustained calibration (13) | one per pass | one step per rung (rate, latency percentiles, queue, KV-cache usage, backlog growth); idle p95 and frozen rates in the summary |
| Profiling pass (14) | one per cell | per-second running, waiting, KV-cache usage, preemptions, generated tokens, GPU utilisation, memory, power and temperature; the cell summary |
| Matched real answers (15) | one per cell, plus one `summary` run | the same telemetry and the cell's scored counts; the summary run holds the per-action table with intervals and the paired differences |

Each run's config holds the hardware record, git commit, script arguments, seed, background rate and action or policy. Runs are grouped by source directory and typed `calibration`, `profiling`, `matched` or `matched-summary`.

Not uploaded: question text, model outputs, per-request rows, code, the git remote, host name, user name and client system metrics. The raw files stay on the machine that ran the experiment.

## Use

Install the client (`pip install wandb`; the tests do not need it) and log in once per machine with `wandb login` (add `--host` for a dedicated instance). The key is stored in `~/.netrc`; never put it in the repository or in a config file here.

```sh
python 17_log_to_wandb.py --entity TEAM --project PROJECT --tags rtx4080-laptop \
  --calibration /absolute/path/calibration-directory \
  --profile /absolute/path/profiling-directory \
  --matched /absolute/path/matched-directory
```

`--dry-run` lists the runs without contacting the service. Each option can be repeated. Score a matched directory before uploading it, otherwise it has no summary run.

Run ids are derived from the directory name and cell, so uploading a directory again skips what is already there. The service never reuses the id of a deleted run: to replace uploaded runs, delete them and raise `ID_VERSION` in `experiment_tracking.py`.

## Rules

- Tag every upload with its hardware. Runs from different machines are never averaged on the dashboard.
- Upload only complete, logged runs. Dashboard numbers and report numbers come from the same files.

## Tests

`tests/test_experiment_tracking.py` checks the run descriptions against committed results and a small matched directory, and the upload calls against a stand-in client. No test contacts the service.
