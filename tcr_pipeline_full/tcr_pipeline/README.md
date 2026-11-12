> Updated evaluation: see [docs/evaluation.md](../../docs/evaluation.md). 07/08 now use nested evaluation; historical scripts are in legacy/. Matched E1 replay is documented in [docs/matched-serving.md](../../docs/matched-serving.md). The original measurement-stage description below is retained for provenance.

# tcr_pipeline: measurement stage for load-aware test-time compute routing

Measures two things for Qwen3-1.7B on MATH-500, then writes one report:

1. **Accuracy of every action** `a = (N, B)`: N parallel samples with majority vote, and a thinking budget of B tokens (B=0 is non-thinking mode, capped at 1024 tokens).
2. **How much each action slows other requests** on a live vLLM server at different load levels (the externality).

It does not train the router yet. Its outputs are the inputs for the cost model, probe, simulator and allocator.

## Requirements

- NVIDIA GPU with at least 12 GB free (tested: RTX 4080 Laptop, WSL2). The script picks the model by GPU memory: under 22 GB uses Qwen3-1.7B.
- Python 3.12, vLLM 0.32.0, torch 2.13 (cu130), transformers, scikit-learn, math-verify, requests.
- Internet on the first run (MATH-500 and model weights are downloaded and cached).
- About 15 GB free disk.

## Run

```bash
python run_pipeline.py smoke        # 10-25 min plumbing check; must print SMOKE OK
python run_pipeline.py serve --rollouts <folder with pilot chunk_*.json>   # 1-1.5 h: live-server sweep only, reuses pilot labels
python run_pipeline.py main --hours 48                                     # full run
```

Measured on the RTX 4080 laptop: labeling 159 min, probe features 0.5 min, live-server sweep 262 min, 421 min in total.

- Every stage is resumable. After a crash, run the same command and finished work is skipped.
- `--hours` only limits the final sweep (no new cells start after the deadline). It does not limit labeling.
- Use a fresh folder for each new run and do not copy old results in: copied chunks would be mistaken for finished work.
- Run inside `tmux`, with the laptop plugged in and Windows sleep off.

## Stages

| File                           | Task                                                                                                                |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| `run_pipeline.py`            | Runs the stages in order. Modes: `smoke`, `serve`, `main`. Writes `tcr_config.json` (settings for the run). |
| `00_preflight.py`            | Checks disk, GPU memory, vLLM, dataset, tokenizer and a free port 8001 before any long job.                         |
| `01_label_rollouts.py`       | 16 traces per question, cut at every budget, forced final answer, correctness labels.                               |
| `02_probe_features.py`       | Last-token prefill hidden states per prompt (inputs for the success probe).                                         |
| `03_decide_values.py`        | Derives the N grid, dominated budgets, dev/test split and request-length distributions from the labels.             |
| `04_serve_measure.py`        | Starts a vLLM server and measures idle latency, capacity, and the externality sweep.                                |
| `05_make_report.py`          | Writes `report/overnight_report.md`.                                                                              |
| `optional_pilot_analysis.py` | Probe AUROC and router-versus-fixed analysis. Run by hand after stages 01 and 02.                                   |
| `tcr_common.py`              | Shared helpers (data loading, answer checking, config).                                                             |
| `test_fake_vllm_server.py`   | Fake server for testing the load code without a GPU.                                                                |

## How the experiment values are chosen (nothing is picked by hand)

- **N grid:** always {1, 2, 4}, then doubles while 2N samples still gain at least 1 accuracy point (mean over budgets of 1024 and above). In the full run this stopped at 4, so the sweep measured N up to 4 (labels cover N up to 16).
- **Latency targets:** 2x and 4x the idle p95 of a reference request (full run: 20.7 s and 41.4 s).
- **Capacity:** closed-loop ramp of clients; maximum request rate (full run: 2.78 req/s).
- **Load levels:** fractions of capacity: 0, 0.5, 0.8, 1.0, 1.3.
- **Dev/test split:** 20% dev, 80% test, by hash of the question id.
- **Request sizes in the load test:** the real token counts per budget, from the labels.

## Externality sweep

For each load level, background Poisson requests run at that rate. A control cell has no extra stream. Each other cell adds a stream of action `(N, B)` requests at 5% of capacity (`target_rate_frac`). The effect of an action is its background p95 minus the control p95. Requests that do not finish are kept as censored and counted as SLO misses (no silent cap). N samples are sent as N parallel requests sharing a prompt; latency is the slowest one.

## Outputs (`tcr_results/main/`)

| Path                                                | Content                                                            |
| --------------------------------------------------- | ------------------------------------------------------------------ |
| `report/overnight_report.md`                      | Read this first: chosen values, accuracy table, externality table. |
| `decisions.json`                                  | Frozen values and the accuracy table.                              |
| `lengths.json`                                    | Empirical token counts per budget.                                 |
| `rollouts/chunk_*.json`                           | Per-question, per-trace, per-budget correctness.                   |
| `probe/hidden.npz`                                | Prompt hidden states (large, not in git).                          |
| `load/calibration.json`, `load/ext_cells.jsonl` | Idle latency and capacity; one row per sweep cell.                 |
| `logs/`                                           | One log per stage.                                                 |

## Known limits

- B=0 is capped at 1024 tokens. 22% of B=0 samples hit the cap and have no answer, so B=0 accuracy (0.658 at N=1) is a lower bound (0.84 among those that finished).
- Two repetitions per cell. Differences under about 3 s are noise.
- One model, one GPU, synthetic fixed-length requests.

## Troubleshootin

# TCR pipeline (files named by task, run order = number)

| file                       | task                                                                                     |
| -------------------------- | ---------------------------------------------------------------------------------------- |
| 00_preflight.py            | checks GPU memory, disk, vllm, dataset, tokenizer, free port before any long job         |
| run_pipeline.py            | runs everything below in order (smoke / serve / main)                                    |
| 01_label_rollouts.py       | 16 traces per question, cut at every thinking budget, forced answers, correctness labels |
| 02_probe_features.py       | last-token prefill hidden states per prompt (inputs of the success probe)                |
| 03_decide_values.py        | derives N grid, B flags, dev/test split, request-length distributions from the labels    |
| 04_serve_measure.py        | vLLM server: idle latency, capacity, load regimes, externality sweep (open-loop)         |
| 05_make_report.py          | writes report/overnight_report.md                                                        |
| optional_pilot_analysis.py | probe AUROC + router-vs-fixed analysis (run by hand after 01+02)                         |
| tcr_common.py              | shared helpers (data, answer checking)                                                   |
| test_fake_vllm_server.py   | fake server for testing the load code without a GPU                                      |
