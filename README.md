
# Load-aware test-time compute routing (Team 24, COMS 6998 HPML)

When an LLM server is busy, a request that "thinks more" (longer reasoning, or more parallel samples) slows down everyone else, not just itself. This project asks whether choosing the amount of test-time compute per question, and per server load, beats any fixed strategy on correct answers per GPU-second at a p95 latency target.

**Status:** labels, cap-check measurements, load sweep and exploratory routing scripts are available. A corrected nested evaluation and matched-replay harness are now implemented; live end-to-end routing gains remain unproven.

The latest measured results on the RTX 4080 are the [three-seed controller run](docs/results/2026-11-22-controller-150q-three-seeds-4080/INTERPRETATION.md), the [assembled three-seed 150-question run](docs/results/2026-11-16-representative-cohort-assembled-4080/INTERPRETATION.md) and the [50-question baseline](docs/results/2026-11-15-baseline-4080/INTERPRETATION.md); every logged run is also in Weights & Biases at [forge.coreweave.com/wandb/hpml-final-project-team24/latte](https://forge.coreweave.com/wandb/hpml-final-project-team24/latte) (public, tag `rtx4080-laptop`). Start with [independent evaluation](docs/evaluation.md) and [matched serving](docs/matched-serving.md). Per-machine light and contended background rates come from the [sustained calibration](docs/sustained-calibration.md). Historical numerical results below are exploratory. The latest original scripts had test-set operating-point selection; the corrected 07/08 entry points invoke 09_evaluate.py, with historical code archived in legacy/.

## Repository layout

| Folder            | What it is                                                                               |
| ----------------- | ---------------------------------------------------------------------------------------- |
| `tcr_pipeline_full/tcr_pipeline/` | Current pipeline: labeling, probe features, load measurements, report. Start here.       |
| `Pilot_Test/`   | First pilot (LATTE, 150 questions). Superseded by `tcr_pipeline_full/tcr_pipeline/`, kept for reference. |

## Setup in one line

An action is `a = (N, B)`: N parallel samples (majority vote) and a thinking budget of B tokens (B=0 is non-thinking mode, capped at 1024 tokens).

- Model: Qwen3-1.7B. Data: MATH-500. Server: vLLM 0.32.0.
- Tested on an RTX 4080 Laptop GPU (12 GB) under WSL2.

## Main results so far (`tcr_pipeline_full/tcr_pipeline/tcr_results/main/report/overnight_report.md`)

Accuracy of majority vote over N samples (500 questions, 16 traces each):

| N \ B | 0     | 256   | 512   | 1024  | 2048  |
| ----- | ----- | ----- | ----- | ----- | ----- |
| 1     | 0.658 | 0.233 | 0.427 | 0.629 | 0.766 |
| 4     | 0.723 | 0.238 | 0.444 | 0.653 | 0.791 |
| 16    | 0.769 | 0.242 | 0.453 | 0.670 | 0.808 |

Increase in background p95 latency caused by an extra stream of requests using action (N, B), against a control with no extra stream:

| Load (fraction of capacity 2.78 req/s) | (1,0)  | (4,0)  | (1,2048) | (4,2048) |
| -------------------------------------- | ------ | ------ | -------- | -------- |
| 0.5                                    | -0.4 s | +1.5 s | +2.9 s   | +9.3 s   |
| 0.8                                    | +0.8 s | +3.1 s | +4.3 s   | +41 s    |
| 1.0                                    | +1.5 s | +2.9 s | +1.7 s   | +33 s    |

Takeaways:

- Depth acts like a switch: at N=1, thinking to 256 or 1024 tokens is worse than not thinking.
- Width in non-thinking mode is cheap: B=0 with N=16 reaches 0.769, about the same as one 2048-token trace.
- Under load the affordable set of actions shrinks: (4,2048) is harmless at 0.5 load but breaks the 41 s target at 0.8.

## Caveats

- In B=0, 22% of samples hit the 1024-token cap and have no answer (accuracy is 0.84 among those that finished), so B=0 accuracy is a lower bound.
- Two repetitions per cell. Differences under about 3 s are noise.
- One model, one GPU, synthetic fixed-length requests. The sweep measured N only up to 4.

## Reproduce

```bash
cd tcr_pipeline_full/tcr_pipeline
python run_pipeline.py smoke        # 10-25 min, must print SMOKE OK
python run_pipeline.py main --hours 48
```

Details and the file list are in `tcr_pipeline_full/tcr_pipeline/README.md`.

## Next

1. Re-label B=0 with a 2048-token cap.
2. Fit a cost model (does p95 collapse onto one curve of total token load?).
3. Train the success probe (cross-validated).
4. Simulator and allocator with a p95 controller.
5. Evaluate on the 400 held-out questions against fixed actions, a fixed-cost router, difficulty bins and an oracle, with confidence intervals; spot-check on the real server.

## Latest experiment reports

Read these in order; each report includes raw evidence, configuration and limitations:

1. [Matched Serving Interference Study](docs/results/2026-11-12-serving-interference/README.md): forced-length T4 replay shows load-sensitive background interference, but deep targets timed out; it does not measure answer quality.
2. [Real Answer Completion Validation](docs/results/2026-11-12-real-answer-validation/README.md): five-question idle A6000 diagnostic validates EOS-aware shallow sampling and two-stage deep answers.
3. [Matched Real-Answer Load Comparison](docs/results/2026-11-12-real-answer-load/README.md): four complete paired blocks, 16 cells and 60 target decisions. Higher arrival rates increased latency, but no queueing was observed; background delivery misses came from output caps. Load-aware routing gains remain unproven.
4. [Mid-point baseline, RTX 4080](docs/results/2026-11-15-baseline-4080/INTERPRETATION.md): 50 questions, 3 seeds, real queueing at calibrated light and contended rates. `(1,0)` is best per GPU-second at both rates; the cheaper of width and depth flips with load.
5. [Load ladder](docs/results/2026-11-15-load-ladder-4080/INTERPRETATION.md): the same cohort at four background rates, locating where depth stops being affordable.
6. [Representative 150-question cohort, assembled three seeds](docs/results/2026-11-16-representative-cohort-assembled-4080/INTERPRETATION.md): extra compute is worth 6 to 8 accuracy points; by correct on-time answers the best fixed action changes with load (intervals exclude zero); per GPU-second `(1,0)` still wins. The [original run](docs/results/2026-11-16-representative-cohort-4080/INTERPRETATION.md) is kept as the record of the GPU power problem.
7. [Load-adaptive controller, three seeds](docs/results/2026-11-22-controller-150q-three-seeds-4080/INTERPRETATION.md): a frozen controller reading KV-cache usage stays within noise of the best fixed action at both rates and beats a load-blind mixture at the contended rate. It does not beat the best fixed action, and `(1,0)` still wins per GPU-second.

Each W&B run holds metrics and settings only; questions, model outputs and per-job records stay in the committed `summary.json` and `scored.jsonl` files of each folder.

Reproduction: [matched replay](docs/matched-serving.md), [idle real answers](docs/real-answer-validation.md), and [real-answer load comparison](docs/real-answer-load.md). The reports retain the failed setup attempt and final allocation accounting. GPU experiments require an explicitly provisioned server; the local tests do not allocate GPUs.
