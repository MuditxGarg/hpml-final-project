
# Load-aware test-time compute routing (Team 24, COMS 6998 HPML)

When an LLM server is busy, a request that "thinks more" (longer reasoning, or more parallel samples) slows down everyone else, not just itself. This project asks whether choosing the amount of test-time compute per question, and per server load, beats any fixed strategy on correct answers per GPU-second at a p95 latency target.

**Status:** measurement stage done (labels and load sweep). The router, cost model and simulator are next. We have not yet shown that routing wins.

## Repository layout

| Folder            | What it is                                                                               |
| ----------------- | ---------------------------------------------------------------------------------------- |
| `tcr_pipeline/` | Current pipeline: labeling, probe features, load measurements, report. Start here.       |
| `Pilot_Test/`   | First pilot (LATTE, 150 questions). Superseded by `tcr_pipeline/`, kept for reference. |

## Setup in one line

An action is `a = (N, B)`: N parallel samples (majority vote) and a thinking budget of B tokens (B=0 is non-thinking mode, capped at 1024 tokens).

- Model: Qwen3-1.7B. Data: MATH-500. Server: vLLM 0.32.0.
- Tested on an RTX 4080 Laptop GPU (12 GB) under WSL2.

## Main results so far (`tcr_pipeline/tcr_results/main/report/overnight_report.md`)

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
cd tcr_pipeline
python run_pipeline.py smoke        # 10-25 min, must print SMOKE OK
python run_pipeline.py main --hours 48
```

Details and the file list are in `tcr_pipeline/README.md`.

## Next

1. Re-label B=0 with a 2048-token cap.
2. Fit a cost model (does p95 collapse onto one curve of total token load?).
3. Train the success probe (cross-validated).
4. Simulator and allocator with a p95 controller.
5. Evaluate on the 400 held-out questions against fixed actions, a fixed-cost router, difficulty bins and an oracle, with confidence intervals; spot-check on the real server.
