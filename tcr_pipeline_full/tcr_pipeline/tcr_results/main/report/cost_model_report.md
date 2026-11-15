# Cost model: background p95 vs offered load
40 cells (load x action, averaged over repetitions). Reference request = 1005 tokens; extra stream = 0.05 x capacity in requests/s.

| load definition | in-sample R^2 | Spearman | leave-one-action-out MAE (s) | worst action MAE (s) |
|---|---|---|---|---|
| tokens (N x mean tokens(B)) | 0.974 | 0.973 | 4.72 | 14.2 (4, 2048) |
| samples only (N) | 0.796 | 0.931 | 9.37 | 27.1 (4, 2048) |
| requests only | 0.749 | 0.904 | 9.02 | 28.9 (4, 2048) |

## Biggest residuals under the token model (observed minus fitted p95, seconds)
- load 0.8, action (4,2048): observed 61.8, fitted 50.2, offered load 1.14
- load 1.0, action (4,1024): observed 39.1, fitted 50.2, offered load 1.20
- load 1.3, action (1,2048): observed 71.1, fitted 61.5, offered load 1.38
- load 1.3, action (4,0): observed 54.0, fitted 61.5, offered load 1.42
- load 1.3, action (4,1024): observed 76.9, fitted 83.9, offered load 1.50
- load 1.3, action (2,2048): observed 90.9, fitted 83.9, offered load 1.47
- load 0.5, action (4,2048): observed 25.5, fitted 23.0, offered load 0.84
- load 1.3, action (2,1024): observed 59.5, fitted 61.5, offered load 1.40

## Predicted background p95 (s) if ALL of the extra stream used action (N,B), token model; B=0 uses the new lengths if available
Values above the largest observed offered load are clipped (marked *).
| action | load 0.5 | load 0.8 | load 1.0 |
|---|---|---|---|
| (1,0) | 16.1 | 22.5 | 28.3 |
| (2,0) | 17.1 | 23.5 | 28.6 |
| (4,0) | 18.9 | 25.8 | 50.2 |
| (8,0) | 21.2 | 28.7 | 57.1 |
| (16,0) | 28.6 | 61.4 | 98.5 |
| (1,2048) | 17.9 | 23.9 | 28.6 |
| (2,2048) | 19.8 | 27.5 | 50.2 |
| (4,2048) | 23.0 | 50.2 | 59.5 |

Noise is about +-3 s (2 repetitions). Control cells (no extra stream) are included as N=0.