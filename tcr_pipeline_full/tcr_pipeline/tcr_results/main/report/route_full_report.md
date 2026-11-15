# Routing over the full (N,B) grid under a p95 SLO (labels: rollouts_v2, 400 test questions)

25 actions. rho = arrival rate as a fraction of capacity (reference requests of 1005 tokens). Predicted background p95 comes from the cost model (06); offered loads above 1.64 (largest measured) are clipped, so those cells are optimistic.

## SLO: p95 <= 20.7 s (2x idle p95)

| rho | best fixed | fixed mix | level-bin | probe | oracle | probe - best baseline (95% CI) |
|---|---|---|---|---|---|---|
| 0.4 | 0.774 (1706 tok) | 0.774 (1706 tok) | 0.769 (1689 tok) | 0.772 (1543 tok) | 0.848 (1508 tok) | -0.002 [-0.017, +0.013] |
| 0.6 | 0.723 (722 tok) | 0.744 (1115 tok) | 0.748 (935 tok) | 0.756 (1085 tok) | 0.827 (1016 tok) | +0.008 [-0.007, +0.021] |
| 0.8 | 0.723 (722 tok) | 0.728 (820 tok) | 0.726 (765 tok) | 0.730 (853 tok) | 0.813 (807 tok) | +0.002 [-0.008, +0.013] |
| 1.0 | 0.439 (522 tok) | 0.681 (692 tok) | 0.600 (529 tok) | 0.712 (692 tok) | 0.793 (667 tok) | +0.031 [+0.020, +0.041] |
| 1.2 | 0.439 (522 tok) | 0.555 (562 tok) | 0.600 (529 tok) | 0.650 (516 tok) | 0.755 (560 tok) | +0.050 [+0.016, +0.082] |

## SLO: p95 <= 41.4 s (4x idle p95)

| rho | best fixed | fixed mix | level-bin | probe | oracle | probe - best baseline (95% CI) |
|---|---|---|---|---|---|---|
| 0.4 | 0.774 (1706 tok) | 0.780 (2723 tok) | 0.775 (2077 tok) | 0.791 (2715 tok) | 0.880 (2744 tok) | +0.011 [-0.003, +0.026] |
| 0.6 | 0.774 (1706 tok) | 0.774 (1883 tok) | 0.773 (1825 tok) | 0.780 (1892 tok) | 0.858 (1746 tok) | +0.006 [-0.009, +0.021] |
| 0.8 | 0.723 (722 tok) | 0.759 (1411 tok) | 0.761 (1372 tok) | 0.765 (1414 tok) | 0.842 (1378 tok) | +0.004 [-0.011, +0.019] |
| 1.0 | 0.723 (722 tok) | 0.744 (1115 tok) | 0.748 (935 tok) | 0.756 (1085 tok) | 0.827 (1016 tok) | +0.008 [-0.006, +0.022] |
| 1.2 | 0.723 (722 tok) | 0.733 (919 tok) | 0.748 (935 tok) | 0.736 (909 tok) | 0.822 (925 tok) | -0.012 [-0.027, +0.002] |

Unconstrained best fixed action: (16, 2048) acc 0.810 at 27297 tokens.
Feasible = predicted p95 under SLO; if nothing is feasible the cheapest point is shown (accuracy collapses).