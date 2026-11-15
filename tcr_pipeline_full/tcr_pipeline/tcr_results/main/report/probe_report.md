# Success probe (labels: rollouts_v2; dev 100 / test 400 questions)

AUROC of "this trace is correct", score = predicted success of the question.

| layer index | dev AUROC B=0 | dev AUROC B=2048 | test AUROC B=0 | test AUROC B=2048 |
|---|---|---|---|---|
| 0 (layer 0) | 0.436 | 0.453 | 0.452 | 0.431 |
| 1 (layer 4) | 0.803 | 0.821 | 0.759 | 0.787 |
| 2 (layer 8) | 0.837 | 0.821 | 0.773 | 0.802 |
| 3 (layer 12) | 0.828 | 0.808 | 0.810 | 0.829 |
| 4 (layer 16) | 0.828 | 0.840 | 0.836 | 0.860 |
| 5 (layer 20) | 0.807 | 0.831 | 0.835 | 0.856 |
| 6 (layer 24) | 0.830 | 0.852 | 0.830 | 0.859 |
| 7 (layer 28) | 0.839 | 0.854 | 0.828 | 0.858 |

Layer chosen on dev: index 7 (layer 28).

| score | test AUROC B=0 | test AUROC B=2048 |
|---|---|---|
| probe (hidden state) | 0.828 | 0.858 |
| prompt length (longer = harder) | 0.725 | 0.766 |
| MATH level (higher = harder) | 0.733 | 0.722 |
| ridge on [length, level] | 0.772 | 0.795 |

## Routing frontier at N=1 on the 400 test questions
All non-thinking: acc 0.723 at 722 tokens. All thinking(2048): acc 0.774 at 1706 tokens.
Accuracy at equal mean tokens (random mixing of the two fixed actions = straight line):

| mean tokens | random mix | probe gain (p2048 - p0) | probe, B=2048 success only (hard first) | MATH level (hardest first) | oracle (true gain) |
|---|---|---|---|---|---|
| 968 | 0.736 | 0.747 | 0.738 | 0.748 | 0.808 |
| 1214 | 0.749 | 0.764 | 0.755 | 0.757 | 0.816 |
| 1460 | 0.761 | 0.767 | 0.769 | 0.772 | 0.816 |