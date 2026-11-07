# Non-thinking mode (B=0) vs token cap
500 questions x 16 samples, same questions as the main run.

| cap (tokens) | answered | acc among answered | acc N=1 | acc N=4 | acc N=8 | acc N=16 | mean tokens/sample |
|---|---|---|---|---|---|---|---|
| 512 | 0.492 | 0.911 | 0.448 | 0.512 | 0.534 | 0.549 | 419 |
| 1024 | 0.775 | 0.845 | 0.655 | 0.724 | 0.750 | 0.759 | 593 |
| 2048 | 0.926 | 0.776 | 0.719 | 0.772 | 0.792 | 0.799 | 727 |
| 4096 | 0.985 | 0.741 | 0.729 | 0.773 | 0.798 | 0.804 | 800 |

Generation length: median 522, p90 1729, p99 4096 tokens.
Compare the 2048-thinking budget: acc N=1 = 0.766 (main report). If the cap at 1024 matches the main run figures (0.658 at N=1), the main B=0 results are reproduced.