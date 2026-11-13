# Matched Real-Answer Load Comparison: failed startup attempt

Job13889319 ran on ins091 from November12,2026 at22:29:11 to approximately22:32:41 America/New_York. Slurm reports FAILED, exit1, elapsed3m30s. The GPU was released; the user job queue is empty. No replacement allocation was submitted.

The model server and unscored eight-token warmup succeeded. The experiment then failed before calibration or comparison requests: the CLI argparse namespace includes the dispatch function `fn`, and the new driver attempted to JSON-serialize the entire namespace into its manifest. The resulting TypeError is a runner bug, not a model or cluster failure. The earlier tests covered HTTP callbacks and scoring, but missed the actual CLI-to-manifest path.

No low/high-load comparison results exist from this attempt. There are zero paired blocks and no target accuracy, background SLO or routing conclusions to report. Saved inputs and setup/server logs are in setup-evidence.zip; the stack trace and final Slurm accounting are in failure-and-accounting.txt.

The local fix excludes only the argparse dispatch callback from persisted configuration. A regression invokes the real CLI with a stubbed tokenizer/server/calibration, verifies JSON manifest persistence and the whole-block deadline guard. It reproduced the same failure before the fix. After the fix, all42 tests pass, including local HTTP integration. Generation parameters, cohort, rates and experimental protocol are unchanged. The fix has not yet been rerun on GPU.

The authorized single attempt consumed3m30s of the120-minute ceiling. Per the no-replacement-job restriction, another submission requires user authorization; a retry within the original cumulative ceiling would have at most116m30s remaining. Monitoring ends because this job is terminal.
