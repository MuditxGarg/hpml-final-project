# Addendum to the same-state fixed-action rerun

Written 2026-11-21 before the repeated block started.

The first rerun stopped at about 23:16 during seed 20261118, light rate, with 2 of 4 cells complete. Its server process ended at the same time. That partial block is kept on disk and is not used.

The whole seed 20261118 light block is repeated from its first cell in a new output directory, with the same plan, calibration, trace, code commit d1df19b and a new server process pinned to 55,664 KV-cache tokens (log: server-kv-matched-restart-20261121.log). The analysis rule in same-state-fixed-rerun-note.md is unchanged.
