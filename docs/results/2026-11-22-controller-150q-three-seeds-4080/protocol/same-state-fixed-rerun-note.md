# Same-state rerun of the fixed actions, seeds 20261117 and 20261118

Written 2026-11-21 before any cell of this rerun started.

Reason: the policy cells of these two seeds ran on 2026-11-21 at a mean GPU power of about 122 to 127 W (current limit 140 W). The fixed-action cells they would be compared with recorded about 103 to 115 W. Control p95 was 0.1 to 1.2 s lower in the policy run. A cross-state comparison could favour the policies near the KV-cache limit.

Rerun: control, (1,0), (4,0), (1,2048), same plan, calibration, traces, server process and code commit d1df19b. Contended blocks first, then light blocks.

Analysis rule, fixed now:
- Primary comparison: seed 20261116 uses its existing fixed and policy blocks. Seeds 20261117 and 20261118 use the fixed blocks of this rerun and the policy blocks of 2026-11-21.
- The earlier cross-state comparison is reported next to it and is not discarded.
- Thresholds and policies are not changed.
- If control p95 of a rerun block differs from the policy block control by more than 2 s, that block is flagged in the report.
