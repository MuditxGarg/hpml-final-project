# Reports in this folder

All files here come from runs on the RTX 4080 Laptop GPU (Qwen3-1.7B, MATH-500, 500 questions). Do not pool them with results from other hardware.

| File | Written by | File date | Status |
|---|---|---|---|
| `overnight_report.md` | `05_make_report.py` | committed earlier | Current |
| `nothink_cap_report.md` | `01c_compare_nothink_cap.py` | committed earlier | Current |
| `accuracy_v2.md` | `01d_merge_nothink_cap.py` | 2026-11-06 | Current. Accuracy table with the `B=0` cap at 2048 tokens. |
| `cost_model_report.md`, `cost_model_curve.png` | `06_fit_cost_model.py` | 2026-11-06 | Historical. See note 1. |
| `probe_report.md`, `routing_frontier.png` | `legacy/07_train_probe.py` | 2026-11-06 | Superseded. See note 2. |
| `route_full_report.md`, `route_full.png` | `legacy/08_route_full_actions.py` | 2026-11-06 | Superseded. See note 2. |

The 2026-11-06 files were recovered on 2026-11-15 from the original run folder on Mudit's laptop and committed unchanged. They were produced before this code was under git, so no commit hash is recorded for them. The scripts that wrote them are byte-identical to the versions now in this repo (`06_fit_cost_model.py`, `legacy/07_train_probe.py`, `legacy/08_route_full_actions.py`).

Their label source is `../rollouts_v2/` and `../lengths_v2.json`, written by `01d_merge_nothink_cap.py`. Their probe features are the original 4080 BF16 `probe/hidden.npz`, which is not in git (`*.npz` is ignored).

## Notes

1. **Cost model.** The leave-one-action-out check in `docs/results/2026-11-12/` is the one to cite. It shows the token-only curve missing the saturation knee for `(4,2048)` at 0.8 load.
2. **Probe and routing.** These two reports use a single 100/400 dev/test split and the original evaluation procedure. That procedure was replaced by the nested evaluation in `09_evaluate.py` (protocol in `docs/evaluation.md`), which does all selection on training folds only. Some intervals in `route_full_report.md` exclude zero in the probe's favour; the nested evaluation did not confirm a gain over fixed actions or a fixed mixture. Do not quote these two reports as evidence of a routing gain.
