# Matched Real-Answer Load Comparison Implementation Plan

> Implement inline with superpowers:executing-plans.

**Goal:** Run the accepted low/high-load real-answer diagnostic with matched background traffic.
**Architecture:** Optional scheduler callback reuses tested scheduling; standalone driver handles calibration, fixed traces, paired cells and offline scoring.
**Tech Stack:** Python, existing vLLM HTTP client, transformers, math-verify, Slurm offline cache.
**Spec:** ../specs/2026-11-12-real-answer-load-design.md

## Constraints and review focus
No new GPU until approved. Preserve old synthetic scheduler tests. Match background requests and target seeds; exclude control targets. Scheduled-arrival latency must include dispatch lag and both deep stages. Persist failures before idle checks. Refuse blocks that cannot fit deadline. Do not call finite-burst throughput steady-state capacity.

- [x] Add meaningful tests for callback/control handling, matched deterministic traces, no-answer/late scoring and whole-block time guard; observe failure.
- [x] Implement optional callback and real_answer_load.py; run fake HTTP integration and existing tests.
- [x] Prepare bounded same-node offline batch and 495-background-prompt bundle; verify hashes and resource estimate.
- [x] Review implementation and fix material findings; commit runnable assets.
- [ ] Request120min free cluster allocation, then execute only when approved, download/audit results and confirm release.

Local implementation complete:41tests pass including HTTP integration; batch syntax and git diff checks pass. Independent reviewer found missing paired deltas/queue summaries; regression observed RED then GREEN and full suite passed after fix. Input bundle has5targets,495disjoint backgrounds, hashes and exactcode. GPU not requested/allocated yet; request120min once reviewable assets committed.
