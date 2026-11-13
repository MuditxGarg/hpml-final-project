# Real Answer Completion Validation Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prepare and validate a real-answer HTTP experiment before requesting/using GPU resources.
**Architecture:** New separate runner imports the existing HTTP/idle utilities; preparation joins saved prompts and labels, generation records raw outputs, scoring reuses existing math helpers without changing replay behavior.
**Tech Stack:** Python standard library, transformers tokenizer at execution, math-verify for offline grading.
**Spec:** ../specs/2026-11-12-real-answer-validation-design.md

## Constraints and review focus
Keep the 87s SLO distinct from the 180s observation deadline. Preserve failed samples and entire-question paired completion. Record exact token continuation and prefix/suffix. No reuse of prior expired GPU authorization. Check timeouts across both stages, malformed/missing boxes, tie-breaking, and late but correct answers. Keep gold away from generation inputs.

## Tasks
- [ ] Add tests for EOS-aware shallow requests, natural/forced deep continuation, common deadline, failed-sample voting, late answers and deterministic difficulty sampling; observe failure.
- [ ] Implement real_answer_validation.py with prepare/run/score commands, immutable plan, question block guard, immediate JSONL persistence, stage/token evidence and offline report.
- [ ] Run targeted tests and complete suite; build five-question input and portable bundle; inspect plan IDs and request bounds.
- [ ] Review implementation, record evidence and resource handoff. Execute live pilot only after new GPU permission; otherwise leave runnable assets ready.

## Execution ledger
Scope: accepted real-answer validation step, implemented inline. Existing feature checkout is used; earlier naming edits are preserved. New GPU request pending.
