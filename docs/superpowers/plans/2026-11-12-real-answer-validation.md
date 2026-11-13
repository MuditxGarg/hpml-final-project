# Real Answer Completion Validation Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prepare and validate a real-answer HTTP experiment before requesting/using GPU resources.
**Architecture:** New separate runner imports the existing HTTP/idle utilities; preparation joins saved prompts and labels, generation records raw outputs, scoring reuses existing math helpers without changing replay behavior.
**Tech Stack:** Python standard library, transformers tokenizer at execution, math-verify for offline grading.
**Spec:** ../specs/2026-11-12-real-answer-validation-design.md

## Constraints and review focus
Keep the 87s SLO distinct from the 180s observation deadline. Preserve failed samples and entire-question paired completion. Record exact token continuation and prefix/suffix. No reuse of prior expired GPU authorization. Check timeouts across both stages, malformed/missing boxes, tie-breaking, and late but correct answers. Keep gold away from generation inputs.

## Tasks
- [x] Add tests for EOS-aware shallow requests, natural/forced deep continuation, common deadline, failed-sample voting, late answers and deterministic difficulty sampling; observe failure.
- [x] Implement real_answer_validation.py with prepare/run/score commands, immutable plan, question block guard, immediate JSONL persistence, stage/token evidence and offline report.
- [x] Run targeted tests and complete suite; build five-question input and portable bundle; inspect plan IDs and request bounds.
- [ ] Review implementation, record evidence and resource handoff. Execute live pilot only after new GPU permission; otherwise leave runnable assets ready.

## Execution ledger
Scope: accepted real-answer validation step, implemented inline. Existing feature checkout is used; earlier naming edits are preserved. New GPU request pending.

34 tests passed including HTTP runner integration; fresh reviewer findings on hard deadline, no-answer delivery, and failed-stage evidence fixed and regression-tested.
User selected Insomnia; previous project chat and HW2 scripts supplied connection/free partition. VPN + user SSH authentication completed. Job13886133 cancelled after1m46s because PyPI default selected CUDA13 on driver575.57.08. Corrected officialvLLM0.32.0+cu129 wheel job13886164 uses55-minute limit; no automatic requeue. Total upper bound56m46s. Pending live results.

User authorized120min cumulativeGPU. Job13886164 cancelled29m10s after preserving complete vLLM/cuDNN/CUDA wheels; serial pip network throughput~0.86MB/s. Job13887170 usesuv parallel installer and savedwheelhouse,80min limit, same15action plan/code, no requeue. Parent allocation upper bound1m46s+29m10s+80m=110m56s (termination grace also below120m).

Job13887170 failed2m55s on pypi.nvidia.com cusolver download; no answers. Confirmed exactsameversion official files.pythonhosted.org URL responds200 fromlogin. Added13 hash-pinned directsource constraints; no dependencyversion changes. Job13887790 submitted80min, cumulativeparent upperbound113m51s.

Read-only download timing showed Mac~20MB/s vsnode~1MB/s. Job13887790 cancelled16m08s; cumulativeparent49m59s. Exact198package lock compiledonnode, allLinuxwheels downloadedlocally (2platform/indexissues resolved:glibc2.34 llguidance,torchvisionofficialindex lackswheelhash). Model9files~4.08GB allavailableLFShashesverified. Transferinprogress withoutGPU. Offlinebatch65min, totalparentbound114m59s. Code35tests passed; localempty-runreportfix notchangedactivegeneration.

Offlinejob13888699 installed200packages and passeduvcheck in<1min, then failed41s because PyPI torchcodec0.17.0 links libnvrtc.so.13. Verified pinnedvLLM catches unavailableFFmpeg RuntimeError but notthisOSError. Use officialtorchcodec0.17.0+cpu forunusedaudio/video paths (hashd859426bd2fed833ed0826c6d3fefa7a261178cd550420bd536e9f073e7eea56); textmodel remainsFP16CUDA12.9. Cumulativeparent50m40s, next65min bound115m40s.
