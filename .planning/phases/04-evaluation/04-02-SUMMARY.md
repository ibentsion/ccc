---
phase: 04-evaluation
plan: 02
subsystem: training
tags: [clearml, eval, curriculum, exact-match, edit-similarity, run_eval]

# Dependency graph
requires:
  - phase: 04-evaluation/04-01
    provides: "run_eval function and eval_metrics module with exact_match/edit_similarity"
  - phase: 03-curriculum-training-loop/03-01
    provides: "multi-stage curriculum loop in training/curriculum.py"
provides:
  - "Per-stage eval after each trainer.fit() logging eval_exact_match + eval_edit_sim to ClearML"
  - "Final test-set eval in separate eval-final ClearML Task after all stages"
  - "Greedy decode with max_new_tokens = stage_cfg.max_lines * 50"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "run_eval patched via source-module monkey-patching (not patch() targeting curriculum namespace) because lazy 'from X import Y' inside main() binds at call time"
    - "get_logger().report_scalar(title='eval', series='eval_exact_match', iteration=stage_idx+1) for ClearML per-stage scalar logging"
    - "Ground truths collected from dataloader.dataset in list comprehension; shuffle=False ensures order alignment with batches"

key-files:
  created: []
  modified:
    - training/curriculum.py
    - tests/test_curriculum.py

key-decisions:
  - "run_eval imported inside main() alongside other lazy imports — same EXP-01 pattern as pytorch/lightning to keep Task.init before any ML import"
  - "Ground truths from eval_val_dl.dataset (not separate dataloader) — shuffle=False ensures index alignment with tokenized batches"
  - "eval-final Task created after the stage loop with task_name=eval-final — separate ClearML task for test-set eval (EVAL-03)"
  - "Mocked run_eval via training.eval_metrics module patching (not patch('training.curriculum.run_eval')) because lazy import binds name at main() call time, not at module load"

patterns-established:
  - "Mock source module attributes for lazy imports (from X import Y inside functions)"

requirements-completed: [EVAL-01, EVAL-02, EVAL-03]

# Metrics
duration: 8min
completed: 2026-03-30
---

# Phase 4 Plan 2: Evaluation Summary

**Per-stage eval after trainer.fit() and final test-set eval via eval-final ClearML Task, both logging exact_match and edit_sim scalars**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-30T15:30:00Z
- **Completed:** 2026-03-30T15:38:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Wired `run_eval` into the curriculum loop: once per stage after `trainer.fit()`, before adapter save
- Added `eval-final` ClearML Task after the stage loop for test-set evaluation
- `max_new_tokens = stage_cfg.max_lines * 50` for greedy decode scaling per stage complexity
- Updated test suite: 6 new eval integration tests + updated 2 existing count assertions; all 13 curriculum tests pass alongside 10 eval_metrics tests (23 total)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add per-stage eval and final test-set eval to curriculum.py** - `47dc498` (feat)
2. **Task 2: Update curriculum tests to verify eval integration** - `2f1c15b` (test)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `training/curriculum.py` - Added `from training.eval_metrics import run_eval` inside main(), per-stage eval block after trainer.fit(), eval-final Task block after the stage loop
- `tests/test_curriculum.py` - Added dataset mocks on val/test DataLoaders, get_logger mock, run_eval source-module patching, 6 new test methods, updated 2 count assertions

## Decisions Made

- `run_eval` mocked via `training.eval_metrics.run_eval = mock` (not `patch("training.curriculum.run_eval")`) because the lazy `from training.eval_metrics import run_eval` inside `main()` binds the name in `training.curriculum` namespace at call time, not at module load time — `patch()` would raise `AttributeError` since the attribute doesn't exist before main() runs.
- `test_train_dataloader_receives_correct_curriculum_stage` updated to assert val_stages == `[1,1,2,2,3,3]` (sorted) since val_dataloader is now called twice per stage: once during training setup, once for per-stage eval.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed `patch("training.curriculum.run_eval")` approach**
- **Found during:** Task 2 (test execution)
- **Issue:** `unittest.mock.patch("training.curriculum.run_eval")` raises `AttributeError` because `run_eval` is bound in curriculum's namespace only when `main()` executes the `from training.eval_metrics import run_eval` statement, not at module load time. The patch target didn't exist at patch application time.
- **Fix:** Monkey-patched `training.eval_metrics.run_eval` directly (same pattern as other source-module mocks in the test file). The `from ... import` inside `main()` then reads the patched function.
- **Files modified:** tests/test_curriculum.py
- **Verification:** All 13 tests pass.
- **Committed in:** 2f1c15b (Task 2 commit)

**2. [Rule 1 - Bug] Updated val_dataloader stage assertion from [1,2,3] to [1,1,2,2,3,3]**
- **Found during:** Task 2 (test design)
- **Issue:** Adding per-stage eval causes val_dataloader to be called twice per stage — once in the training setup block and once for the eval block. The existing assertion `sorted(val_stages) == [1, 2, 3]` would fail.
- **Fix:** Updated assertion to `self.assertEqual(sorted(val_stages), [1, 1, 2, 2, 3, 3])`.
- **Files modified:** tests/test_curriculum.py
- **Verification:** All 13 tests pass.
- **Committed in:** 2f1c15b (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 - bug)
**Impact on plan:** Both fixes necessary for correctness. No scope creep.

## Issues Encountered

None beyond the two auto-fixed deviations above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- EVAL-01, EVAL-02, EVAL-03 requirements complete
- Phase 4 evaluation pipeline is fully wired: eval_metrics module (Plan 01) + curriculum integration (Plan 02)
- No blockers

---
*Phase: 04-evaluation*
*Completed: 2026-03-30*
