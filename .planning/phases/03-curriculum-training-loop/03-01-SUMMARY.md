---
phase: 03-curriculum-training-loop
plan: 01
subsystem: training
tags: [pytorch-lightning, clearml, peft, qlora, curriculum-learning]

requires:
  - phase: 02-qlora-training-scaffold
    provides: QLoRALightningModule, train.py single-stage pattern, ClearML EXP-01 pattern
  - phase: 01-data-pipeline
    provides: CurriculumDataModule, PipelineConfig, StageConfig, HybridReplayDataset

provides:
  - training/curriculum.py: multi-stage curriculum entry point that loops over all stages automatically
  - tests/test_curriculum.py: 7 mock-based tests verifying loop logic without GPU

affects:
  - 03-02 and later plans that extend or evaluate the curriculum loop
  - Any plan adding new stages or modifying stage sequencing logic

tech-stack:
  added: []
  patterns:
    - "Lazy imports inside main() so Task.init fires before any pytorch import (EXP-01)"
    - "Fresh QLoRALightningModule + Trainer per stage for clean optimizer/scheduler state"
    - "PeftModel.from_pretrained for adapter continuity between stages"
    - "sys.modules injection + module attribute monkey-patching for Python 3.6 mock tests"

key-files:
  created:
    - training/curriculum.py
    - tests/test_curriculum.py
  modified:
    - training/tokenizer.py

key-decisions:
  - "All pytorch/lightning/peft imports inside main() so Task.init fires before any pytorch import (EXP-01 pattern from train.py)"
  - "Fresh QLoRALightningModule + pl.Trainer per stage ensures constant LR with no optimizer state carryover"
  - "task.close() called before next Task.init so each stage has its own isolated ClearML task"
  - "Test mocking uses sys.modules injection + curriculum.Task direct patch rather than unittest.mock.patch() because curriculum binds Task at module level"

patterns-established:
  - "Curriculum loop pattern: setup datamodule once, loop stages with fresh model+trainer+task per stage"
  - "Adapter continuity: PeftModel.from_pretrained(model.model.base_model.model, prev_adapter_dir)"
  - "ClearML naming: f'stage_{n}_gap_{min_lines}_{max_lines}'"

requirements-completed: [TRAIN-04]

duration: 8min
completed: 2026-03-30
---

# Phase 3 Plan 1: Curriculum Training Loop Summary

**Multi-stage curriculum loop in training/curriculum.py that chains 3 stages automatically, each with its own ClearML Task, fresh optimizer, and PeftModel adapter continuity via from_pretrained**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-30T10:08:08Z
- **Completed:** 2026-03-30T10:15:42Z
- **Tasks:** 1 (TDD: red → green)
- **Files modified:** 3

## Accomplishments

- `training/curriculum.py` loops over all `PipelineConfig.stages` without manual intervention
- Each stage gets a scoped ClearML Task with `f"stage_{n}_gap_{min}_{max}"` naming, closed before next iteration
- `PeftModel.from_pretrained` loads prior adapter for stages > 1 so learning accumulates
- Fresh `QLoRALightningModule` + `pl.Trainer` per stage ensures constant LR scheduler resets
- `train_dataloader(curriculum_stage=N)` and `val_dataloader(curriculum_stage=N)` called per stage for correct hybrid replay mix
- All 7 mock-based tests pass in Python 3.6 without GPU or ML libraries

## Task Commits

Each task was committed atomically:

1. **Task 1: Create training/curriculum.py — multi-stage curriculum loop** - `99c9add` (feat)

## Files Created/Modified

- `training/curriculum.py` - Multi-stage curriculum entry point with stage loop, per-stage ClearML Tasks, adapter continuity, and artifact upload
- `tests/test_curriculum.py` - 7 mock-based tests verifying loop behavior (Task.init count, naming, close, PeftModel, Trainer count, save_adapter paths, curriculum_stage params)
- `training/tokenizer.py` - Fixed `list[dict]` → `List[dict]` for Python 3.6 compatibility (deviation fix)

## Decisions Made

- Lazy imports inside `main()` consistent with `train.py` EXP-01 pattern (Task.init before pytorch imports)
- `curriculum.Task` patched directly on the module object in tests because `from clearml import Task` binds at module load — `sys.modules` injection alone insufficient
- `QLoRALightningModule` and `CurriculumDataModule` patched on their source modules (training.model, data.datamodule) since they are imported inside main()

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed `list[dict]` type hint in training/tokenizer.py for Python 3.6**
- **Found during:** Task 1 (writing tests)
- **Issue:** `list[dict]` PEP 585 syntax was introduced in Python 3.9; test runner is Python 3.6 which raises `TypeError: 'type' object is not subscriptable`
- **Fix:** Added `from typing import List` import and changed `list[dict]` to `List[dict]` in `TokenizedCollator.__call__`
- **Files modified:** training/tokenizer.py
- **Verification:** Tests run and module imports without error
- **Committed in:** 99c9add (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Fix necessary for test suite to run. No scope creep.

## Issues Encountered

- Python 3.6 test environment doesn't support `unittest.mock.patch()` for lazy imports easily; resolved by using sys.modules injection combined with direct module attribute patching after reload
- `call.args` / `call.kwargs` properties don't exist in Python 3.6 unittest.mock; used `call[0]` / `call[1]` indexing instead

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `training/curriculum.py` is the main curriculum entry point; ready to be invoked once GPU environment and dataset are available
- All 7 behavioral invariants verified via mock tests
- Adapter continuity, fresh optimizer, and per-stage ClearML isolation are all in place

## Self-Check: PASSED

- training/curriculum.py: FOUND
- tests/test_curriculum.py: FOUND
- Commit 99c9add: FOUND

---
*Phase: 03-curriculum-training-loop*
*Completed: 2026-03-30*
