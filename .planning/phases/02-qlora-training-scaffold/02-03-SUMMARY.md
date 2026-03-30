---
phase: 02-qlora-training-scaffold
plan: 03
subsystem: training
tags: [clearml, pytorch-lightning, peft, tensorboard, qlora, curriculum-learning]

# Dependency graph
requires:
  - phase: 02-01
    provides: TrainingConfig, TokenizedCollator, load_tokenizer
  - phase: 02-02
    provides: QLoRALightningModule with save_adapter
  - phase: 01-03
    provides: CurriculumDataModule, FIMDataset, HybridReplayDataset
provides:
  - training/train.py — main entry point executing a full single-stage QLoRA training run
  - CurriculumDataModule with optional collate_fn injection for tokenized batches
affects:
  - phase-03-curriculum-orchestration

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "ClearML Task.init before all pytorch/lightning imports for TensorBoard auto-capture"
    - "Collate function injection pattern: DataModule accepts optional collate_fn, no circular import"
    - "PEFT adapter saved manually via save_pretrained, Lightning checkpointing disabled"
    - "All heavy imports inside main() to ensure Task.init runs at module import time"

key-files:
  created:
    - training/train.py
  modified:
    - data/datamodule.py

key-decisions:
  - "All pytorch/lightning imports inside main() so Task.init() fires before any pytorch import (EXP-01 requirement)"
  - "collate_fn injected into CurriculumDataModule rather than imported in datamodule.py — keeps data/training dependency one-directional"
  - "enable_checkpointing=False prevents Lightning from saving full quantized model; PEFT save_pretrained saves only adapter weights"
  - "log_every_n_steps=1 for fine-grained loss curves in ClearML"

patterns-established:
  - "ClearML integration pattern: Task.init at module level before pytorch, hyperparams via task.connect(), artifact via task.upload_artifact()"
  - "DataModule collate injection pattern: accept collate_fn=None, fall back to module-level _collate_fn"

requirements-completed: [EXP-01, EXP-02, EXP-03, EXP-04, TRAIN-07]

# Metrics
duration: 2min
completed: 2026-03-30
---

# Phase 02 Plan 03: Training Entry Point Summary

**ClearML-integrated QLoRA training entry point with TensorBoard loss logging, hyperparameter tracking, PEFT adapter artifact upload, and TokenizedCollator injection into CurriculumDataModule**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-30T03:51:55Z
- **Completed:** 2026-03-30T03:53:46Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Updated `CurriculumDataModule` to accept optional `collate_fn`, enabling TokenizedCollator injection without circular imports
- Created `training/train.py` with ClearML Task.init before any pytorch/lightning imports (EXP-01), TensorBoardLogger for auto-capture (EXP-02), hyperparameters via `task.connect()` (EXP-03), and adapter uploaded via `task.upload_artifact()` (EXP-04)
- PEFT adapter saved to disk and Lightning checkpointing disabled so only the LoRA adapter weights are persisted (TRAIN-07)

## Task Commits

1. **Task 1: Update DataModule to use TokenizedCollator** - `e04eee8` (feat)
2. **Task 2: Training entry point with ClearML integration** - `1ab18a4` (feat)

## Files Created/Modified
- `training/train.py` — Main training entry point wiring ClearML, Lightning Trainer, PEFT adapter save
- `data/datamodule.py` — Added `collate_fn=None` parameter and `self.custom_collate_fn` injection in all dataloaders

## Decisions Made
- All pytorch/lightning imports inside `main()` so `Task.init()` fires before any pytorch import — required for ClearML TensorBoard auto-hook to attach (EXP-01)
- Collate function injected into CurriculumDataModule rather than imported directly — keeps data/training dependency one-directional, no circular import risk
- `enable_checkpointing=False` prevents Lightning from saving full quantized model state; `save_adapter` saves only adapter weights via PEFT
- `log_every_n_steps=1` for fine-grained per-step loss curves visible in ClearML

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- `python-codes-25k.jsonl` not present in worktree — symlinked from repo root to allow test_datamodule.py to pass (file is gitignored/untracked in worktree, not a code change)

## User Setup Required
None - no external service configuration required beyond ClearML credentials already configured in the environment.

## Next Phase Readiness
- Phase 2 scaffold is complete: `python training/train.py` will execute a full training loop on 1-line gaps with loss in ClearML and adapter on disk
- Phase 3 (curriculum orchestration) can now build on top of this entry point to chain stages
- Verify `use_reentrant=False` gradient flow on first real run (loss should decrease in first 50 steps)
- Verify ClearML scalar capture by checking TensorBoard hook attaches after `Task.init()` placement

---
*Phase: 02-qlora-training-scaffold*
*Completed: 2026-03-30*
