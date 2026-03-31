# Roadmap: CodeComplete Fine-Tuning

## Overview

Four phases take this project from raw JSONL to a curriculum-trained QLoRA adapter with per-stage metrics in ClearML. Phase 1 locks the data format before any GPU is touched. Phase 2 validates the full Lightning + PEFT + bitsandbytes + ClearML integration stack on a single training stage. Phase 3 adds the curriculum loop on top of that validated scaffold. Phase 4 runs systematic evaluation on the completed model.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Data Pipeline** - Load JSONL, create FIM-formatted gap pairs, validate PSM token format, produce reproducible train/val/test splits
- [ ] **Phase 2: QLoRA Training Scaffold** - Single-stage training run validating Lightning + PEFT + bitsandbytes + ClearML integration end-to-end
- [x] **Phase 3: Curriculum Training Loop** - Multi-stage curriculum with hybrid replay, stage transitions, per-stage adapter saves and ClearML artifacts (completed 2026-03-30)
- [ ] **Phase 4: Evaluation** - Per-stage and final test-set evaluation with Exact Match, Edit Similarity, and Pass@1

## Phase Details

### Phase 1: Data Pipeline
**Goal**: A validated dataset is ready — FIM-formatted training pairs at all N-line gap levels, reproducible splits, hybrid replay DataLoaders
**Depends on**: Nothing (first phase)
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05
**Success Criteria** (what must be TRUE):
  1. Running the pipeline on `python-codes-25k.jsonl` produces deterministic train/val/test splits (same seed = same split every time)
  2. A sample batch can be inspected and every item has the correct PSM FIM token structure: `<｜fim▁begin｜>{prefix}<｜fim▁hole｜>{suffix}<｜fim▁end｜>`
  3. DataLoaders for any curriculum stage N can be constructed and yield batches mixing 75% current-stage and 25% prior-stage samples
  4. Gap size N is configurable per stage and the masking excludes function signatures, class definitions, and trivially-empty lines
**Plans**: 3 plans

Plans:
- [x] 01-01-PLAN.md — JSONL loader, deterministic train/val/test splitter, shared config dataclasses
- [x] 01-02-PLAN.md — FIM gap creator: select N eligible lines, format as PSM tokens
- [ ] 01-03-PLAN.md — CurriculumDataModule: Lightning DataModule with per-stage hybrid replay DataLoaders

### Phase 2: QLoRA Training Scaffold
**Goal**: A single curriculum stage (1-line gaps) trains to completion with loss logged in ClearML, adapter saved to disk, and no silent integration failures
**Depends on**: Phase 1
**Requirements**: TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-05, TRAIN-06, TRAIN-07, EXP-01, EXP-02, EXP-03, EXP-04
**Success Criteria** (what must be TRUE):
  1. Training loss decreases monotonically over the first 50 steps, confirming the 3-step QLoRA init order is correct and gradients flow to adapter weights
  2. A ClearML Task appears in the UI with loss curves captured via TensorBoard auto-hook and hyperparameters logged under the task
  3. A PEFT adapter directory is saved to disk after training completes and is successfully uploaded to ClearML as a named artifact
  4. The saved `.ckpt` file contains only adapter weights (not frozen base-model weights), keeping checkpoint size compact
**Plans**: 3 plans

Plans:
- [x] 02-01-PLAN.md — Training config dataclass + tokenizer with prompt masking (labels=-100 on non-middle tokens)
- [x] 02-02-PLAN.md — QLoRA LightningModule: 3-step init, PagedAdamW32bit, ConstantLR, PEFT adapter save
- [x] 02-03-PLAN.md — Training entry point with ClearML Task.init, TensorBoardLogger, artifact upload

### Phase 3: Curriculum Training Loop
**Goal**: The full curriculum runs end-to-end — stages advance automatically, replay buffer keeps prior-stage knowledge, each stage produces its own ClearML Task and adapter artifact
**Depends on**: Phase 2
**Requirements**: TRAIN-04
**Success Criteria** (what must be TRUE):
  1. Training automatically advances from Stage 1 (1-line gaps) through configurable later stages without manual intervention
  2. Each curriculum stage produces a separate ClearML Task with its own loss curves and a PEFT adapter artifact
  3. Constant learning rate is maintained across all stages (no cosine decay to near-zero before hard examples appear)
**Plans**: 1 plan

Plans:
- [x] 03-01-PLAN.md — Multi-stage curriculum loop: per-stage ClearML Tasks, adapter continuity, fresh optimizer per stage

### Phase 4: Evaluation
**Goal**: The trained model is evaluated systematically — Exact Match and Edit Similarity per stage on validation set, final test-set results logged to ClearML
**Depends on**: Phase 3
**Requirements**: EVAL-01, EVAL-02, EVAL-03
**Success Criteria** (what must be TRUE):
  1. Exact Match score is computed and logged to ClearML for each curriculum stage on the validation set
  2. Edit Similarity score is computed and logged to ClearML for each curriculum stage on the validation set
  3. A final evaluation run on the held-out test set completes after all curriculum stages and results appear in ClearML
**Plans**: 2 plans

Plans:
- [ ] 04-01-PLAN.md — Eval metrics module: exact_match, edit_similarity, run_eval with TDD
- [ ] 04-02-PLAN.md — Wire eval into curriculum loop: per-stage + final test-set eval with ClearML logging

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Data Pipeline | 1/3 | In progress | - |
| 2. QLoRA Training Scaffold | 1/3 | In Progress|  |
| 3. Curriculum Training Loop | 1/1 | Complete   | 2026-03-30 |
| 4. Evaluation | 0/2 | Not started | - |
