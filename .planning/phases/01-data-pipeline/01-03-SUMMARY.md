---
phase: 01-data-pipeline
plan: 03
subsystem: data
tags: [pytorch, pytorch-lightning, dataloader, curriculum-learning, fim, replay, tdd]

# Dependency graph
requires:
  - phase: 01-data-pipeline plan 01
    provides: load_jsonl, split_dataset, StageConfig, PipelineConfig
  - phase: 01-data-pipeline plan 02
    provides: create_fim_sample, GapError, FIM token constants

provides:
  - "data/curriculum_dataset.py: FIMDataset (pre-generates FIM samples per stage), HybridReplayDataset (25% deterministic replay mix)"
  - "data/datamodule.py: CurriculumDataModule (LightningDataModule with per-stage train/val/test DataLoaders)"
  - "tests/test_datamodule.py: 10 automated tests verifying replay ratio, batch structure, and stage isolation"

affects:
  - 02-qlora-training (calls dm.train_dataloader(stage) and dm.val_dataloader(stage))
  - 03-curriculum (trains successive stages via CurriculumDataModule.train_dataloader(N))

# Tech tracking
tech-stack:
  added: [torch 1.10.2, pytorch-lightning 1.5.10, torchmetrics, fsspec, tensorboard, absl-py, pyDeprecate]
  patterns:
    - "TDD red-green: write failing tests, then implement"
    - "Index-based deterministic replay: idx%4==0 routes to prior stage (25%), no runtime randomness"
    - "Pre-generate FIM samples at dataset construction time (eager, not lazy)"
    - "Markdown fence stripping: extract code from output field via regex before FIM gap creation"

key-files:
  created:
    - data/curriculum_dataset.py
    - data/datamodule.py
    - tests/test_datamodule.py
  modified: []

key-decisions:
  - "Code is in 'output' field (not 'input') wrapped in ```python fences; added _extract_code() to strip them"
  - "idx%4==0 routing (deterministic 25% replay) preferred over runtime randomness for reproducible batches"
  - "FIMDataset eager pre-generation: all FIM samples built at __init__ time; GapError records silently skipped"
  - "HybridReplayDataset.__len__ returns len(current) so DataLoader iterates current-stage length only"
  - "_collate_fn returns plain Python lists (not tensors) — tokenization deferred to Phase 2"
  - "Installed torch/pytorch-lightning via pip --no-deps sequentially due to Python 3.6 grpcio/aiohttp build failures"

patterns-established:
  - "Replay routing: prior and idx % 4 == 0 pattern for deterministic 25% replay without RNG state"
  - "_extract_code() helper: regex strips ```python...``` markdown fences from output field"
  - "FIMDataset construction: random.Random(seed).choice(line_choices) + seed+i per record for varied n"

requirements-completed: [DATA-05]

# Metrics
duration: 22min
completed: 2026-03-29
---

# Phase 1 Plan 03: Curriculum DataModule Summary

**PyTorch Lightning CurriculumDataModule with FIMDataset (eager PSM sample generation) and HybridReplayDataset (deterministic 25% prior-stage replay via idx%4), wiring JSONL loader and FIM gap creator into stage-specific DataLoaders**

## Performance

- **Duration:** 22 min
- **Started:** 2026-03-29T10:36:54Z
- **Completed:** 2026-03-29T10:58:16Z
- **Tasks:** 2 (TDD: 4 commits — 2x test then feat)
- **Files modified:** 3

## Accomplishments
- FIMDataset pre-generates all FIM samples at construction; GapError records silently skipped; uses seed+i per record for reproducible varied gap sizes
- HybridReplayDataset routes idx%4==0 to prior stage (25% deterministic replay), idx%4!=0 to current stage (75%) — no runtime RNG, fully reproducible batches
- CurriculumDataModule.train_dataloader(N) wraps HybridReplayDataset; val_dataloader(N) uses current stage only (no replay); test_dataloader() uses last stage
- _collate_fn collates list[dict] into dict[str, list] — plain Python lists, no tokenization (deferred to Phase 2)
- All 10 pytest tests pass; end-to-end smoke test confirms Stage 2 replay fraction = 0.25 and all Stage 1 fim_text strings have correct FIM_BEGIN/FIM_HOLE/FIM_END/EOT structure

## Task Commits

Each task was committed atomically:

1. **TDD RED — Failing tests Task 1** - `7d819d4` (test)
2. **TDD GREEN — FIMDataset + HybridReplayDataset** - `b2ead8d` (feat)
3. **TDD GREEN — CurriculumDataModule** - `360c8c7` (feat)
4. **Bug fix — wrong data field (output vs input)** - `664e393` (fix)

**Plan metadata:** (pending — see final commit)

_Note: TDD tasks have multiple commits (test RED then feat GREEN). Bug fix is a separate commit._

## Files Created/Modified
- `data/curriculum_dataset.py` — _extract_code(), FIMDataset, HybridReplayDataset
- `data/datamodule.py` — _collate_fn, CurriculumDataModule (setup, train_dataloader, val_dataloader, test_dataloader)
- `tests/test_datamodule.py` — 10 pytest tests for dataset and datamodule behavior

## Decisions Made
- Code field: `output` (not `input`) contains Python code in the JSONL dataset; wrapped in markdown fences that must be stripped
- Deterministic replay: `idx % 4 == 0` gives exactly 25% without any runtime RNG — same idx always returns same item
- Eager pre-generation: FIMDataset builds all samples at `__init__` time, making `__getitem__` O(1) and avoiding repeated FIM computation during training
- Plain list collation: `_collate_fn` returns Python lists, not tensors — tokenization (tokenizer.encode) belongs in Phase 2 training loop
- Test record count: 300 records needed for stage2 to produce sufficient samples (2+ consecutive eligible lines rare in short code snippets)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] test_hybrid_replay_ratio used too few records (50) causing ZeroDivisionError**
- **Found during:** Task 1 GREEN phase (first test run)
- **Issue:** FIMDataset with stage2 config (min_lines=2) produces 0 samples from 50 records because 2+ consecutive eligible lines are rare in short code snippets
- **Fix:** Increased test to use 300 records; added assertion that both stage1_ds and stage2_ds are non-empty
- **Files modified:** tests/test_datamodule.py
- **Verification:** test_hybrid_replay_ratio passes with replay_pct=0.25
- **Committed in:** `7d819d4` (test RED commit)

**2. [Rule 1 - Bug] FIMDataset used wrong data field: r["input"] instead of r["output"]**
- **Found during:** Task 2 verification (end-to-end smoke test)
- **Issue:** The JSONL 'input' field contains plain-text task descriptions, not Python code. Code is in 'output' wrapped in ```python ... ``` markdown fences. With 'input', stage2 produces 1 sample from 19850 train records; replay fraction = 0.01 (expected ~0.25)
- **Fix:** Added _extract_code() helper using regex to strip markdown fences from 'output' field; falls back to 'input' if 'output' absent. After fix: stage1=19842, stage2=14428, stage3=6339 samples
- **Files modified:** data/curriculum_dataset.py
- **Verification:** Smoke test passes with replay fraction = 0.25; all 10 tests pass
- **Committed in:** `664e393` (dedicated fix commit)

---

**Total deviations:** 2 auto-fixed (2x Rule 1 — bugs)
**Impact on plan:** Both essential for correctness. The data field bug would have produced ~0 valid training samples for stages 2+. No scope creep.

## Issues Encountered
- Python 3.6 (Anaconda base) has complex dependency resolution for pytorch-lightning. grpcio and aiohttp fail to build wheels. Used `pip install --no-deps` sequentially for each dependency (torch, pytorch-lightning, torchmetrics, fsspec, pyDeprecate, tqdm, tensorboard, absl-py, protobuf<4, typing_extensions) to avoid build failures.
- python-codes-25k.jsonl is not in the worktree (.gitignore). Created symlink to /home/ido/git/ccc/python-codes-25k.jsonl for test runs.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `CurriculumDataModule(config).setup()` then `train_dataloader(N)` is ready for Phase 2 training loop
- Batch structure: `{"fim_text": [str, ...], "prefix": [...], "suffix": [...], "middle": [...], "n_lines": [int, ...]}` — each value is a Python list of batch_size items
- Stage 1 DataLoader: all items n_lines=1, FIM_BEGIN/FIM_HOLE/FIM_END/EOT structure verified
- Stage 2 DataLoader: ~25% Stage 1 replay (n_lines=1), ~75% Stage 2 samples (n_lines in [2,3])
- Phase 2 must: tokenize fim_text strings before feeding to model, choose batch_size appropriate for GPU memory

---
*Phase: 01-data-pipeline*
*Completed: 2026-03-29*
