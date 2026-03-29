---
phase: 01-data-pipeline
plan: 01
subsystem: data
tags: [jsonl, dataset, splitting, dataclasses, python, pytest, tdd]

# Dependency graph
requires: []
provides:
  - load_jsonl function: loads python-codes-25k.jsonl into list of dicts
  - split_dataset function: deterministic train/val/test splits via seeded shuffle
  - StageConfig dataclass: curriculum stage configuration (stage, min_lines, max_lines)
  - PipelineConfig dataclass: top-level pipeline config with split ratios, seed, stages list
affects:
  - 01-data-pipeline (plans 02 and 03 import StageConfig/PipelineConfig and use split_dataset output)
  - 02-training-loop (needs train/val/test splits for DataLoader construction)

# Tech tracking
tech-stack:
  added: [pytest, dataclasses (Python 3.6 backport)]
  patterns: [TDD red-green, seeded shuffle for reproducibility, Python stdlib only for core logic]

key-files:
  created:
    - data/__init__.py
    - data/config.py
    - data/dataset.py
    - tests/__init__.py
    - tests/test_dataset.py
    - .gitignore
  modified: []

key-decisions:
  - "Used random.Random(seed).shuffle on a copy of records to ensure original list not mutated and determinism is isolated per call"
  - "Python stdlib only (json, random, dataclasses) — no external deps except pytest for testing"
  - "Used anaconda Python 3.6 (only env with pytest available); installed dataclasses backport via pip"

patterns-established:
  - "Seeded shuffle pattern: random.Random(seed).shuffle(list(records)) — always copy first"
  - "Ratio validation: abs(sum - 1.0) > 1e-6 raises ValueError"

requirements-completed: [DATA-01, DATA-04]

# Metrics
duration: 3min
completed: 2026-03-29
---

# Phase 1 Plan 01: Dataset Loading and Splitting Summary

**JSONL loader and deterministic train/val/test splitter using seeded shuffle, with shared StageConfig/PipelineConfig dataclasses for downstream pipeline plans**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-29T10:28:59Z
- **Completed:** 2026-03-29T10:31:46Z
- **Tasks:** 1 (TDD — 2 commits: test + feat)
- **Files modified:** 6

## Accomplishments
- JSONL loader that reads python-codes-25k.jsonl (24,813 records) line-by-line, skipping blanks, with clear FileNotFoundError on missing files
- Deterministic split_dataset function: same seed always yields identical splits; different seeds yield different orderings
- StageConfig and PipelineConfig dataclasses providing shared config types for Plans 02 and 03
- 7 pytest tests covering all behaviors (fields, file-not-found, no data loss, determinism, seed diversity, bad ratios, mutation safety)

## Task Commits

Each task was committed atomically:

1. **RED — failing tests** - `ed35f78` (test)
2. **GREEN — implementation** - `b0f89af` (feat)
3. **Chore — .gitignore** - `c85e11f` (chore)

## Files Created/Modified
- `data/__init__.py` - Empty package init
- `data/config.py` - StageConfig and PipelineConfig dataclasses (stdlib dataclasses module)
- `data/dataset.py` - load_jsonl and split_dataset functions
- `tests/__init__.py` - Empty test package init
- `tests/test_dataset.py` - 7 pytest tests for all dataset behaviors
- `.gitignore` - Ignores __pycache__, .pyc, .pytest_cache, symlinked JSONL

## Decisions Made
- Used `random.Random(seed).shuffle(list(records))` so each call creates an independent RNG with the seed, guaranteeing determinism without shared state or global random mutations
- Remainder records go to test set (not val) so train+val+test == len(records) exactly even with int rounding
- Used anaconda Python 3.6 environment (only environment with pytest installed); installed `dataclasses` backport via pip since Python 3.6 predates the stdlib module

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed dataclasses backport for Python 3.6**
- **Found during:** Task 1 (verifying config imports)
- **Issue:** System has Python 3.6 (anaconda) as only env with pytest; Python 3.6 lacks `dataclasses` module (added in 3.7)
- **Fix:** Ran `/home/ido/anaconda3/bin/pip install dataclasses` to install backport 0.8
- **Files modified:** None (pip install only)
- **Verification:** `from data.config import StageConfig, PipelineConfig` succeeds
- **Committed in:** b0f89af (implementation commit)

**2. [Rule 2 - Missing Critical] Added .gitignore**
- **Found during:** Post-task cleanup
- **Issue:** __pycache__, .pytest_cache, and JSONL symlink were untracked and would pollute commits
- **Fix:** Created .gitignore covering Python artifacts and symlinked JSONL
- **Files modified:** .gitignore
- **Verification:** git status shows clean working tree
- **Committed in:** c85e11f

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 missing critical)
**Impact on plan:** Both necessary for correctness and clean repo. No scope creep.

## Issues Encountered
- `python` not in PATH (only `python3`); plan verification uses `python -m pytest`. Resolved by using anaconda's Python directly `/home/ido/anaconda3/bin/python` for all test runs. The anaconda env is the correct runtime for this project.
- System Python 3.12 lacks pip/venv/pytest — only anaconda Python 3.6 has pytest available.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `load_jsonl` and `split_dataset` ready for Plans 02 and 03 to import from `data.dataset`
- `StageConfig` and `PipelineConfig` ready for import from `data.config`
- Dataset confirmed: 24,813 records split as train=19850, val=2481, test=2482 (seed=42)
- Note: Plan verification command uses `python` not `python3` — anaconda Python 3.6 is the project runtime

---
*Phase: 01-data-pipeline*
*Completed: 2026-03-29*
