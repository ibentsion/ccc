---
phase: 01-data-pipeline
plan: 02
subsystem: data
tags: [fim, psm, deepseek-coder, gap-creation, pytest, tdd]

# Dependency graph
requires: []
provides:
  - "data/fim.py: create_fim_sample, select_gap_lines, format_psm, GapError"
  - "tests/test_fim.py: 11 automated tests for FIM token structure and exclusion rules"
  - "PSM FIM format locked: FIM_BEGIN/FIM_HOLE/FIM_END/EOT token constants"
affects:
  - 01-data-pipeline (plan 03 — dataset loader consumes create_fim_sample)
  - 02-qlora-training (GapDataset uses create_fim_sample for all training samples)
  - 03-curriculum (CurriculumDataModule uses n parameter to vary gap size per stage)

# Tech tracking
tech-stack:
  added: [pytest (already in anaconda base)]
  patterns:
    - "Run detection algorithm: scan eligible_indices for consecutive integer runs to collect valid_starts"
    - "TDD red-green: write failing imports first, then implement"
    - "PSM format: FIM_BEGIN + prefix + FIM_HOLE + suffix + FIM_END + middle + EOT"

key-files:
  created:
    - data/fim.py
    - data/__init__.py
    - tests/test_fim.py
    - tests/__init__.py
  modified: []

key-decisions:
  - "PSM (Prefix-Suffix-Middle) order confirmed per research SUMMARY and DeepSeek-Coder FIM convention"
  - "Exclusion rule: def/class/blank/comment lines are ineligible — guards training signal quality"
  - "Run detection algorithm chosen over simpler sliding window for correctness with sparse eligible lines"
  - "Using code.split('\\n') not splitlines() to preserve index alignment on empty lines"

patterns-established:
  - "Eligible line detection: strip() empty check, startswith('#'), re.match for def/class"
  - "GapError(ValueError) pattern for domain-specific exceptions in data pipeline"
  - "FIM token constants at module top, copy-pasted (not retyped) to preserve Unicode"

requirements-completed: [DATA-02, DATA-03, DATA-04]

# Metrics
duration: 12min
completed: 2026-03-29
---

# Phase 1 Plan 02: FIM Gap Creation Summary

**PSM FIM formatter with consecutive-eligible-line gap selector using run detection, with 11 passing pytest tests and verified against real python-codes-25k.jsonl data**

## Performance

- **Duration:** 12 min
- **Started:** 2026-03-29T10:28:56Z
- **Completed:** 2026-03-29T10:41:00Z
- **Tasks:** 1 (TDD: 2 commits — test then feat)
- **Files modified:** 4

## Accomplishments
- Implemented `select_gap_lines` using consecutive-integer run detection to find valid gap positions among eligible lines
- Implemented `format_psm` producing correct PSM string: `FIM_BEGIN + prefix + FIM_HOLE + suffix + FIM_END + middle + EOT`
- All 11 pytest tests pass; 5/5 real JSONL samples verify correct token structure and ordering
- Exclusion rules (def, class, blank, comment) enforced — zero ineligible lines ever enter the gap

## Task Commits

Each task was committed atomically:

1. **TDD RED — Failing tests** - `982003e` (test)
2. **TDD GREEN — Implementation** - `994e67f` (feat)

**Plan metadata:** (pending — see final commit)

_Note: TDD task has two commits (test RED then feat GREEN)_

## Files Created/Modified
- `data/fim.py` — GapError, _is_ineligible, select_gap_lines, format_psm, create_fim_sample, FIM token constants
- `data/__init__.py` — Empty package marker
- `tests/test_fim.py` — 11 pytest tests covering token structure, reconstruction, exclusion rules, determinism, n_lines
- `tests/__init__.py` — Empty package marker

## Decisions Made
- PSM order (not SPM): per research SUMMARY "FIM format must use PSM, not SPM" and DeepSeek-Coder canonical format
- Consecutive-run algorithm over sliding window: handles sparse eligible line distributions correctly (e.g. only lines 2,3 and 7,8,9 eligible — a window of 3 must not span the gap)
- `code.split("\n")` not `splitlines()`: preserves index alignment when empty lines appear at end of string
- Test seed pair (0, 1) not (0, 99): confirmed via `random.Random.choice` analysis that seeds 0 and 99 both resolve to index 6 on the 12-element list — switched to seeds 0 and 1 (indices 6 and 2)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test_different_seeds_may_differ using verified seed pair**
- **Found during:** Task 1 GREEN phase (first test run)
- **Issue:** Seeds 0 and 99 both map to `choice=6` on a 12-element list via `random.Random`. Test asserted they differ, which was false.
- **Fix:** Verified all seed → choice mappings; replaced (seed=0, seed=99) with (seed=0, seed=1) which produce choices 6 and 2 respectively.
- **Files modified:** tests/test_fim.py
- **Verification:** `test_different_seeds_may_differ` passes; 11/11 tests green
- **Committed in:** `994e67f` (feat commit, included alongside implementation)

---

**Total deviations:** 1 auto-fixed (Rule 1 — bug in test seed selection)
**Impact on plan:** Necessary fix for test correctness. No scope creep.

## Issues Encountered
- Python 3.12 on this system has no pip/venv/ensurepip. Used Anaconda base (Python 3.6, pytest 3.8.2) which was already installed. Tests run identically.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `create_fim_sample(code, n, seed)` is ready for plan 03 (dataset loader + JSONL pipeline)
- Token constants (FIM_BEGIN, FIM_HOLE, FIM_END, EOT) are locked and importable
- GapError defined — plan 03 can filter/skip samples that raise it
- 5/5 real samples from python-codes-25k.jsonl produce valid PSM structure

---
*Phase: 01-data-pipeline*
*Completed: 2026-03-29*
