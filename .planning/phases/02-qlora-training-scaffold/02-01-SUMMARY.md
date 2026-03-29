---
phase: 02-qlora-training-scaffold
plan: 01
subsystem: training
tags: [qlora, lora, peft, bitsandbytes, transformers, fim, tokenizer, dataclass]

requires:
  - phase: 01-data-pipeline
    provides: "FIM token constants (FIM_BEGIN, FIM_HOLE, FIM_END, EOT) and collate_fn sample dict format"

provides:
  - "TrainingConfig frozen dataclass with all QLoRA/LoRA/optimizer/scheduler hyperparameters"
  - "load_tokenizer function returning tokenizer with left-padding and pad=eos"
  - "TokenizedCollator converting FIM sample dicts to masked tensors for training"

affects: [02-02, 02-03, 02-04, training-loop, clearml-logging]

tech-stack:
  added: [transformers, torch]
  patterns:
    - "FIM masking: labels[i, :pad_len+len(prefix_ids)] = -100 so loss is computed only on middle tokens"
    - "Left-padding for decoder-only models: tokenizer.padding_side = left"
    - "Frozen TrainingConfig dataclass for immutable hyperparameter bundles"

key-files:
  created:
    - training/__init__.py
    - training/config.py
    - training/tokenizer.py
    - tests/test_tokenizer.py

key-decisions:
  - "bnb_4bit_compute_dtype stored as string 'float16' to avoid top-level torch import in config.py"
  - "Left-padding used for decoder-only model batching (padding_side=left)"
  - "Prompt masking: mask pad_len + len(prefix_ids) positions; middle tokens retain real label values"
  - "Tests skip via pytest.importorskip when transformers not installed in test environment"

patterns-established:
  - "FIM collation pattern: tokenize full fim_text, then mask non-middle prefix in labels"
  - "Prefix boundary computed as FIM_BEGIN + prefix + FIM_HOLE + suffix + FIM_END"

requirements-completed: [TRAIN-01, TRAIN-02, TRAIN-05, TRAIN-06]

duration: 5min
completed: 2026-03-29
---

# Phase 2 Plan 01: Training Config and Tokenizer Summary

**Frozen TrainingConfig dataclass with NF4/double-quant QLoRA defaults and TokenizedCollator masking FIM prefix tokens with -100 for middle-only loss computation**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-03-29T20:06:40Z
- **Completed:** 2026-03-29T20:11:40Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- TrainingConfig frozen dataclass with all 7 LoRA target modules, NF4 quantization, double quantization, constant LR scheduler, and paged_adamw_32bit optimizer
- TokenizedCollator implements PSM prompt masking: labels mask padding tokens plus all prefix/FIM/suffix tokens with -100, leaving only middle tokens with real label values
- Tests cover tokenizer config, output keys, single-sample masking correctness, and batch left-padding behavior (skip cleanly when transformers absent)

## Task Commits

1. **Task 1: Training config and tokenizer with prompt masking** - `10a04b1` (feat)
2. **Task 2: Test prompt masking correctness** - `56dc78d` (test)

## Files Created/Modified

- `training/__init__.py` - Empty package init
- `training/config.py` - TrainingConfig frozen dataclass with all hyperparameters
- `training/tokenizer.py` - load_tokenizer and TokenizedCollator with FIM prompt masking
- `tests/test_tokenizer.py` - Four tests: tokenizer config, output keys, single masking, batch masking

## Decisions Made

- `bnb_4bit_compute_dtype` stored as string `"float16"` — avoids importing torch at config module level, resolved to dtype at use site
- Left-padding chosen for decoder-only batching so padding tokens don't interfere with causal attention on real tokens
- Prefix boundary for masking computed as `FIM_BEGIN + prefix + FIM_HOLE + suffix + FIM_END` (matching PSM format from `data/fim.py`)
- Tests use `pytest.importorskip("transformers")` — skip cleanly in environments without transformers instead of hard-failing

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- `transformers` not installed in test environment; tests skip via `pytest.importorskip` as planned. Exit code 5 (no tests collected) is correct behavior.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `training/config.py` and `training/tokenizer.py` are ready for use by the QLoRA model setup (02-02)
- TokenizedCollator accepts the exact dict format produced by `_collate_fn` in `data/datamodule.py`
- Tests will run fully when `transformers` (and the DeepSeek tokenizer) are available in the training environment

---
*Phase: 02-qlora-training-scaffold*
*Completed: 2026-03-29*

## Self-Check: PASSED

- FOUND: training/__init__.py
- FOUND: training/config.py
- FOUND: training/tokenizer.py
- FOUND: tests/test_tokenizer.py
- FOUND: .planning/phases/02-qlora-training-scaffold/02-01-SUMMARY.md
- FOUND: commit 10a04b1 (feat: training config and tokenizer)
- FOUND: commit 56dc78d (test: tokenizer and prompt masking tests)
