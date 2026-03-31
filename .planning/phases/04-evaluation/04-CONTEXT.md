# Phase 4: Evaluation - Context

**Gathered:** 2026-03-31
**Status:** Ready for planning

<domain>
## Phase Boundary

Systematic evaluation of the curriculum-trained model. Computes Exact Match and Edit Similarity per curriculum stage on the validation set, and runs a final evaluation on the held-out test set. All metrics logged to ClearML.

Out of scope: inference serving, Pass@1 execution testing, CodeBLEU (all v2).

</domain>

<decisions>
## Implementation Decisions

### Eval Architecture
- **D-01:** Per-stage eval (EVAL-01/02) runs inline in `training/curriculum.py` — after each stage's `trainer.fit()` completes and before `task.close()`. No separate script.
- **D-02:** Final test-set eval (EVAL-03) also runs inline at the end of `curriculum.py`, after the last stage's task closes. Opens a new ClearML Task named `eval-final` for the test run.

### Inference Strategy
- **D-03:** Greedy decode (no beam search, no sampling). Deterministic and fast — standard for Exact Match.
- **D-04:** `max_new_tokens = stage_cfg.max_lines * 50`. Scales with gap size (Stage 1 ≈ 50 tokens, Stage 3 ≈ 150). Avoids padding waste and truncation.

### Validation Set Scope
- **D-05:** Stage-specific clean samples — `datamodule.val_dataloader(curriculum_stage=N)`. Already confirmed: `val_dataloader` uses no hybrid replay (only current-stage FIMDataset). No changes needed to datamodule.
- **D-06:** Test-set eval uses `datamodule.test_dataloader()` which uses the last stage config (already in codebase).

### ClearML Logging
- **D-07:** Per-stage eval metrics (`eval_exact_match`, `eval_edit_sim`) logged into the same ClearML Task as training for that stage, using `task.get_logger().report_scalar(...)`. Logged before `task.close()`.
- **D-08:** Final test-set metrics logged to a separate `eval-final` ClearML Task (new Task.init after last stage's task closes). Follows EXP-01 pattern — Task.init before pytorch model loads.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements
- `.planning/REQUIREMENTS.md` — EVAL-01, EVAL-02, EVAL-03 define the acceptance criteria for this phase

### Existing Code (integration points)
- `training/curriculum.py` — the curriculum loop where eval will be added (inline)
- `training/model.py` — QLoRALightningModule; model.model.generate() is the generation entry point
- `data/datamodule.py` — CurriculumDataModule; val_dataloader and test_dataloader confirmed to provide clean stage-specific samples
- `training/config.py` — TrainingConfig; StageConfig.max_lines used to derive max_new_tokens

No external evaluation specs — requirements fully captured in decisions above.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `CurriculumDataModule.val_dataloader(curriculum_stage=N)` — already stage-specific clean (no hybrid replay). Ready to use for per-stage eval.
- `CurriculumDataModule.test_dataloader()` — uses last-stage config. Ready to use for EVAL-03.
- `QLoRALightningModule.model` — the PEFT model; call `.generate()` on it for inference.
- `training/curriculum.py` — the existing loop structure; eval inserts after `trainer.fit()`, before `task.close()`.

### Established Patterns
- EXP-01: `Task.init()` before any pytorch/lightning/peft imports. The `eval-final` task must follow this — but since pytorch is already imported in the curriculum loop by that point, the final eval task will be opened with pytorch already in memory. This is acceptable since the eval-final task doesn't need TensorBoard auto-hook.
- ClearML scalar logging: use `task.get_logger().report_scalar(title, series, value, iteration)`.

### Integration Points
- Eval inserts into `curriculum.py`'s stage loop: after `trainer.fit()` line 96, before `task.close()` line 110.
- Final eval appends after the `for` loop ends (after last `task.close()`).

</code_context>

<specifics>
## Specific Ideas

No specific implementation references beyond the decisions above — standard greedy eval loop.

</specifics>

<deferred>
## Deferred Ideas

- Pass@1 execution test — v2 requirement (EVAL-V2-01), requires Python sandbox
- CodeBLEU metric — v2 requirement (EVAL-V2-02)
- HumanEval-Infilling benchmark — v2 requirement (EVAL-V2-03)

None of these were discussed — already captured in REQUIREMENTS.md as v2 scope.

</deferred>

---

*Phase: 04-evaluation*
*Context gathered: 2026-03-31*
