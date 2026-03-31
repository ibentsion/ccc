# Phase 4: Evaluation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.

**Date:** 2026-03-31
**Phase:** 04-evaluation

---

## Area 1: Eval Architecture

**Q:** Where does evaluation live?
**Options:** Inline in curriculum.py / Standalone eval script / Both
**Selected:** Inline in curriculum.py

**Q:** Where should EVAL-03 (final test-set eval) live?
**Options:** Inline at end of curriculum.py / Separate eval script
**Selected:** Inline at end of curriculum.py

---

## Area 2: Inference Strategy

**Q:** How should the model generate completions for scoring?
**Options:** Greedy decode / Beam search (k=4)
**Selected:** Greedy decode

**Q:** How to cap max_new_tokens for generation?
**Options:** Derive from stage config / Fixed cap (256)
**Selected:** Derive from stage config (max_lines * 50)

---

## Area 3: Validation Set Scope

**Q:** Which samples to evaluate on per stage?
**Options:** Stage-specific clean / Hybrid replay val dataloader / Full val set flat
**Selected:** Stage-specific clean

*Note: val_dataloader already confirmed to use no hybrid replay — no datamodule changes needed.*

---

## Area 4: ClearML Placement

**Q:** Where do per-stage eval metrics (EVAL-01/02) get logged?
**Options:** Same training Task / Separate eval Task per stage
**Selected:** Same training Task (logged before task.close())
