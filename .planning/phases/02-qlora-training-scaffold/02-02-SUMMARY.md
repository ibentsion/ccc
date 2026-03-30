---
phase: 02-qlora-training-scaffold
plan: "02"
subsystem: training
tags: [qlora, peft, bitsandbytes, lightning, lora, quantization]
dependency_graph:
  requires: ["02-01"]
  provides: ["QLoRALightningModule", "training/model.py"]
  affects: ["02-03-clearml-trainer"]
tech_stack:
  added: [peft, bitsandbytes, pytorch_lightning]
  patterns:
    - "3-step QLoRA init: from_pretrained -> prepare_model_for_kbit_training -> get_peft_model"
    - "device_map={\"\":0} to avoid DDP conflicts with Lightning"
    - "PagedAdamW32bit + ConstantLR for constant-LR QLoRA training"
    - "PEFT save_pretrained for adapter-only checkpoint (not full model)"
    - "sys.modules mock injection for no-GPU unit testing of heavy ML deps"
key_files:
  created:
    - training/model.py
    - tests/test_model.py
  modified: []
decisions:
  - "device_map={\"\":0} used instead of device_map=auto to prevent Lightning DDP conflicts"
  - "save_hyperparameters() NOT called; hyperparams logged in Plan 03 via ClearML task.connect()"
  - "gradient_checkpointing_enable called with use_reentrant=False to avoid reentrant autograd issues"
  - "ConstantLR(factor=1.0, total_iters=0) implements no-decay constant LR per TRAIN-05"
metrics:
  duration: "5 min"
  completed: "2026-03-29"
  tasks: 2
  files: 2
---

# Phase 02 Plan 02: QLoRA LightningModule Summary

QLoRALightningModule wrapping quantized DeepSeek-Coder with LoRA adapters via exact 3-step init sequence, PagedAdamW32bit optimizer, and PEFT-only adapter saving.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | QLoRA LightningModule | 93d51cf | training/model.py |
| 2 | Test model init order and adapter save | a93f6e8 | tests/test_model.py |

## What Was Built

**training/model.py** — `QLoRALightningModule(pl.LightningModule)`:
- `__init__`: Exact 3-step QLoRA init (from_pretrained with BitsAndBytesConfig + device_map={"":0} -> prepare_model_for_kbit_training -> get_peft_model with LoraConfig)
- gradient_checkpointing_enable with use_reentrant=False
- `training_step` / `validation_step`: loss from HF CausalLM (ignore_index=-100 built-in), logs train_loss/val_loss
- `configure_optimizers`: PagedAdamW32bit + ConstantLR scheduler
- `save_adapter`: PEFT save_pretrained (adapter weights only)

**tests/test_model.py** — 7 mock-based tests covering:
- Exact 3-step init order verified via call_order tracking
- device_map={"":0} (not "auto")
- BitsAndBytesConfig load_in_4bit=True
- LoraConfig with all 7 target modules
- training_step logs train_loss
- save_adapter calls model.save_pretrained(path)
- configure_optimizers returns PagedAdamW32bit + ConstantLR

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fake LightningModule missing `__call__` dispatch**
- **Found during:** Task 2 (TDD GREEN phase, test_training_step)
- **Issue:** Mock `_LightningModule` didn't dispatch `__call__` to `forward()`, so `self(batch...)` in training_step raised TypeError
- **Fix:** Added `__call__` method to fake base class that delegates to `self.forward()`
- **Files modified:** tests/test_model.py
- **Commit:** a93f6e8 (included in task commit)

**2. [Rule 1 - Bug] Fake `_PagedAdamW32bit` didn't extend `torch.optim.Optimizer`**
- **Found during:** Task 2 (test_configure_optimizers)
- **Issue:** `torch.optim.lr_scheduler.ConstantLR` checks `isinstance(optimizer, Optimizer)` and raised TypeError
- **Fix:** Made `_PagedAdamW32bit` extend `torch.optim.Optimizer` with proper super().__init__()
- **Files modified:** tests/test_model.py
- **Commit:** a93f6e8 (included in task commit)

**3. [Rule 3 - Blocking] `transformers.__spec__` was None, breaking pytorch_lightning import**
- **Found during:** Task 2 (collection-time error)
- **Issue:** Setting `sys.modules["transformers"] = fake_module` with `__spec__=None` caused `importlib.util.find_spec` to raise `ValueError` inside torchmetrics import chain
- **Fix:** Set `__spec__ = importlib.util.spec_from_loader(name, loader=None)` on all fake modules
- **Files modified:** tests/test_model.py
- **Commit:** a93f6e8 (included in task commit)

## Known Stubs

None - training/model.py wires to real PEFT/HF APIs. No placeholder data flows.

## Self-Check: PASSED
