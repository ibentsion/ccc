# PyTorch Lightning + QLoRA / PEFT Integration Research

**Researched:** 2026-03-29
**Confidence:** MEDIUM — Core patterns verified via official PEFT docs and Lightning docs; Lightning-specific issues verified via GitHub issues. Some edge cases rely on community sources only.

---

## 1. LightningModule Structure for PEFT/QLoRA

### The Canonical Wrapping Sequence

The mandatory initialization order, verified against PEFT official docs, is:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch

class QLoRALightningModule(LightningModule):
    def __init__(self, model_id: str, lora_config: LoraConfig):
        super().__init__()
        # Step 1: Load with quantization config baked in at from_pretrained time
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=None,      # see section 3 — must be None for Lightning
        )

        # Step 2: Prepare for k-bit training BEFORE get_peft_model
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

        # Step 3: Wrap with PEFT
        self.model = get_peft_model(base_model, lora_config)

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        return outputs.loss

    def configure_optimizers(self):
        # Only optimizer trainable (adapter) params — all others are frozen
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=2e-4)
```

### Why This Order Matters

1. `BitsAndBytesConfig` must be passed to `from_pretrained`. The `BitsandbytesPrecision` Lightning plugin does NOT inject quantization configuration into `from_pretrained` — it only replaces `nn.Linear` layers after the fact, which causes shape mismatches (e.g. `[2048, 2048]` becomes `[262144000, 1]`). Use the HuggingFace config path, not the plugin.
2. `prepare_model_for_kbit_training` must run before `get_peft_model`. It freezes the base model, upcasts normalization layers to fp32 for numerical stability, and enables gradient checkpointing. Running it after PEFT wrapping breaks the input gradient hook.
3. `get_peft_model` adds the trainable LoRA adapters on top of the frozen, quantized base.

### configure_model() vs __init__()

Lightning's `configure_model()` hook is designed for sharded strategies (FSDP, DeepSpeed) where model parameters must be initialized inside a special context. For single-GPU QLoRA with bitsandbytes, **do model setup in `__init__`** — bitsandbytes requires the quantization config to be present when weights are first loaded, and `configure_model()` can interfere with that by allowing Lightning to move/modify layers before quantization is finalized.

If FSDP sharding is required, load with `device_map=None` and handle placement explicitly inside `configure_model()`.

---

## 2. Handling 4-bit Quantization with Lightning Trainer

### The Core Compatibility Problem

Lightning's `BitsandbytesPrecision` plugin (`plugins=BitsandbytesPrecision(mode="nf4-dq")`) is designed for Lightning Fabric and replaces `nn.Linear` layers automatically. When used with the full `Trainer` and HuggingFace models that are loaded in 4-bit via `from_pretrained(..., quantization_config=...)`, the plugin and the HuggingFace quantization pipeline conflict. The result is parameter shape corruption.

**Verified issue:** GitHub issue #19732 in `Lightning-AI/pytorch-lightning` documents the error pattern: layers expecting shape `[hidden, hidden]` receive shapes like `[hidden*hidden*factor, 1]` because the plugin tries to re-quantize already-quantized weights.

### Recommended Approach: Manual BitsAndBytes Config, No Plugin

```python
# Correct — use HuggingFace quantization path directly
trainer = Trainer(
    accelerator="gpu",
    devices=1,
    precision="bf16-mixed",   # for non-quantized compute
    # Do NOT add BitsandbytesPrecision plugin here
)
```

Load the 4-bit model inside `__init__` with `BitsAndBytesConfig`, then pass the plain `Trainer`. Lightning handles the training loop; bitsandbytes handles quantization entirely within the model object.

### device_map Constraint

Setting `device_map="auto"` on `from_pretrained` instructs HuggingFace Accelerate to distribute layers across devices. This conflicts with Lightning's device management: Lightning owns device placement. The workaround:

- Single GPU: set `device_map={"": 0}` or `device_map=None` and let Lightning move the model.
- Multi-GPU: do not use `device_map="auto"` with Lightning. Use Lightning's built-in DDP strategy. Note that 4-bit quantized models have limited DDP support as of early 2026 — QLoRA multi-GPU training is better handled with FSDP or Lightning Fabric.

### Platform Constraint

bitsandbytes 4-bit quantization requires CUDA and Linux. The `BitsandbytesPrecision` plugin explicitly documents this. macOS and Windows (without WSL2) are unsupported.

---

## 3. Gradient Checkpointing Compatibility with QLoRA

### The Problem

When gradient checkpointing is enabled on a frozen base model with LoRA adapters on top, the input tensors to checkpointed layers may not have `requires_grad=True`. PyTorch's reentrant gradient checkpointing runs the forward pass under `torch.no_grad()`, so it cannot find the gradient path through the adapter layers. The symptom is a `RuntimeError: element 0 of tensors does not require grad`.

### The Fix: Two Complementary Steps

**Step 1 — Use non-reentrant checkpointing** (confirmed fix, PEFT issue #1142):

```python
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
```

Non-reentrant mode records the autograd graph during the forward pass, so gradients flow through even frozen-parameter regions into the adapter weights.

**Step 2 — Enable input gradient hooks** (belt-and-suspenders):

```python
model.enable_input_require_grads()
```

This installs a hook on the embedding layer that forces input tensors to carry gradients. It runs automatically inside `prepare_model_for_kbit_training` when `use_gradient_checkpointing=True` is passed. If you call `gradient_checkpointing_enable` separately (e.g. after PEFT wrapping), call `enable_input_require_grads()` manually.

### Canonical Combined Call

```python
base_model = prepare_model_for_kbit_training(
    base_model,
    use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)
```

This single call handles both steps correctly. Do not call `gradient_checkpointing_enable()` again after this.

### Lightning Trainer Interaction

Lightning's Trainer has its own `gradient_checkpointing` parameter. Do not use it when managing gradient checkpointing via PEFT — it will call `model.gradient_checkpointing_enable()` without `use_reentrant=False`, undoing the fix. Let PEFT/`prepare_model_for_kbit_training` own this setting.

---

## 4. Multi-Stage Curriculum Training in Lightning

### Two Patterns

#### Pattern A: Multiple Trainer.fit Calls (Recommended)

Create a new `Trainer` instance for each stage. Each stage gets independent `max_epochs`, callbacks, and optimizers.

```python
# Stage 1 — warm-up on easy examples
stage1_trainer = Trainer(max_epochs=5, callbacks=[ModelCheckpoint(dirpath="ckpts/stage1/")])
stage1_trainer.fit(model, datamodule=stage1_dm)
stage1_ckpt = stage1_trainer.checkpoint_callback.best_model_path

# Stage 2 — harder examples, different LR schedule
stage2_trainer = Trainer(max_epochs=10, callbacks=[ModelCheckpoint(dirpath="ckpts/stage2/")])
stage2_trainer.fit(model, datamodule=stage2_dm, ckpt_path=stage1_ckpt)
```

**Critical epoch counter issue:** When `ckpt_path` is provided to a subsequent `fit()` call, Lightning restores the full loop state including the epoch counter from the checkpoint. If stage 1 ran 5 epochs and stage 2's `max_epochs=10`, training will only run 5 more epochs (from epoch 5 to epoch 10), not 10 new epochs. To get N fresh epochs in stage 2:

- Set `max_epochs = stage1_epochs + stage2_epochs` on the stage 2 Trainer (e.g. `max_epochs=15` for 10 new epochs after a 5-epoch stage 1). This is the simplest workaround.
- Alternatively, reset the loop counter programmatically (fragile, touches Lightning internals):
  ```python
  stage2_trainer.fit_loop.epoch_progress.reset_on_epoch()
  ```

#### Pattern B: Callbacks for In-Loop Stage Transitions

For tighter integration (e.g. switching datasets mid-run based on a loss threshold), implement a custom callback:

```python
class CurriculumCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == self.stage1_epochs - 1:
            # Swap the dataloader
            trainer.datamodule.advance_stage()
            # Optionally swap optimizer config
            pl_module.current_stage += 1
```

This avoids checkpoint round-trips but couples stage logic into the training loop. Suitable for soft curriculum where stage boundaries do not require a full model reload or PEFT adapter swap.

### PEFT Adapter Swapping Between Stages

If each stage uses a different LoRA adapter (e.g. Stage 1 tunes only attention, Stage 2 adds FF layers), the cleanest approach is to save the Stage 1 adapter, reload the base model, and create a new `LightningModule` for Stage 2. Lightning does not natively support hot-swapping PEFT configs on an active model.

---

## 5. Checkpoint Saving and Loading with PEFT Adapters

### What Lightning Saves by Default

Lightning's `ModelCheckpoint` callback saves a `.ckpt` file containing the full `state_dict()` of the `LightningModule`. For a PEFT-wrapped model, `self.model.state_dict()` contains both the frozen base weights and the adapter weights, making the checkpoint enormous (same size as the full model). This defeats the point of PEFT.

### Strategy 1: Save Only Adapter Weights (Recommended)

Override `on_save_checkpoint` and `on_load_checkpoint` to save PEFT adapters separately, and strip the base model weights from Lightning's checkpoint:

```python
class QLoRALightningModule(LightningModule):

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        # Save only adapter weights to a parallel directory
        adapter_dir = Path(self.hparams.output_dir) / f"adapter_step_{self.global_step}"
        self.model.save_pretrained(str(adapter_dir))

        # Strip full state_dict from Lightning checkpoint to avoid bloat
        # Keep only non-base-model keys (e.g. optimizer state, hyperparams)
        checkpoint["state_dict"] = {
            k: v for k, v in checkpoint["state_dict"].items()
            if "base_model.model" not in k
        }
        # Store the adapter path so on_load_checkpoint can find it
        checkpoint["adapter_dir"] = str(adapter_dir)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        adapter_dir = checkpoint.get("adapter_dir")
        if adapter_dir:
            # Load the adapter weights back into the already-initialized model
            self.model.load_adapter(adapter_dir, adapter_name="default")
```

**Important:** `on_load_checkpoint` runs after `__init__` but before weights are restored from `state_dict`. Because you stripped `state_dict`, call `load_adapter` here to reload from the PEFT directory. The base model must already be loaded (4-bit) in `__init__` before this hook runs.

### Strategy 2: Use ModelCheckpoint with save_weights_only=False + Post-Processing

An alternative is to let Lightning save the full checkpoint, then call `model.save_pretrained(adapter_dir)` at the end of training. This simplifies the training loop at the cost of large checkpoint files during training.

### Loading for Inference

```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)
```

### Loading for Continued Training

```python
base_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
base_model = prepare_model_for_kbit_training(base_model, ...)
model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=True)
```

The `is_trainable=True` flag keeps adapter layers in training mode. Without it, `load_adapter` loads in inference mode with frozen weights.

### DType Gotcha

PEFT adapters saved in fp16 are sometimes loaded in fp32 (PEFT issue #2421). After loading, verify:

```python
print(model.model.layers[0].self_attn.q_proj.lora_A.default.weight.dtype)
```

If wrong, cast manually: `model = model.half()` or set `torch_dtype=torch.float16` in `from_pretrained`.

---

## 6. Exact Placement of prepare_model_for_kbit_training and gradient_checkpointing_enable

### Decision Tree

```
Load model with BitsAndBytesConfig (load_in_4bit=True)
         |
         v
prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,          # enables checkpointing
    gradient_checkpointing_kwargs={            # prevents requires_grad bug
        "use_reentrant": False
    }
)
         |
         v
get_peft_model(model, lora_config)
         |
         v
Assign to self.model in LightningModule.__init__
```

### What prepare_model_for_kbit_training Does Internally

1. Casts all normalization layers (LayerNorm, RMSNorm) to fp32 — prevents NaN loss from reduced-precision reductions.
2. Upcasts embedding and lm_head to fp32 for stable cross-entropy computation.
3. Freezes all model parameters (`requires_grad=False` everywhere).
4. Calls `model.enable_input_require_grads()` — installs embedding-level hook so gradients flow into unfrozen adapter layers.
5. Calls `model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=...)`.

### What NOT to Do

- Do not call `model.gradient_checkpointing_enable()` independently after this — it will override the `use_reentrant=False` setting.
- Do not pass `gradient_checkpointing=True` to Lightning's `Trainer` — it calls its own `enable_gradient_checkpointing()` path which bypasses the `use_reentrant=False` and PEFT input-require-grads hooks.
- Do not call `prepare_model_for_kbit_training` after `get_peft_model` — the freeze operation will also freeze the LoRA adapter parameters, leaving nothing to train.

---

## 7. Known Issues Summary

| Issue | Root Cause | Workaround |
|-------|-----------|------------|
| `BitsandbytesPrecision` plugin causes shape mismatch | Plugin re-quantizes already-quantized layers | Use `BitsAndBytesConfig` in `from_pretrained` only; no plugin |
| `device_map="auto"` conflicts with Lightning device management | Accelerate and Lightning both try to own device placement | Set `device_map=None` or `device_map={"": 0}` |
| Gradient checkpointing + LoRA: `requires_grad` is False | Reentrant checkpointing runs under `no_grad` | `use_reentrant=False` + `enable_input_require_grads()` |
| Lightning Trainer's `gradient_checkpointing=True` breaks PEFT setup | Trainer calls `enable_gradient_checkpointing` without PEFT-compatible kwargs | Do not use Trainer's gradient_checkpointing flag; manage via PEFT |
| Second `Trainer.fit` call ends immediately (epoch counter restored) | Lightning restores full loop state from checkpoint | Set `max_epochs = prior_epochs + new_epochs` on the new Trainer |
| PEFT checkpoint bloat in Lightning `.ckpt` files | `state_dict()` includes frozen base model weights | Override `on_save_checkpoint` to strip base weights; use `save_pretrained` |
| Adapters loaded in wrong dtype | PEFT saves fp16 adapters that load as fp32 | Verify dtype after load; cast if needed |

---

## 8. Confidence Assessment

| Area | Confidence | Source |
|------|-----------|--------|
| Mandatory init order (load → prepare → get_peft_model) | HIGH | PEFT official docs (huggingface.co/docs/peft/en/developer_guides/quantization) |
| BitsandbytesPrecision plugin conflict | HIGH | Lightning GitHub issue #19732 (direct error evidence) |
| device_map="auto" conflict | MEDIUM | GitHub issues + community pattern; no official Lightning statement |
| Gradient checkpointing use_reentrant=False fix | HIGH | PEFT GitHub issue #1142 confirmed by maintainers |
| Lightning Trainer gradient_checkpointing flag bypass | MEDIUM | Inferred from PEFT issue + Lightning docs; not explicitly stated in Lightning docs |
| on_save_checkpoint pattern for adapter-only saving | MEDIUM | Lightning checkpoint docs + PEFT save_pretrained pattern; no official combined example |
| Multi-stage epoch counter behavior | HIGH | Lightning GitHub issue #2823 + Trainer docs on ckpt_path |
| DType mismatch on load | MEDIUM | PEFT GitHub issue #2421 |

---

## 9. Sources

- PEFT Quantization Guide: https://huggingface.co/docs/peft/en/developer_guides/quantization
- PEFT Checkpoint Docs: https://github.com/huggingface/peft/blob/main/docs/source/developer_guides/checkpoint.md
- Lightning N-Bit Precision Docs: https://lightning.ai/docs/pytorch/stable/common/precision_intermediate.html
- Lightning Checkpoint Intermediate Docs: https://lightning.ai/docs/pytorch/stable/common/checkpointing_intermediate.html
- Lightning GitHub Issue #19732 (BitsandbytesPrecision shape mismatch): https://github.com/Lightning-AI/pytorch-lightning/issues/19732
- Lightning GitHub Issue #18295 (bitsandbytes usage): https://github.com/Lightning-AI/pytorch-lightning/issues/18295
- Lightning GitHub Issue #2823 (max_epochs reset): https://github.com/Lightning-AI/lightning/issues/2823
- Lightning Multi-Stage Training Discussion #19249: https://github.com/Lightning-AI/pytorch-lightning/discussions/19249
- PEFT Issue #1142 (gradient checkpointing + LoRA): https://github.com/huggingface/peft/issues/1142
- PEFT Issue #2421 (adapter dtype on load): https://github.com/huggingface/peft/issues/2421
- HuggingFace PEFT checkpoint forum: https://discuss.huggingface.co/t/correct-way-to-save-load-adapters-and-checkpoints-in-peft/77836
- Lightning 4-bit Quantization blog: https://lightning.ai/pages/blog/4-bit-quantization-with-lightning-fabric/
