# Project Research Summary

**Project:** CodeComplete — QLoRA Fine-Tuning with Curriculum Learning
**Domain:** LLM fine-tuning for Python code fill-in-the-middle (FIM) completion
**Researched:** 2026-03-29
**Confidence:** MEDIUM-HIGH

---

## Executive Summary

This project fine-tunes DeepSeek-Coder 1.3B or 6.7B (dense transformer decoders, NOT MoE) for Python missing-line completion using QLoRA (4-bit NF4 quantization) and a staged curriculum that progresses from 1-line gaps to N-line gaps. The canonical implementation stack is `transformers` + `peft` + `bitsandbytes` + PyTorch Lightning for the training loop, plus ClearML for experiment tracking. All four components are well-documented individually, but their combination surfaces several non-obvious conflicts that must be handled in the exact right order to avoid silent failures or OOM crashes.

The recommended curriculum approach is a single continuous Lightning training run with per-stage checkpoint saves, using a hybrid replay strategy (75% current-stage samples + 25% replay from prior stages) and a constant learning rate through stages — NOT cosine decay that reaches near-zero before the final stage. Research shows that curriculum learning benefits are modest for token-level completion metrics, but meaningful for harder multi-line completions (Stage 3+). The primary evaluation signal should be Exact Match per N-line bucket, supplemented by Pass@1 execution tests at stage boundaries, not CodeBLEU (which correlates poorly with functional correctness).

The top risks are: (1) the `BitsandbytesPrecision` Lightning plugin silently corrupting quantized model weights — avoid it entirely and use `BitsAndBytesConfig` directly; (2) cosine LR decay reaching near-zero before hard examples appear in later curriculum stages — use constant LR through staging; (3) catastrophic forgetting of easy patterns when advancing stages without a replay buffer; (4) PEFT adapter weights not being auto-logged by ClearML because `save_pretrained()` bypasses `torch.save()` — upload the adapter directory manually via `task.upload_artifact()`.

---

## Key Findings

### Recommended Stack

DeepSeek-Coder 1.3B and 6.7B are both standard dense decoder-only models (verified against arxiv 2401.14196). The architecture supports 16K context natively, but training should cap `max_seq_length` at the realistic sample length (512–2048 tokens) to avoid padding waste and memory spikes. Both models require `trust_remote_code=True` at load time.

**Core technologies:**
- `transformers` + `BitsAndBytesConfig`: load model in 4-bit NF4 with double quantization — saves ~0.37 bits/parameter with no observed quality loss
- `peft` (`prepare_model_for_kbit_training` + `LoraConfig` + `get_peft_model`): the three-step initialization order is mandatory and order-sensitive
- `bitsandbytes` 4-bit QLoRA: NF4, NOT FP4; `bfloat16` compute dtype on Ampere+ GPUs
- PyTorch Lightning `Trainer`: training loop orchestration; do NOT use Lightning's `BitsandbytesPrecision` plugin or `gradient_checkpointing=True` Trainer flag
- ClearML `Task.init()`: experiment tracking via TensorBoard auto-hooking; no native `ClearMLLogger` class exists in Lightning
- `paged_adamw_32bit`: required optimizer for stable QLoRA training on consumer hardware

**LoRA configuration (starting point for < 10K samples):**
- `r=16`, `lora_alpha=32` (2x ratio is the established standard)
- `target_modules`: all 7 projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- `lora_dropout=0.05`, `bias="none"`

**Hardware requirements:**
- DeepSeek-Coder 1.3B QLoRA training: 4–6 GB VRAM minimum; RTX 3060 (12 GB) comfortable
- DeepSeek-Coder 6.7B QLoRA training: 8–12 GB VRAM; 12 GB practical minimum; 24 GB (RTX 3090/4090) ideal

### Expected Features / Implementation Scope

**Must implement (core pipeline):**
- Dataset loader for JSONL with instruction/input/output format
- Gap creation pipeline (masking N consecutive lines, generating FIM-format training pairs using DeepSeek-Coder's PSM tokens: `<｜fim▁begin｜>`, `<｜fim▁hole｜>`, `<｜fim▁end｜>`)
- QLoRA-wrapped LightningModule with correct 3-step init sequence
- Per-stage DataModule or DataLoader swap mechanism
- Hybrid replay buffer (25% prior-stage hard examples per batch)
- Per-stage checkpoint save via `on_save_checkpoint` override (adapter only, not full model)
- ClearML per-stage task with adapter artifact upload

**Should implement (quality signals):**
- Exact Match (EM) evaluation per N-line bucket — primary offline metric
- Edit Similarity metric — essential for multi-line stages where EM is too strict
- Pass@1 execution test on 200-sample subset at each stage boundary — the only metric that measures functional correctness
- Loss-plateau detection per stage (flag to logs if < 5% relative improvement before stage ends)
- Cyclomatic complexity annotation on samples for intra-stage ordering (2 lines of `radon`)

**Defer to later:**
- Horizon-Length Prediction auxiliary head (arxiv 2410.03103) — up to 24% relative EM improvement; compatible with QLoRA; medium implementation cost; good v2 candidate
- HumanEval-Infilling benchmark integration — run on final model only
- Multi-GPU / FSDP training (4-bit QLoRA has limited DDP support as of early 2026)
- Model-loss-based difficulty scoring (computationally expensive at dataset scale)
- DPO alignment after SFT (EMNLP 2025 paper approach)

### Architecture Approach

The system is structured as a single continuous Lightning training run that advances through curriculum stages by swapping the active DataLoader after a fixed number of steps. A custom `CurriculumCallback` handles stage transitions, checkpoint saves, and ClearML artifact uploads. Each stage is logged as a separate ClearML Task (Pattern A) for independent reproducibility and clean per-stage metrics. The PEFT adapter is saved to disk after each stage using `model.save_pretrained()` and manually uploaded; the Lightning `.ckpt` file is stripped of frozen base-model weights to stay compact.

**Major components:**

1. **GapDataset / CurriculumDataModule** — reads JSONL, creates FIM-format training pairs with N-line gaps, stratifies by difficulty (N lines + cyclomatic complexity), exposes stage-specific DataLoaders with replay buffer mixing
2. **QLoRALightningModule** — wraps the quantized DeepSeek-Coder model with PEFT adapters; `training_step` computes causal LM loss with prompt masking; `configure_optimizers` returns `paged_adamw_32bit` over trainable params only
3. **CurriculumCallback** — monitors step counts, triggers stage transitions, saves per-stage adapters, uploads to ClearML
4. **EvalCallback** — runs EM + Edit Similarity on val set every N steps; runs Pass@1 on 200-sample subset at stage boundaries
5. **ClearML integration layer** — one `Task.init()` per stage, `TensorBoardLogger` for auto-scalar capture, manual `task.upload_artifact()` for PEFT adapter directories

### Critical Pitfalls

1. **Never use Lightning's `BitsandbytesPrecision` plugin** — It re-quantizes already-quantized layers and corrupts weight shapes (confirmed Lightning GitHub issue #19732). Load with `BitsAndBytesConfig` inside `__init__` and use a plain `Trainer` with `precision="bf16-mixed"`.

2. **Mandatory initialization order for QLoRA** — `from_pretrained(quantization_config=...)` → `prepare_model_for_kbit_training(use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})` → `get_peft_model(...)`. Wrong order breaks gradient flow or freezes adapter weights. Do NOT call Lightning's `gradient_checkpointing=True` Trainer flag — it overrides `use_reentrant=False` and breaks the PEFT gradient hook.

3. **`device_map="auto"` conflicts with Lightning device management** — Set `device_map=None` or `device_map={"": 0}`. Accelerate and Lightning both try to own device placement; `device_map="auto"` wins unpredictably.

4. **Cosine LR decay kills curriculum learning** — Cosine decay that reaches near-zero before the final stage means hard examples get near-zero gradient signal. Use constant LR (2e-4) through all stages; apply cosine decay only within the final stage, or use cyclic LR with mild resets at stage boundaries.

5. **PEFT adapter not auto-logged by ClearML** — `model.save_pretrained()` does not call `torch.save()` in a way ClearML reliably hooks. Always call `task.upload_artifact("peft-adapter-stage-N", adapter_dir)` explicitly after saving.

6. **Lightning epoch counter persists across multi-stage `Trainer.fit()` calls** — When passing `ckpt_path` to a second `Trainer.fit()`, Lightning restores the epoch counter from the checkpoint. Stage 2 `Trainer(max_epochs=10)` only runs 5 new epochs if Stage 1 ran 5. Fix: set `max_epochs = sum_of_all_prior_epochs + new_stage_epochs` on each subsequent Trainer.

7. **Wrong EOS token causes broken inference** — DeepSeek-Coder has two EOS tokens: `32014` (`<｜end▁of▁sentence｜>`) for base/completion mode, `32021` (`<|EOT|>`) for instruction mode. Using the instruction variant at inference time causes non-terminating outputs. Set `eos_token_id=32014` at inference for completion tasks.

8. **FIM format must use PSM, not SPM** — Use `<｜fim▁begin｜>{prefix}<｜fim▁hole｜>{suffix}<｜fim▁end｜>` consistently. Mixing PSM and SPM within a training run degrades FIM quality. Do NOT include function signatures or class definition lines in the gap (structural understanding, not missing-line completion).

9. **Checkpoint bloat in Lightning `.ckpt` files** — `self.model.state_dict()` for a PEFT model includes frozen base-model weights, creating multi-GB checkpoint files. Override `on_save_checkpoint` to strip base-model keys and save only the adapter via `model.save_pretrained()`.

10. **Task.init() must precede all framework imports** — ClearML installs hooks at `Task.init()` time. If `torch` or `pytorch_lightning` is imported first, scalar auto-logging may silently fail. Also: add `TensorBoardLogger` to the Trainer explicitly; ClearML captures scalars by hooking TensorBoard, not Lightning directly.

---

## Implications for Roadmap

### Phase 1: Data Pipeline and FIM Format Validation

**Rationale:** Everything downstream depends on correctly formatted training data. The PSM FIM format, gap creation, and dataset schema need to be locked before any training code is written. This phase has no GPU dependency.

**Delivers:** Validated JSONL dataset with FIM-formatted samples at all N-line gap levels; difficulty annotations (line count + cyclomatic complexity via `radon`); train/val/test split; filtered dataset (removed trivially easy gaps: `pass`, blank lines, closing brackets; removed out-of-scope gaps: function signatures, class definitions)

**Addresses:** FIM format consistency pitfall; trivially easy/hard gap filtering; dataset versioning in ClearML

**Key decision point:** Confirm whether DeepSeek-Coder base or instruct variant will be fine-tuned. Instruct variant is more common in tutorials but base model FIM capability may be stronger for this task.

### Phase 2: QLoRA Training Scaffold (Single Stage, No Curriculum)

**Rationale:** Validate the full Lightning + PEFT + bitsandbytes + ClearML integration stack on a single-stage run before adding curriculum complexity. Most integration bugs (device_map, gradient checkpointing, plugin conflicts, checkpoint bloat) are best caught here on a small subset.

**Delivers:** Working `QLoRALightningModule` with correct 3-step init; `paged_adamw_32bit` optimizer over adapter params only; per-step loss logging via TensorBoard auto-capture in ClearML; adapter-only checkpoint save pattern; first trained adapter on Stage 1 (1-line gaps) data

**Avoids:** BitsandbytesPrecision plugin conflict; device_map conflict; gradient checkpointing requires_grad bug; PEFT checkpoint bloat

**Research flag:** Validate `use_reentrant=False` fix is working by checking that training loss actually decreases. If it does not, the gradient hook is broken.

### Phase 3: Curriculum Training Loop

**Rationale:** Add curriculum stage transitions, hybrid replay buffer, and per-stage evaluation on top of the validated Phase 2 scaffold.

**Delivers:** `CurriculumCallback` with step-count-based stage transitions; hybrid replay (75/25 split); constant LR across stages (cosine decay within final stage only); per-stage adapter artifacts in ClearML; EM + Edit Similarity per stage; Pass@1 gate at stage boundaries

**Avoids:** Catastrophic forgetting (replay buffer); cosine LR decay undermining later stages; epoch counter reset bug in multi-Trainer pattern

**Implementation note:** Use the callback-based in-loop stage transition (Pattern B from lightning-qlora.md) rather than multiple `Trainer.fit()` calls. This avoids the epoch counter accumulation bug entirely and keeps curriculum logic decoupled from the Trainer.

### Phase 4: Evaluation and Iteration

**Rationale:** Systematic per-stage and final model evaluation; hyperparameter tuning informed by ClearML experiment comparison.

**Delivers:** EM per N-line bucket on held-out test set; Pass@1 on 200-sample execution subset; comparison of 1.3B vs 6.7B adapter quality; documentation of which curriculum stage contributed what improvement; decision on whether to add Horizon-Length Prediction auxiliary head (v2 candidate)

**Key risk:** Exact Match will underestimate quality for multi-line stages. Report Edit Similarity alongside EM at Stage 3+. Do not conclude curriculum is not working from EM alone.

### Phase Ordering Rationale

- Data pipeline first because FIM format bugs are invisible until you try to load the model and get misshapen input sequences.
- Single-stage scaffold second because the Lightning + bitsandbytes + ClearML integration has 5+ independently-breakable points; validate each in isolation before layering curriculum logic.
- Curriculum loop third because it depends on a stable training scaffold.
- Evaluation last because meaningful evaluation metrics (Pass@1) require a sandbox and a model that has completed at least Stage 2 training.

### Research Flags

**Phase 2 needs watchful validation (not additional research):**
- Verify `use_reentrant=False` gradient flow is working (check training loss decreases in first 50 steps)
- Verify adapter dtype after checkpoint load (PEFT issue #2421 — fp16 saves sometimes load as fp32)
- Verify ClearML scalar capture is working by checking TensorBoard hook attached before `Task.init()` call placement

**Phase 3 has limited empirical grounding for exact step counts:**
- The curriculum schedule in curriculum-learning.md (Section 9) is a starting point; actual steps per stage need tuning against the expanded dataset size after gap creation. Rule of thumb: each stage should see 2–3 passes over its primary difficulty subset.
- The 75/25 replay ratio is from ACL 2024; optimal value for this specific dataset size is unknown.

**Phases with standard patterns (no additional research needed):**
- Phase 1: dataset construction and FIM format are fully specified
- Phase 4: evaluation metrics and their trade-offs are well-documented

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| QLoRA config (LoRA r/alpha, NF4, double quant) | HIGH | QLoRA paper + Lightning AI empirical study + Unsloth docs agree |
| Lightning + PEFT integration order and pitfalls | HIGH | Verified via Lightning GitHub issues (#19732, #2823) and PEFT issue #1142 |
| ClearML integration pattern | HIGH | Official ClearML docs; no-native-logger confirmed via Lightning issue tracker |
| Curriculum scheduling strategy | MEDIUM | ACL 2024 and Dec 2024 FIM paper are production-validated; optimal step counts and replay ratios are dataset-specific |
| Curriculum benefit magnitude | MEDIUM | Research shows modest token-level gains; stronger gains on hard subsets. Controlled study for this exact task (missing-line Python) does not exist |
| Optimal r for this dataset size | MEDIUM | r=16 is the consensus starting point for < 10K samples; no published study for narrow code completion specifically |
| 1.3B vs 6.7B quality tradeoff | LOW | No published comparison for narrow domain fine-tuning at this scale; 6.7B is generally better but 1.3B may suffice |

**Overall confidence:** MEDIUM-HIGH — the implementation patterns are well-grounded; the curriculum scheduling parameters need empirical validation against the actual expanded dataset.

### Gaps to Address

- **Optimal LoRA rank for code completion on < 10K samples:** r=16 is the safe start; run a 2-point sweep (r=8, r=16) in Phase 4 using ClearML experiment comparison. Cost: two additional training runs.

- **FIM fine-tuning for base vs instruct model variant:** DeepSeek-Coder GitHub issue #68 is unresolved. If using the instruct variant, the existing instruction-following behavior may interfere with FIM generation. Validate on a small sample before committing to full training.

- **Expanded dataset size after gap creation:** python-codes-25k.jsonl has ~25K source samples. After creating all valid N-line gaps (1 through ~10) per sample, the effective dataset size is much larger. Actual stage step counts in the curriculum schedule must be recalculated based on this expanded size before Phase 3.

- **Pass@1 sandbox setup:** Execution-based evaluation requires a Python sandbox. This is not a training concern but must be set up before Phase 4. Consider a lightweight containerized runner using `subprocess` with timeout or a dedicated evaluation service.

---

## Sources

### Primary (HIGH confidence)
- [DeepSeek-Coder paper (arxiv 2401.14196)](https://arxiv.org/pdf/2401.14196) — architecture, FIM tokens, context length
- [QLoRA paper (Dettmers et al., 2023, arxiv 2305.14314)](https://arxiv.org/abs/2305.14314) — NF4 quantization, double quant
- [Lightning AI LoRA Insights](https://lightning.ai/pages/community/lora-insights/) — empirical rank/alpha study
- [Sebastian Raschka LoRA tips](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) — alpha/rank ratio
- [PEFT official quantization guide](https://huggingface.co/docs/peft/en/developer_guides/quantization) — mandatory init order
- [PEFT issue #1142](https://github.com/huggingface/peft/issues/1142) — use_reentrant=False fix
- [Lightning GitHub issue #19732](https://github.com/Lightning-AI/pytorch-lightning/issues/19732) — BitsandbytesPrecision shape corruption
- [Lightning GitHub issue #2823](https://github.com/Lightning-AI/lightning/issues/2823) — epoch counter behavior
- [ClearML Task SDK docs](https://clear.ml/docs/latest/docs/clearml_sdk/task_sdk/) — Task.init() parameters and placement
- [ClearML Dataset SDK docs](https://clear.ml/docs/latest/docs/clearml_data/clearml_data_sdk/) — dataset versioning
- [Improving FIM via Context & Curriculum Learning (arxiv 2412.16589)](https://arxiv.org/abs/2412.16589) — production-validated CL for FIM; DS-Coder 1B/7B gains
- [Curriculum Learning for Small Code LMs (ACL SRW 2024, arxiv 2407.10194)](https://arxiv.org/abs/2407.10194) — hybrid replay strategy; CL gains for completion vs execution tasks
- [Horizon-Length Prediction (arxiv 2410.03103)](https://arxiv.org/abs/2410.03103) — auxiliary head technique; 24% relative EM improvement

### Secondary (MEDIUM confidence)
- [DeepSeek-Coder official fine-tuning script](https://github.com/deepseek-ai/DeepSeek-Coder/blob/main/finetune/finetune_deepseekcoder.py) — prompt format, padding side
- [Does the Definition of Difficulty Matter? (arxiv 2411.00973)](https://arxiv.org/abs/2411.00973) — difficulty metric interchangeability
- [CL for LLM Pretraining: Learning Dynamics (arxiv 2601.21698)](https://arxiv.org/pdf/2601.21698) — cosine LR + CL interaction
- [SAFIM benchmark (arxiv 2403.04814)](https://arxiv.org/html/2403.04814v1) — FIM difficulty taxonomy
- [DeepSpeed CL tutorial](https://www.deepspeed.ai/tutorials/curriculum-learning/) — sequence-length pacing functions
- [Unsloth LoRA hyperparameters guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide) — dropout on small datasets
- [PEFT issue #2421](https://github.com/huggingface/peft/issues/2421) — adapter dtype mismatch on load

### Tertiary (LOW confidence)
- [DeepSeek-Coder issue #68 (FIM format)](https://github.com/deepseek-ai/DeepSeek-Coder/issues/68) — FIM fine-tuning on instruct variant; unresolved
- [Fine-Tuning DeepSeek Coder 6.7B (practitioner blog, Mar 2025)](https://kennycason.com/posts/2025-03-10-finetuning-deepseek-coder-6.7b.html) — corroborating practitioner patterns

---

*Research completed: 2026-03-29*
*Ready for roadmap: yes*
