# QLoRA Fine-Tuning: DeepSeek-Coder 1.3B and 6.7B

**Domain:** Code completion fine-tuning with QLoRA
**Researched:** 2026-03-29
**Overall confidence:** MEDIUM-HIGH (most findings corroborated by multiple sources; some specifics verified against official repo)

---

## 1. DeepSeek-Coder Architecture Relevant to LoRA

DeepSeek-Coder 1.3B and 6.7B are standard dense transformer decoder models (not MoE). They use:

- **Attention projections:** `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **FFN/MLP projections (SwiGLU-style):** `gate_proj`, `up_proj`, `down_proj`
- Vocabulary size: 32,000 tokens with BPE tokenizer
- Native context length: 16K tokens (supports project-level code completion)
- Trained on 2T tokens: 87% code, 13% natural language

DeepSeek-Coder V2 (a different, later model) uses MLA and MoE. The 1.3B and 6.7B are standard dense models — the recommendations here do not apply to V2.

**Confidence:** HIGH — verified against official DeepSeek-Coder paper (arxiv 2401.14196) and GitHub repo.

---

## 2. Recommended LoRA Configuration

### Target Modules

**Use all linear layers.** Both the QLoRA paper empirical results and Unsloth's benchmarks confirm that targeting all projection layers outperforms attention-only targeting.

```python
target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
```

If you are memory-constrained and must drop layers, drop `o_proj` first, then the attention projections, keeping the FFN layers (`gate_proj`, `up_proj`, `down_proj`). This is a fallback — all-linear is preferred.

**Confidence:** HIGH — multiple sources including Lightning AI experiments and Unsloth documentation agree.

### Rank (r)

| Dataset Size | Recommended r | Rationale |
|---|---|---|
| < 1K samples | 8–16 | Low diversity, risk of overfit |
| 1K–10K samples | 16–32 | Balanced; good starting point |
| > 10K samples | 32–64 | More capacity warranted |

For most code completion fine-tuning on < 10K samples, **start with r=16**.

Lightning AI experiments found r=256 with alpha=512 optimal for an instruction-following 50K dataset on an A100 — this is not applicable to small coding datasets where overfitting risk dominates.

**Confidence:** MEDIUM — range is based on cross-source consensus; optimal value requires experimentation per dataset.

### Alpha (lora_alpha)

Set **alpha = 2 × r** as the default. This gives a scaling factor of 2.0.

```python
lora_alpha = 32   # when r=16
```

This rule is widely endorsed (Sebastian Raschka, Unsloth docs, Lightning AI experiments). The ratio alpha/r determines the effective learning rate of the adapter. Values below 1x or above 4x tend to degrade performance.

**Confidence:** HIGH — consistent across multiple authoritative sources.

### Dropout (lora_dropout)

```python
lora_dropout = 0.05   # small dataset, prevent overfitting
```

Use `0.05` for datasets under 10K samples. Unsloth notes that `0.0` is computationally optimized, but a small dropout provides regularization that is beneficial on small datasets. Use `0.1` if you observe training loss collapsing below validation loss early.

**Confidence:** MEDIUM — based on practitioner consensus, not a controlled study for code tasks.

### Bias

```python
bias = "none"
```

Universal recommendation. Training bias terms adds negligible benefit and increases complexity.

---

## 3. bitsandbytes 4-Bit Quantization Config

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NOT fp4
    bnb_4bit_compute_dtype=torch.bfloat16,   # bf16 if GPU supports it, else float16
    bnb_4bit_use_double_quant=True,     # nested quantization — saves ~0.37 bits/param
)
```

### NF4 vs FP4

**Use NF4.** It is information-theoretically optimal for weights that follow a zero-centered normal distribution (which pre-trained transformer weights do). FP4 has no fixed format and consistently performs worse than NF4 for fine-tuning. This is the original QLoRA paper recommendation, confirmed by all subsequent sources.

**Confidence:** HIGH — established in the QLoRA paper (Dettmers et al., 2023) and verified across multiple 2024–2025 sources.

### Double Quantization

**Enable it** (`bnb_4bit_use_double_quant=True`). It quantizes the quantization constants themselves, saving approximately 0.37 bits/parameter (~3 GB on a 65B model, proportionally less on 6.7B but still meaningful). There is no observed quality loss from enabling it.

### Compute Dtype

Use `bfloat16` on Ampere+ GPUs (RTX 30xx, A100, RTX 40xx). Use `float16` on older hardware. The compute dtype is used for the forward/backward pass arithmetic — it does not affect the storage format.

**Confidence:** HIGH.

---

## 4. Training Hyperparameters for Small Datasets (< 10K samples)

### Learning Rate

```python
learning_rate = 2e-4
```

Start at `2e-4`. If loss does not decrease within the first 50 steps, lower to `1e-4`. Learning rates above `5e-4` cause adapter overshooting and training instability.

For very small datasets (< 1K samples), start at `1e-4` to be conservative.

### LR Scheduler

```python
lr_scheduler_type = "cosine"
```

Cosine annealing is the standard recommendation. Linear decay is an acceptable alternative but cosine is preferred in most practitioner reports.

### Warmup

```python
warmup_ratio = 0.05   # 5% of total steps
# OR
warmup_steps = 10     # absolute minimum; use ratio for variable-length runs
```

Use `warmup_ratio=0.05` to `0.10` (5–10% of total steps). For a 100-step training run on a tiny dataset, this means 5–10 warmup steps. Warmup prevents large gradient updates at initialization when the adapter weights are random.

### Batch Size and Gradient Accumulation

```python
per_device_train_batch_size = 2
gradient_accumulation_steps = 8   # effective batch = 16
```

Target an **effective batch size of 16**. Larger effective batches stabilize training but require more memory. Use gradient accumulation to achieve this without OOM. For 8 GB VRAM:

- DeepSeek-Coder 1.3B with QLoRA: batch size 4, accumulation 4 is feasible
- DeepSeek-Coder 6.7B with QLoRA: batch size 1–2, accumulation 8 is required

### Number of Epochs

| Dataset Size | Recommended Epochs |
|---|---|
| < 500 samples | 5–10 (risk of memorization; monitor eval loss) |
| 500–2K samples | 3–5 |
| 2K–10K samples | 1–3 |

A key finding from Lightning AI's experiments: multi-epoch training on static instruction datasets can cause the model to "actively unlearn" general abilities. For code completion, the risk is less severe than instruction tuning, but still present. Monitor validation loss — stop if it diverges upward from training loss.

### Optimizer

```python
optim = "paged_adamw_32bit"
```

`paged_adamw_32bit` is the standard for QLoRA. It uses NVIDIA's unified memory (paging) to prevent OOM spikes during gradient checkpointing. Regular AdamW works but will OOM more frequently on constrained hardware.

### Gradient Checkpointing

```python
gradient_checkpointing = True
```

Required for 6.7B on consumer GPUs. Reduces memory by recomputing activations during backward pass rather than storing them, at the cost of ~20% training speed. On 8 GB VRAM it is non-negotiable for 6.7B.

### Max Gradient Norm

```python
max_grad_norm = 0.3
```

Standard for QLoRA. Prevents gradient explosion during early training steps.

### Sequence Length

DeepSeek-Coder supports up to 16K context. For training, keep `max_seq_length` at the longest realistic sample in your dataset, not 16K, to avoid wasted padding and memory spikes. A practical value for code completion datasets is 512–2048 tokens.

---

## 5. Full Reference Configuration

```python
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig, TrainingArguments
import torch

# 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# LoRA adapter
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Training arguments (adjust per dataset size)
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,          # effective batch = 16
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    max_grad_norm=0.3,
    fp16=False,
    bf16=True,                              # requires Ampere+; else set fp16=True
    logging_steps=10,
    save_strategy="epoch",
)
```

---

## 6. Common Pitfalls

### Critical

**1. Wrong EOS token for code completion vs. instruction mode**

DeepSeek-Coder uses different EOS tokens depending on the task:
- Code completion (base model mode): `eos_token_id = 32014` (`<｜end▁of▁sentence｜>`)
- Instruction following: `eos_token_id = 32021` (`<|EOT|>`)

Using the wrong EOS token causes the model to not terminate outputs correctly during inference. If fine-tuning the instruct variant for code completion, set `eos_token_id=32014` at inference time.

**2. Inconsistent prompt format between training and inference**

The exact format used during fine-tuning must be reproduced at inference time, including system prompts, separators, and special tokens. Even minor formatting differences (e.g., trailing newline, different separator) can severely degrade output quality. Use `tokenizer.apply_chat_template()` if available, or document your exact format.

**3. Sequence length mismatch / padding on the wrong side**

DeepSeek-Coder tokenizer right-pads sequences. If you pad on the left or truncate incorrectly, the model sees malformed inputs. The official fine-tuning script uses right padding with `DataCollatorForSeq2Seq`. For causal LM fine-tuning, loss should be masked on padding tokens and prompt tokens (only compute loss on the target/completion portion).

**4. FIM format if fine-tuning for fill-in-the-middle**

If your code completion task requires FIM (fill-in-the-middle), use the PSM (Prefix-Suffix-Middle) format:
```
<｜fim▁begin｜>{prefix}<｜fim▁hole｜>{suffix}<｜fim▁end｜>
```
Add the BOS token before FIM prompts for base models. The instruction model variants were not specifically fine-tuned on FIM tasks, so fine-tuning on FIM examples is especially important if this capability is needed. GitHub issue #68 on the DeepSeek-Coder repo remains open with no official guidance — treat FIM fine-tuning as experimental.

**5. Gradient checkpointing + long sequences = OOM spikes**

With gradient checkpointing enabled, processing mini-batches with very long sequences (e.g., > 2048 tokens) causes memory spikes proportional to sequence length, not average memory. Paged optimizers (`paged_adamw_32bit`) mitigate this by using unified memory. Keep `max_seq_length` realistic and filter out outlier-length samples from your dataset.

### Moderate

**6. Multi-epoch overfitting on small datasets**

Training for too many epochs on a small static dataset causes the model to memorize examples and degrade on general code tasks. On datasets under 2K samples, monitor validation loss after every epoch and stop when it stops decreasing. The Lightning AI experiments found that doubling the number of training steps on a 50K dataset actively degraded performance — this effect is amplified on smaller datasets.

**7. Alpha/rank ratio misconfiguration**

Setting alpha equal to rank (ratio = 1.0) is conservative and works but leaves capacity on the table. Setting alpha much higher than 2x rank can destabilize training. The safe default is `alpha = 2 * r`. Do not blindly copy configurations from tutorials without checking this ratio — many tutorials use `r=16, alpha=16` (ratio = 1.0), which is suboptimal.

**8. Missing `trust_remote_code=True`**

DeepSeek-Coder uses custom model code. Loading without `trust_remote_code=True` raises an error. Always include this flag:
```python
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    trust_remote_code=True,
    quantization_config=bnb_config,
)
```

**9. Shape mismatch when saving/loading fine-tuned weights**

A reported issue (#137 in the DeepSeek-Coder repo) shows `embed_tokens.weight` shape mismatch on load after full fine-tuning. With QLoRA (adapter only), this is less likely, but always save using `trainer.save_model()` and `tokenizer.save_pretrained()` together, then load with `ignore_mismatched_sizes=True` if needed.

**10. Catastrophic forgetting of general coding ability**

Fine-tuning on a narrow code completion style (e.g., only Python functions) can degrade performance on other languages or tasks the base model knew. If general capability must be preserved, mix in a small amount of diverse code examples from the original training distribution. This is especially relevant for the 1.3B model which has less capacity to absorb narrow fine-tuning without forgetting.

### Minor

**11. Using `load_in_8bit=True` instead of 4-bit**

Some older tutorials use `load_in_8bit`. For consumer GPU fine-tuning, 4-bit QLoRA is the current recommendation. 8-bit inference has higher quality than 4-bit, but for fine-tuning the memory savings of 4-bit + double quantization are generally preferable.

**12. Not enabling `gradient_checkpointing` after calling `get_peft_model`**

Call `model.enable_input_require_grads()` before or after applying PEFT, and enable gradient checkpointing via `TrainingArguments`. Some PEFT versions require explicit `model.gradient_checkpointing_enable()` to be called before `get_peft_model()`.

---

## 7. Hardware Reality Check

| Model | VRAM (4-bit inference) | VRAM (QLoRA training) | GPU Recommendation |
|---|---|---|---|
| DeepSeek-Coder 1.3B | ~1.5 GB | ~4–6 GB | RTX 3060 (12 GB) or better |
| DeepSeek-Coder 6.7B | ~3.5 GB | ~8–12 GB | RTX 3080/4070 (10–12 GB), RTX 3090/4090 (24 GB) ideal |

Training the 6.7B on exactly 8 GB VRAM is technically possible but leaves no margin — batch size must be 1 and sequence length must be short. 12 GB VRAM is the practical minimum for comfortable training.

---

## 8. Verified Source List

- [Fine-Tuning DeepSeek Coder 6.7B using QLoRA (March 2025)](https://kennycason.com/posts/2025-03-10-finetuning-deepseek-coder-6.7b.html) — MEDIUM confidence, practitioner blog
- [DeepSeek-Coder GitHub — Official Fine-tuning Script](https://github.com/deepseek-ai/DeepSeek-Coder/blob/main/finetune/finetune_deepseekcoder.py) — HIGH confidence, official source
- [DeepSeek-Coder-V2 LoRA PR #44](https://github.com/deepseek-ai/DeepSeek-Coder-V2/pull/44/files) — HIGH confidence, official repo
- [Finetuning LLMs with LoRA and QLoRA: Insights from Hundreds of Experiments — Lightning AI](https://lightning.ai/pages/community/lora-insights/) — HIGH confidence, empirical study
- [Practical Tips for Finetuning LLMs Using LoRA — Sebastian Raschka](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) — HIGH confidence, rigorous analysis
- [LoRA Hyperparameters Guide — Unsloth Documentation](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide) — MEDIUM confidence, framework docs
- [Making LLMs more accessible with bitsandbytes and QLoRA — Hugging Face Blog](https://huggingface.co/blog/4bit-transformers-bitsandbytes) — HIGH confidence, official HF source
- [QLoRA Paper (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314) — HIGH confidence, primary source
- [Fine-tuning DeepSeek R1 on a Custom Instructions Dataset — Firecrawl Blog](https://www.firecrawl.dev/blog/fine-tuning-deepseek) — LOW-MEDIUM confidence, practitioner blog
- [DeepSeek-Coder Issues #68 (FIM format)](https://github.com/deepseek-ai/DeepSeek-Coder/issues/68) — MEDIUM confidence (open issue, no resolution)
- [DeepSeek-Coder Issues #71 (FIM token behavior)](https://github.com/deepseek-ai/DeepSeek-Coder/issues/71) — MEDIUM confidence (community resolution)
- [DeepSeek-Coder: When the Large Language Model Meets Programming (arxiv 2401.14196)](https://arxiv.org/pdf/2401.14196) — HIGH confidence, official paper

---

## 9. Open Questions / Areas Needing Validation

1. **Optimal r for code completion specifically** — The Lightning AI r=256 finding is for instruction tuning on 50K samples; there is no controlled study for code completion on < 10K samples. r=16 is a safe start but may not be optimal.

2. **FIM fine-tuning data format** — No official guidance exists for structuring training data to improve FIM capability. GitHub issue #68 is unresolved. Treat any FIM fine-tuning as experimental.

3. **1.3B vs 6.7B quality threshold for domain-specific code** — No published study compares the two specifically for narrow domain fine-tuning. The 6.7B is generally better, but the 1.3B may be sufficient for single-language, limited-vocabulary completion tasks.

4. **bf16 vs fp16 for compute dtype** — bf16 is theoretically better (larger dynamic range, less overflow risk) and consistently recommended, but the practical quality difference for 4-bit QLoRA is not well-documented for DeepSeek-Coder specifically.
