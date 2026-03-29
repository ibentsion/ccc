from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str = "deepseek-ai/deepseek-coder-1.3b-base"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    learning_rate: float = 2e-4
    lr_scheduler: str = "constant"
    optimizer: str = "paged_adamw_32bit"
    gradient_checkpointing: bool = True
    max_seq_length: int = 2048
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 1
    max_steps: int = -1
    warmup_steps: int = 0
    output_dir: str = "outputs"
    clearml_project: str = "CodeComplete"
