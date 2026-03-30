import torch
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from training.config import TrainingConfig


class QLoRALightningModule(pl.LightningModule):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config

        compute_dtype = torch.float16 if config.bnb_4bit_compute_dtype == "float16" else torch.bfloat16

        # Step 1: Load base model with quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map={"": 0},  # NOT "auto" — avoids DDP conflicts with Lightning
            trust_remote_code=True,
        )

        # Step 2: Prepare model for k-bit training
        base_model = prepare_model_for_kbit_training(base_model)

        if config.gradient_checkpointing:
            base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # Step 3: Apply LoRA adapters
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias=config.lora_bias,
            target_modules=config.lora_target_modules,
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base_model, lora_config)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], batch["attention_mask"], batch["labels"])
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], batch["attention_mask"], batch["labels"])
        loss = outputs.loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        import bitsandbytes as bnb
        optimizer = bnb.optim.PagedAdamW32bit(self.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=0)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def save_adapter(self, path: str):
        self.model.save_pretrained(path)
