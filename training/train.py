"""Single-stage QLoRA training with ClearML experiment tracking."""
import os
from clearml import Task

# ClearML Task.init MUST be called BEFORE pytorch/lightning imports
# so TensorBoard auto-hook attaches correctly (EXP-01)


def main():
    from training.config import TrainingConfig
    from training.model import QLoRALightningModule
    from training.tokenizer import load_tokenizer, TokenizedCollator
    from data.config import PipelineConfig, StageConfig
    from data.datamodule import CurriculumDataModule
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger

    # --- Config ---
    train_config = TrainingConfig()
    pipeline_config = PipelineConfig(
        jsonl_path="python-codes-25k.jsonl",
        stages=[StageConfig(stage=1, min_lines=1, max_lines=1)],
    )

    # --- ClearML (EXP-01, EXP-02, EXP-03) ---
    task = Task.init(
        project_name=train_config.clearml_project,
        task_name=f"stage-{pipeline_config.stages[0].stage}-{pipeline_config.stages[0].min_lines}line",
        auto_connect_frameworks={"tensorboard": True},
    )

    # Log hyperparameters (EXP-03)
    task.connect({
        "model_name": train_config.model_name,
        "lora_r": train_config.lora_r,
        "lora_alpha": train_config.lora_alpha,
        "lora_dropout": train_config.lora_dropout,
        "lora_target_modules": str(train_config.lora_target_modules),
        "learning_rate": train_config.learning_rate,
        "lr_scheduler": train_config.lr_scheduler,
        "batch_size": train_config.per_device_batch_size,
        "gradient_accumulation_steps": train_config.gradient_accumulation_steps,
        "max_seq_length": train_config.max_seq_length,
        "bnb_4bit_quant_type": train_config.bnb_4bit_quant_type,
        "bnb_4bit_use_double_quant": train_config.bnb_4bit_use_double_quant,
        "stage": pipeline_config.stages[0].stage,
        "min_lines": pipeline_config.stages[0].min_lines,
        "max_lines": pipeline_config.stages[0].max_lines,
    })

    # --- Tokenizer ---
    tokenizer = load_tokenizer(train_config.model_name)
    collator = TokenizedCollator(tokenizer, max_seq_length=train_config.max_seq_length)

    # --- Data ---
    datamodule = CurriculumDataModule(
        config=pipeline_config,
        batch_size=train_config.per_device_batch_size,
        collate_fn=collator,
    )

    # --- Model ---
    model = QLoRALightningModule(train_config)

    # --- Logger (EXP-02: TensorBoard for ClearML auto-capture) ---
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(train_config.output_dir, "tb_logs"),
        name="qlora_training",
    )

    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs=train_config.num_train_epochs,
        max_steps=train_config.max_steps if train_config.max_steps > 0 else -1,
        accumulate_grad_batches=train_config.gradient_accumulation_steps,
        logger=tb_logger,
        log_every_n_steps=1,
        precision="16-mixed",  # matches bnb compute dtype
        accelerator="gpu",
        devices=1,
        enable_checkpointing=False,  # We save PEFT adapter manually, NOT via Lightning ModelCheckpoint
    )

    # --- Train ---
    datamodule.setup(stage="fit")
    train_dl = datamodule.train_dataloader(curriculum_stage=1)
    val_dl = datamodule.val_dataloader(curriculum_stage=1)
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # --- Save adapter (TRAIN-07) ---
    adapter_dir = os.path.join(train_config.output_dir, "adapter_stage_1")
    model.save_adapter(adapter_dir)
    print(f"Adapter saved to {adapter_dir}")

    # --- Upload to ClearML (EXP-04) ---
    task.upload_artifact(
        name="peft_adapter_stage_1",
        artifact_object=adapter_dir,
    )
    print("Adapter uploaded to ClearML")

    # --- Close ---
    task.close()


if __name__ == "__main__":
    main()
