"""Multi-stage curriculum training with per-stage ClearML tracking."""
import os
from clearml import Task


def main():
    from training.config import TrainingConfig
    from training.model import QLoRALightningModule
    from training.tokenizer import load_tokenizer, TokenizedCollator
    from data.config import PipelineConfig, StageConfig
    from data.datamodule import CurriculumDataModule
    from training.eval_metrics import run_eval
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger
    from peft import PeftModel

    # --- Config ---
    train_config = TrainingConfig()
    pipeline_config = PipelineConfig(
        jsonl_path="python-codes-25k.jsonl",
        stages=[
            StageConfig(stage=1, min_lines=1, max_lines=1),
            StageConfig(stage=2, min_lines=2, max_lines=2),
            StageConfig(stage=3, min_lines=3, max_lines=3),
        ],
    )

    # --- Tokenizer (shared across stages) ---
    tokenizer = load_tokenizer(train_config.model_name)
    collator = TokenizedCollator(tokenizer, max_seq_length=train_config.max_seq_length)

    # --- Data (setup once, dataloaders per stage) ---
    datamodule = CurriculumDataModule(
        config=pipeline_config,
        batch_size=train_config.per_device_batch_size,
        collate_fn=collator,
    )
    datamodule.setup(stage="fit")

    prev_adapter_dir = None

    for stage_idx, stage_cfg in enumerate(pipeline_config.stages):
        # --- ClearML Task per stage ---
        task = Task.init(
            project_name=train_config.clearml_project,
            task_name=f"stage_{stage_idx + 1}_gap_{stage_cfg.min_lines}_{stage_cfg.max_lines}",
            auto_connect_frameworks={"tensorboard": True},
        )
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
            "stage": stage_cfg.stage,
            "min_lines": stage_cfg.min_lines,
            "max_lines": stage_cfg.max_lines,
        })

        # --- Model (fresh per stage for clean optimizer state) ---
        model = QLoRALightningModule(train_config)

        # --- Load previous adapter weights for continuity (stages > 1) ---
        if prev_adapter_dir is not None:
            model.model = PeftModel.from_pretrained(
                model.model.base_model.model,
                prev_adapter_dir,
            )

        # --- Logger ---
        tb_logger = TensorBoardLogger(
            save_dir=os.path.join(train_config.output_dir, "tb_logs"),
            name=f"curriculum_stage_{stage_idx + 1}",
        )

        # --- Trainer (fresh per stage = fresh optimizer/scheduler) ---
        trainer = pl.Trainer(
            max_epochs=train_config.num_train_epochs,
            max_steps=train_config.max_steps if train_config.max_steps > 0 else -1,
            accumulate_grad_batches=train_config.gradient_accumulation_steps,
            logger=tb_logger,
            log_every_n_steps=1,
            precision="16-mixed",
            accelerator="gpu",
            devices=1,
            enable_checkpointing=False,
        )

        # --- Train this stage ---
        train_dl = datamodule.train_dataloader(curriculum_stage=stage_idx + 1)
        val_dl = datamodule.val_dataloader(curriculum_stage=stage_idx + 1)
        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

        # --- Per-stage eval ---
        eval_val_dl = datamodule.val_dataloader(curriculum_stage=stage_idx + 1)
        ground_truths = [sample["middle"] for sample in eval_val_dl.dataset]
        max_new_tokens = stage_cfg.max_lines * 50
        eval_results = run_eval(model, tokenizer, eval_val_dl, ground_truths, max_new_tokens)
        logger = task.get_logger()
        logger.report_scalar(title="eval", series="eval_exact_match", value=eval_results["exact_match"], iteration=stage_idx + 1)
        logger.report_scalar(title="eval", series="eval_edit_sim", value=eval_results["edit_sim"], iteration=stage_idx + 1)
        print(f"Stage {stage_idx + 1} eval: EM={eval_results['exact_match']:.4f}, EditSim={eval_results['edit_sim']:.4f}")

        # --- Save adapter ---
        adapter_dir = os.path.join(train_config.output_dir, f"adapter_stage_{stage_idx + 1}")
        model.save_adapter(adapter_dir)
        print(f"Stage {stage_idx + 1} adapter saved to {adapter_dir}")

        # --- Upload to ClearML ---
        task.upload_artifact(
            name=f"peft_adapter_stage_{stage_idx + 1}",
            artifact_object=adapter_dir,
        )

        # --- Close task before next stage ---
        task.close()
        prev_adapter_dir = adapter_dir

    # --- Final test-set evaluation (EVAL-03) ---
    eval_task = Task.init(
        project_name=train_config.clearml_project,
        task_name="eval-final",
        auto_connect_frameworks={"tensorboard": False},
    )

    test_dl = datamodule.test_dataloader()
    test_ground_truths = [sample["middle"] for sample in test_dl.dataset]
    last_stage_cfg = pipeline_config.stages[-1]
    max_new_tokens = last_stage_cfg.max_lines * 50
    test_results = run_eval(model, tokenizer, test_dl, test_ground_truths, max_new_tokens)

    eval_logger = eval_task.get_logger()
    eval_logger.report_scalar(title="eval", series="eval_exact_match", value=test_results["exact_match"], iteration=0)
    eval_logger.report_scalar(title="eval", series="eval_edit_sim", value=test_results["edit_sim"], iteration=0)
    print(f"Final test eval: EM={test_results['exact_match']:.4f}, EditSim={test_results['edit_sim']:.4f}")
    eval_task.close()


if __name__ == "__main__":
    main()
