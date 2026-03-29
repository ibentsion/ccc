# ClearML + PyTorch Lightning Integration Research

**Project:** CodeComplete Fine-Tuning (QLoRA curriculum training)
**Researched:** 2026-03-29
**Overall confidence:** MEDIUM-HIGH (official docs verified; some edge cases LOW confidence)

---

## 1. Does `pytorch_lightning.loggers.ClearMLLogger` Exist?

**No. There is no native `ClearMLLogger` in PyTorch Lightning.**

Multiple GitHub issues have requested it (Lightning issue #5921, #14137), but as of 2026 it has not been merged into the `lightning` package. Do not try to import `pytorch_lightning.loggers.ClearMLLogger` — it does not exist.

The correct integration path is `Task.init()` from the `clearml` package. ClearML monkey-patches PyTorch and TensorBoard at import time and captures metrics automatically.

**Confidence:** HIGH — verified against official ClearML docs and Lightning issue tracker.

---

## 2. Task.init() — Placement and Parameters

### Placement Rule

`Task.init()` must be called **before** any framework code runs. The canonical placement is:

```python
# top of training script, before torch, lightning, transformers imports if possible
from clearml import Task

task = Task.init(
    project_name="CodeComplete",
    task_name="curriculum-stage-1-gap1",
    task_type=Task.TaskTypes.training,
    output_uri="s3://your-bucket/clearml-artifacts",  # optional, defaults to ClearML server
)
```

**Why placement matters:** ClearML installs framework hooks at `Task.init()` time. If Lightning's `Trainer` or PyTorch is imported before `Task.init()`, some auto-logging hooks may miss early events. The official docs explicitly warn that initialization before training frameworks prevents "synchronization issues that can lead to memory leaks or hanging child processes."

### Key `Task.init()` Parameters

| Parameter | Type | Purpose |
|-----------|------|---------|
| `project_name` | str | Nesting with `/` creates sub-projects (e.g. `"CodeComplete/Stage1"`) |
| `task_name` | str | Human-readable run name |
| `task_type` | TaskTypes enum | `training`, `testing`, `data_processing`, etc. |
| `output_uri` | str | Storage for auto-logged model checkpoints (S3, GCS, Azure, local) |
| `auto_connect_arg_parser` | bool or dict | Controls argparse/click/LightningCLI param capture |
| `auto_connect_frameworks` | dict | Per-framework on/off/wildcard control |
| `auto_resource_monitoring` | bool | GPU/CPU/RAM usage tracking (default True) |
| `reuse_last_task_id` | bool | Reuse task if last run was < 24h ago |
| `continue_last_task` | bool | Resume a task, retaining previous artifacts |

### Controlling Auto-Logging Per Framework

```python
task = Task.init(
    project_name="CodeComplete",
    task_name="stage-1",
    auto_connect_frameworks={
        'pytorch': '*.pt',          # only log .pt files, not every checkpoint
        'tensorboard': True,        # capture TensorBoard scalars
        'matplotlib': False,        # suppress matplotlib auto-capture
    }
)
```

Wildcard values (e.g. `'*.pt'`, `'*.ckpt'`) let you filter which checkpoint files are auto-logged. Any framework not listed defaults to `True`.

---

## 3. Logging Hyperparameters

### Method A — `task.connect()` (preferred for training configs)

Connects a live Python dict. ClearML tracks mutations to it throughout training.

```python
hparams = {
    "model_name": "deepseek-ai/deepseek-coder-1.3b-instruct",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "learning_rate": 2e-4,
    "batch_size": 4,
    "max_seq_len": 2048,
    "curriculum_stage": 1,
    "gap_lines_max": 1,
}
task.connect(hparams, name="Training")  # 'name' sets the section in the UI
```

The `name` parameter creates a named section in the ClearML HYPERPARAMETERS tab. Use it to group related parameters:

```python
task.connect(lora_config,    name="LoRA")
task.connect(training_args,  name="Training")
task.connect(data_config,    name="Dataset")
```

### Method B — `task.connect_configuration()` (for nested/blob configs)

For Pydantic models, dataclasses, or YAML config files that shouldn't be individually parsed:

```python
task.connect_configuration(
    configuration={"lora": {"r": 16, "alpha": 32}, "training": {...}},
    name="full_config"
)
# or pass a yaml file path:
task.connect_configuration("/path/to/config.yaml", name="run_config")
```

Stored as a blob (not key-value), so ClearML shows it as a configuration object, not individual hyperparameter rows.

### LightningCLI Auto-Capture

If using `LightningCLI`, ClearML automatically captures all CLI arguments without calling `task.connect()` manually. This is controlled by the `auto_connect_arg_parser` parameter. To disable:

```python
task = Task.init(..., auto_connect_arg_parser=False)
```

---

## 4. Logging Loss Curves and Custom Metrics

### Auto-Logging via TensorBoard / Lightning's `self.log()`

Lightning's `self.log()` routes metrics through TensorBoard internally. ClearML hooks into TensorBoard and captures all logged scalars automatically — **no extra code needed**.

```python
# Inside LightningModule — this is sufficient for ClearML to capture metrics
class CodeFillModel(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/perplexity", torch.exp(loss))
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("val/loss", loss)
        self.log("val/exact_match", self.compute_exact_match(batch))
```

These appear under the task's SCALARS tab with `/` acting as a grouping delimiter.

### Manual Logging via `Logger.report_scalar()`

When you need metrics outside the Lightning training loop, or want explicit control:

```python
from clearml import Logger

logger = Logger.current_logger()

# Single scalar at a given step
logger.report_scalar(
    title="Curriculum",
    series="stage_transitions",
    value=current_stage,
    iteration=global_step
)

# Summary (non-time-series) value — appears in Summary table, not as a graph
logger.report_single_value(name="best_val_loss", value=0.342)
```

**Title/Series organization:**
- Same `title`, different `series` → multiple lines on **one plot**
- Different `title` → separate plots

This is the key for multi-stage curriculum tracking (see Section 6).

---

## 5. Tracking Multiple Curriculum Stages

There are two viable patterns. The choice depends on whether you want cross-stage comparison in one task or isolated reproducibility per stage.

### Pattern A — Separate Task Per Stage (recommended for this project)

Each curriculum stage is an independent ClearML Task. Stages can be chained via a ClearML Pipeline.

```python
# stage_runner.py
from clearml import Task

def run_stage(stage_num: int, prev_checkpoint: str | None):
    task = Task.init(
        project_name="CodeComplete",
        task_name=f"curriculum-stage-{stage_num}",
        task_type=Task.TaskTypes.training,
    )
    hparams = {"stage": stage_num, "gap_lines_max": stage_num}
    task.connect(hparams, name="Curriculum")

    if prev_checkpoint:
        # Log the inherited checkpoint as an input artifact
        task.add_tags([f"stage-{stage_num}", "curriculum"])

    trainer = pl.Trainer(max_epochs=3)
    trainer.fit(model, datamodule)

    task.close()
    return trainer.checkpoint_callback.best_model_path
```

**Advantages:**
- Each stage has a clean independent task page
- Easy to re-run a single stage from the ClearML UI
- Supports ClearML Pipeline chaining (output of stage N feeds stage N+1)
- Hyperparameter optimization can target individual stages

**Disadvantages:**
- Cross-stage loss curves require ClearML's experiment comparison view

### Pattern B — Single Task with Stage Prefix in Series Names

All stages run inside one `Task.init()`. Stage is encoded in the series name.

```python
task = Task.init(project_name="CodeComplete", task_name="full-curriculum")

for stage in range(1, max_stages + 1):
    logger = Logger.current_logger()
    # Prefix stage into the series name
    for step, (train_loss, val_loss) in enumerate(train_stage(stage)):
        logger.report_scalar("loss", f"train/stage-{stage}", train_loss, iteration=step)
        logger.report_scalar("loss", f"val/stage-{stage}",   val_loss,   iteration=step)
```

**Advantages:** Single task, all curves on one page for easy visual comparison

**Disadvantages:**
- Cannot re-run one stage independently from the UI
- If the run crashes mid-curriculum, the whole task is partial
- Less clean for HPO

**Recommendation for this project:** Use Pattern A (separate tasks) because stages checkpoint independently and the pipeline supports resuming from any stage. Use ClearML's comparison view to compare loss curves across stage tasks.

---

## 6. Logging Model Checkpoints and PEFT Adapter Artifacts

### Auto-Logging Checkpoints (simplest path)

When `auto_connect_frameworks` includes `'pytorch': True` (or a wildcard), ClearML automatically intercepts `torch.save()` calls and uploads `.pt` / `.ckpt` files to the configured `output_uri`.

Lightning's `ModelCheckpoint` callback calls `torch.save()` internally, so checkpoints are captured automatically.

To restrict to only best checkpoints (avoid uploading every epoch):

```python
task = Task.init(
    ...,
    auto_connect_frameworks={'pytorch': '*best*.ckpt'},
)
```

### Manual Checkpoint Logging via `OutputModel`

For explicit control — uploading the PEFT adapter folder rather than raw `.pt` files:

```python
from clearml import OutputModel

output_model = OutputModel(task=task, framework="PyTorch")

# After saving the LoRA adapter with model.save_pretrained()
adapter_dir = f"./checkpoints/stage-{stage}/adapter"
model.save_pretrained(adapter_dir)          # saves adapter_config.json + adapter_model.bin

output_model.update_weights(
    weights_filename=adapter_dir,           # uploads the whole folder (zipped)
    # or use register_uri for remote paths:
    # register_uri="s3://bucket/adapters/stage-1/"
)
output_model.update_design(config_dict={
    "base_model": "deepseek-ai/deepseek-coder-1.3b-instruct",
    "lora_r": 16,
    "lora_alpha": 32,
    "stage": stage,
})
```

### Manual Artifact Upload (for non-model files)

```python
# Upload entire adapter directory as a named artifact
task.upload_artifact(
    name=f"peft-adapter-stage-{stage}",
    artifact_object=adapter_dir,            # path to folder — zipped automatically
)

# Upload a single file
task.upload_artifact(
    name=f"tokenizer-config",
    artifact_object="./tokenizer_config.json",
)
```

Artifacts appear in the task's ARTIFACTS tab and can be retrieved by downstream tasks via `task.artifacts["name"].get_local_copy()`.

### ClearML Callback for Lightning (custom)

Since there is no native `ClearMLLogger`, a clean pattern is a custom Lightning callback:

```python
from pytorch_lightning.callbacks import Callback
from clearml import OutputModel, Logger

class ClearMLCheckpointCallback(Callback):
    def __init__(self, task, stage_num: int):
        self.task = task
        self.stage_num = stage_num
        self.output_model = OutputModel(task=task, framework="PyTorch")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        pass  # ClearML auto-logs via torch.save hook

    def on_train_epoch_end(self, trainer, pl_module):
        logger = Logger.current_logger()
        metrics = trainer.callback_metrics
        step = trainer.global_step
        for key, val in metrics.items():
            logger.report_scalar(
                title=key,
                series=f"stage-{self.stage_num}",
                value=float(val),
                iteration=step,
            )
```

---

## 7. ClearML Dataset Versioning for JSONL Training Data

### Creating and Versioning the Dataset

```python
from clearml import Dataset

# One-time: create the dataset from python-codes-25k.jsonl
dataset = Dataset.create(
    dataset_name="python-codes",
    dataset_project="CodeComplete/Datasets",
    dataset_version="1.0",
)
dataset.add_files(path="/home/ido/git/ccc/python-codes-25k.jsonl")
dataset.upload()
dataset.finalize()  # makes it immutable / published
```

Semantic versioning is supported. If `dataset_version` is omitted, ClearML auto-increments from the latest version.

### Consuming the Dataset in Training Scripts

```python
from clearml import Dataset

dataset = Dataset.get(
    dataset_project="CodeComplete/Datasets",
    dataset_name="python-codes",
    dataset_version="1.0",           # pin to exact version for reproducibility
    alias="training_data",           # stores dataset ID in task hyperparameters
)
local_path = dataset.get_local_copy()  # returns path to read-only cached local copy
jsonl_file = f"{local_path}/python-codes-25k.jsonl"
```

The `alias` parameter is important: it stores the resolved dataset ID under the "Datasets" section in the task's hyperparameters, making the exact data version used by each experiment visible in the UI and overridable for remote execution.

### Dataset Lineage for Processed Subsets

If you pre-process into stage-specific subsets (e.g., gap-1-only samples), create child datasets that reference the parent:

```python
processed_dataset = Dataset.create(
    dataset_name="python-codes-gap1",
    dataset_project="CodeComplete/Datasets",
    parent_datasets=[source_dataset.id],   # tracks provenance
    dataset_version="1.0",
)
processed_dataset.add_files(path="./processed/gap1/")
processed_dataset.upload()
processed_dataset.finalize()
```

---

## 8. Auto-Logging vs Manual Logging Trade-offs

| Concern | Auto-Logging | Manual Logging |
|---------|-------------|----------------|
| Setup effort | Minimal — `Task.init()` only | Requires explicit API calls at each metric/artifact |
| Coverage | Captures framework metrics (TensorBoard, torch.save, LightningCLI args) | Captures exactly what you specify |
| Control | Coarse (per-framework on/off/wildcard) | Fine-grained (per-metric, per-artifact) |
| Artifact filtering | Wildcard on file paths | Explicit name + object |
| PEFT adapter folder | Not captured unless adapter is saved as `.pt` | Requires `task.upload_artifact()` or `OutputModel` |
| Stage-prefixed series | Not possible — series names come from Lightning's `self.log()` keys | Full control via `logger.report_scalar(title, series, ...)` |
| Crash safety | Works even if training code crashes before `task.close()` | Must ensure calls happen before crash |
| Remote execution / HPO | Params from argparse/LightningCLI are override-able | `task.connect()` params are also override-able |

**Recommendation:** Use auto-logging as the base (it captures loss curves from `self.log()` for free via TensorBoard hooking), and supplement with manual logging for:
1. PEFT adapter artifact upload (auto-logging won't zip-and-upload a directory)
2. Stage transition markers and curriculum metadata
3. Summary metrics (best val loss per stage via `report_single_value`)
4. Dataset lineage via `Dataset.get(alias=...)`

---

## 9. Summary: Recommended Integration Pattern for This Project

```
Script startup order:
  1. from clearml import Task, Logger, Dataset, OutputModel
  2. task = Task.init(...)                    # before any torch/lightning imports
  3. task.connect(hparams, name="Training")   # log hyperparameters
  4. dataset = Dataset.get(..., alias="...")  # log dataset version
  5. import pytorch_lightning as pl           # after Task.init
  6. trainer = pl.Trainer(...)               # Lightning trainer, no ClearML logger needed
  7. trainer.fit(model, datamodule)           # auto-logging active
  8. task.upload_artifact("peft-adapter", adapter_dir)  # manual artifact upload
  9. task.close()                             # flush and finalize
```

For curriculum stages: one `Task.init()` per stage, structured as `project_name="CodeComplete/Curriculum"`, `task_name=f"stage-{n}-gap{n}"`.

---

## 10. Known Gotchas and Pitfalls

### Pitfall 1: Task.init() After Framework Imports

If you import `torch` or `pytorch_lightning` before `Task.init()`, some hooks may not attach correctly and checkpoint auto-logging may silently fail.

**Fix:** Put `from clearml import Task; task = Task.init(...)` at the very top of the entry-point script, before other imports.

### Pitfall 2: PEFT Adapter Not Auto-Logged

`model.save_pretrained(dir)` writes multiple files to a directory. ClearML's PyTorch auto-logging watches for `torch.save()` calls on `.pt` / `.ckpt` paths. A HuggingFace `save_pretrained()` call writes `adapter_model.bin` (or `adapter_model.safetensors`) and `adapter_config.json` but does NOT call `torch.save()` directly in a way ClearML reliably hooks.

**Fix:** Explicitly upload the adapter directory via `task.upload_artifact("peft-adapter-stage-N", adapter_dir)`.

### Pitfall 3: TensorBoard Required for Auto-Scalar Capture

ClearML captures Lightning `self.log()` scalars by hooking into TensorBoard. If `TensorBoardLogger` is NOT added to the `Trainer`, ClearML may not capture scalars.

**Fix:** Either add `TensorBoardLogger` explicitly, or rely on manual `Logger.report_scalar()` calls in a custom callback. The safest approach:

```python
from pytorch_lightning.loggers import TensorBoardLogger
trainer = pl.Trainer(
    logger=TensorBoardLogger("tb_logs", name="stage-1"),
    ...
)
```

ClearML will then absorb the TensorBoard output automatically.

### Pitfall 4: Scalar Iteration Counter Resets Across Stages

If running all curriculum stages in one script with a single `Task.init()`, Lightning's global step counter may reset between stages if you recreate the `Trainer`. This makes loss curves from different stages overlap at step 0.

**Fix:** Either use separate Tasks per stage (Pattern A), or pass `iteration=global_offset + local_step` to `logger.report_scalar()` in a custom callback.

### Pitfall 5: `task.close()` Not Called on Crash

If training crashes, ClearML marks the task as Failed and flushes what it has. Artifacts uploaded mid-run are preserved. However, `task.connect()` updates after the crash are lost.

**Fix:** Connect all hyperparameters at init time before training starts, not during training.

---

## Sources

- [ClearML PyTorch Lightning Integration](https://clear.ml/docs/latest/docs/integrations/pytorch_lightning/) — HIGH confidence
- [ClearML Task SDK](https://clear.ml/docs/latest/docs/clearml_sdk/task_sdk/) — HIGH confidence
- [ClearML Logger / Fundamentals](https://clear.ml/docs/latest/docs/fundamentals/logger/) — HIGH confidence
- [ClearML Hyperparameters](https://clear.ml/docs/latest/docs/fundamentals/hyperparameters/) — HIGH confidence
- [ClearML Dataset SDK](https://clear.ml/docs/latest/docs/clearml_data/clearml_data_sdk/) — HIGH confidence
- [ClearML Scalar Reporting Guide](https://clear.ml/docs/latest/docs/guides/reporting/scalar_reporting/) — HIGH confidence
- [ClearML Artifacts Fundamentals](https://clear.ml/docs/latest/docs/fundamentals/artifacts/) — HIGH confidence
- [ClearML PyTorch Model Updating Example](https://clear.ml/docs/latest/docs/guides/frameworks/pytorch/model_updating/) — HIGH confidence
- [ClearML Auto-Logging Guide](https://clear.ml/docs/latest/docs/getting_started/auto_log_exp/) — HIGH confidence
- [Lightning issue #5921 — ClearMLLogger request](https://github.com/PyTorchLightning/pytorch-lightning/issues/5921) — confirms no native logger
- [Lightning issue #14137 — ClearML logging support](https://github.com/Lightning-AI/pytorch-lightning/issues/14137) — confirms no native logger
- [ClearML GitHub — pytorch-lightning example](https://github.com/clearml/clearml/blob/master/examples/frameworks/pytorch-lightning/pytorch_lightning_example.py) — MEDIUM confidence (code example)
