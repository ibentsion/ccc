# Requirements: CodeComplete Fine-Tuning

**Defined:** 2026-03-29
**Core Value:** A reproducible training pipeline that produces a curriculum-trained model capable of filling in missing Python lines, with each training stage fully logged in ClearML for experiment comparison.

## v1 Requirements

### Data Pipeline

- [x] **DATA-01**: Pipeline loads JSONL file (instruction/input/output format) and splits into train/validation/test sets with configurable ratios
- [ ] **DATA-02**: Pipeline creates FIM-format training pairs by masking N consecutive lines from complete code samples, using DeepSeek-Coder PSM tokens (`<｜fim▁begin｜>`, `<｜fim▁hole｜>`, `<｜fim▁end｜>`)
- [ ] **DATA-03**: Dataset supports configurable gap size N (number of consecutive lines to mask) per curriculum stage
- [x] **DATA-04**: Gap selection is deterministic given a seed (reproducible splits and masking)
- [x] **DATA-05**: DataModule exposes per-stage DataLoaders with hybrid replay (75% current-stage, 25% prior-stage samples)

### Training

- [x] **TRAIN-01**: QLoRA training with DeepSeek-Coder 1.3B or 6.7B (configurable), using 4-bit NF4 quantization with double quantization enabled
- [x] **TRAIN-02**: LoRA targets all 7 projection layers (q/k/v/o_proj, gate/up/down_proj), r=16, alpha=32
- [x] **TRAIN-03**: Training orchestrated via PyTorch Lightning LightningModule with correct QLoRA init sequence (from_pretrained → prepare_model_for_kbit_training → get_peft_model)
- [ ] **TRAIN-04**: Curriculum advances through configurable stages (e.g., 1-line → 2-line → 3-line gaps) within a single continuous training run
- [x] **TRAIN-05**: Constant learning rate schedule through all curriculum stages (not cosine decay)
- [x] **TRAIN-06**: Prompt masking applied so loss is computed only on the missing lines (not the context)
- [x] **TRAIN-07**: PEFT adapter checkpoint saved to disk after each curriculum stage

### Experiment Tracking

- [ ] **EXP-01**: ClearML experiment initialized via Task.init() before any Lightning/PyTorch imports; one ClearML Task per curriculum stage
- [ ] **EXP-02**: Training and validation loss curves logged per step via TensorBoardLogger (auto-captured by ClearML)
- [ ] **EXP-03**: Hyperparameters (model name, LoRA config, learning rate, batch size, stage config) logged to ClearML per task
- [ ] **EXP-04**: PEFT adapter directory uploaded to ClearML as artifact after each stage completes

### Evaluation

- [ ] **EVAL-01**: Exact Match (EM) score computed on validation set per curriculum stage, logged to ClearML
- [ ] **EVAL-02**: Edit Similarity score computed on validation set per curriculum stage, logged to ClearML
- [ ] **EVAL-03**: Final test set evaluation run after all curriculum stages complete, results logged to ClearML

## v2 Requirements

### Evaluation

- **EVAL-V2-01**: Pass@1 execution test on 200-sample test subset (requires Python sandbox)
- **EVAL-V2-02**: CodeBLEU metric computation
- **EVAL-V2-03**: HumanEval-Infilling benchmark run on final model

### Training Enhancements

- **TRAIN-V2-01**: Horizon-Length Prediction auxiliary head (arxiv 2410.03103 — up to 24% relative EM improvement)
- **TRAIN-V2-02**: Cyclomatic complexity annotation for intra-stage difficulty ordering
- **TRAIN-V2-03**: Loss-plateau detection per stage with early stage advancement

### Infrastructure

- **INFRA-V2-01**: ClearML Dataset versioning for the JSONL source file
- **INFRA-V2-02**: Inference script for running the trained model on new code snippets

## Out of Scope

| Feature | Reason |
|---------|--------|
| Inference API / serving | Not needed — checkpoint + ClearML experiment is the deliverable |
| Multi-language support | Python only for v1 |
| Multi-GPU / FSDP training | 4-bit QLoRA has limited DDP support; single GPU sufficient for 1.3B/6.7B |
| Dataset scraping / augmentation | Existing JSONL is the input |
| DPO alignment after SFT | Complexity not warranted for v1 |
| Lightning BitsandbytesPrecision plugin | Silently corrupts quantized weights — explicitly excluded |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Complete |
| DATA-02 | Phase 1 | Pending |
| DATA-03 | Phase 1 | Pending |
| DATA-04 | Phase 1 | Complete |
| DATA-05 | Phase 1 | Complete |
| TRAIN-01 | Phase 2 | Complete |
| TRAIN-02 | Phase 2 | Complete |
| TRAIN-03 | Phase 2 | Complete |
| TRAIN-04 | Phase 3 | Pending |
| TRAIN-05 | Phase 2 | Complete |
| TRAIN-06 | Phase 2 | Complete |
| TRAIN-07 | Phase 2 | Complete |
| EXP-01 | Phase 2 | Pending |
| EXP-02 | Phase 2 | Pending |
| EXP-03 | Phase 2 | Pending |
| EXP-04 | Phase 2 | Pending |
| EVAL-01 | Phase 4 | Pending |
| EVAL-02 | Phase 4 | Pending |
| EVAL-03 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 19 total
- Mapped to phases: 19
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-29*
*Last updated: 2026-03-29 — traceability populated after roadmap creation*
