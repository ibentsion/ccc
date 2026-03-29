# Project: CodeComplete Fine-Tuning

## Overview

Fine-tune DeepSeek-Coder (1.3B or 6.7B) to complete missing lines of Python code. The system takes a dataset of complete Python code samples, creates fill-in-the-blank training pairs by masking consecutive lines, and trains with QLoRA using curriculum learning — starting from single missing lines and progressively increasing the number of consecutive missing lines. Training is orchestrated with PyTorch Lightning and tracked end-to-end in ClearML.

## Core Value

A reproducible training pipeline that produces a curriculum-trained model capable of filling in missing Python lines, with each training stage fully logged in ClearML for experiment comparison.

## Problem

Code completion models often struggle with multi-line gap filling because standard training treats all samples equally. Curriculum learning — starting simple (1 missing line) and gradually increasing complexity (N consecutive lines) — should improve the model's ability to reason about context and produce coherent multi-line completions.

## Target Users

- Researcher / engineer running experiments locally or on a GPU server

## Tech Stack

- **Model:** DeepSeek-Coder 1.3B or 6.7B (HuggingFace)
- **Training:** QLoRA via `peft` + `bitsandbytes`
- **Orchestration:** PyTorch Lightning (`LightningModule`, `Trainer`)
- **Experiment tracking:** ClearML (metrics, hyperparams, artifacts)
- **Data format:** JSONL with `instruction`, `input` (complete code), `output` fields
- **Language:** Python

## Data Pipeline

- Input: `python-codes-25k.jsonl` (JSONL with instruction/input/output fields)
- Split into train / validation / test sets
- Gap creation: pipeline randomly selects N consecutive lines from the code, removes them, and creates `(context_with_gap, missing_lines)` pairs
- Curriculum stages: N starts at 1 (single missing line) and increases to a configurable range (e.g., 1–5 consecutive lines)

## Curriculum Learning Strategy

Each curriculum stage trains the model on progressively harder gaps:
- Stage 1: 1 missing line
- Stage 2: 1–2 missing lines
- Stage 3: 1–3 missing lines
- (configurable max)

Each stage is a separate ClearML task or a logged phase within the same experiment.

## Success Criteria

- ClearML experiment logged with loss curves, eval metrics (e.g., exact match, BLEU/CodeBLEU) per curriculum stage
- Saved model checkpoints per stage
- Test set evaluation at final stage

## Out of Scope (v1)

- Serving / inference API
- Multi-language support (Python only)
- Distributed multi-node training
- Dataset scraping or augmentation
