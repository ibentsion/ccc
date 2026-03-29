# Curriculum Learning for Code FIM Fine-Tuning

**Project:** CodeComplete Fine-Tuning (DeepSeek-Coder + QLoRA)
**Researched:** 2026-03-29
**Overall confidence:** MEDIUM-HIGH (key claims verified via papers; some implementation details are LOW confidence)

---

## 1. Summary of Relevant Literature

### 1.1 "Improving FIM Code Completions via Context & Curriculum Based Learning" (Dec 2024)
**Source:** https://arxiv.org/abs/2412.16589 — MEDIUM-HIGH confidence (peer-reviewed, production-validated)

The most directly relevant paper for this project. Key design decisions:

- **Difficulty = AST node type**, not length. They extracted "hard-to-complete" span types from production telemetry (Call Expression, Function Definition, Class Definition) because these node types have lower Completion Acceptance Rate in deployed models.
- **Training is not strictly sequential**: They use a fixed sampling distribution throughout the entire run (e.g., 35% random spans, 12% Call Expression, 12% Function Definition, 10% Class Definition, etc.). This is a weighted data-mixing approach, not staged curriculum progression.
- **Three variants tested:**
  - CUFT (Curriculum-Only Fine-Tuning): hard patterns, no retrieval context
  - COFT (Context-Only Fine-Tuning): easy random spans with symbol context injected via BM25/TypeScript compiler
  - CMFT (Combined): both together — **best results**
- **Gains scale inversely with model size**: DS-Coder-1B gained 6.25%; DS-Coder-7B gained 1.25%. Smaller models benefit more from explicit curriculum.
- **Online validation**: A/B tested in production. CAR improved 2.57–4.24% across line groups. This is unusually strong evidence — most curriculum papers only show offline gains.

### 1.2 "Curriculum Learning for Small Code Language Models" (ACL SRW 2024)
**Source:** https://arxiv.org/abs/2407.10194 — HIGH confidence (ACL peer-reviewed)

Trains 1M-parameter decoder-only models from scratch; also fine-tunes Code Llama 7B.

- **Difficulty metric:** Overall Metric (OM) = (Cyclomatic Complexity + Halstead Difficulty) / 2
  - Easy: OM < 2, Medium: 2 ≤ OM < 4, Hard: OM ≥ 4
- **Three scheduling strategies compared:**
  - Sequential (easy → medium → hard, disjoint)
  - Incremental (easy, then easy+medium, then all three)
  - Hybrid (hardest examples from prior stages mixed into each new stage)
- **Results:**
  - **Code execution tasks**: Hybrid CL improved accuracy from 74.58% to 79.23% (baseline). Hard-only training scored 61.78% — far below hybrid.
  - **Code completion tasks**: Minimal gains (token-level: 81.27% vs. 81.23% baseline). Curriculum matters far more for reasoning/execution tasks than for next-token prediction.
  - When fine-tuning Code Llama 7B: execution accuracy 81.29% → 85.18% with hybrid CL.
- **Key takeaway for this project**: For *missing-line completion* (a generative task rather than execution reasoning), the benefit of CL may be modest at the token level. The gains appear in harder subsets of eval data.

### 1.3 "Alignment with Fill-In-the-Middle for Enhancing Code Generation" (EMNLP 2025)
**Source:** https://aclanthology.org/2025.emnlp-main.419.pdf — MEDIUM confidence (proceedings, limited detail extracted)

- Pairs FIM fine-tuning with DPO (Direct Preference Optimization).
- Defines difficulty via **AST node count** for the middle segment to generate.
- Training progresses from shorter/simpler completions (fewer AST nodes) toward longer/deeper ones.
- Reports improved generalization on harder completion tasks when trained with this progressive structure.

### 1.4 "Horizon-Length Prediction: Advancing FIM Capabilities" (Oct 2024)
**Source:** https://arxiv.org/abs/2410.03103 — HIGH confidence (strong empirical results)

Not a curriculum paper per se, but directly relevant:

- **Core idea:** Adds an auxiliary training head that predicts how many tokens remain in the middle segment at each position. This teaches the model to "plan ahead" rather than rely on post-processing.
- **Loss:** L = L_NTP + 0.1 * L_HLP (L_HLP is L1 regression loss on normalized remaining length)
- **Results:** Up to 24% relative improvement on repository-level exact match, 5% average on SAFIM, 18% on code fixing, 6% on CRUXEval — all without inference overhead.
- **Why it matters here:** This is the easiest single technique to add to a FIM fine-tuning run that explicitly addresses the difficulty of multi-line completions. It works alongside any curriculum strategy.

### 1.5 "Does the Definition of Difficulty Matter?" (Nov 2024)
**Source:** https://arxiv.org/abs/2411.00973 — MEDIUM confidence (preprint)

Empirically compares loss-based, perplexity-based, and learning-based difficulty scoring functions:

- Most scoring functions show >70% agreement on which samples are hard, suggesting the choice of metric matters less than expected.
- Loss-based functions (CELoss, CVLoss) give fine-grained per-sample scores; learning-based (CumAcc, FIT) are coarser.
- **Critical finding:** "Could not find evidence for significantly higher performance" with curriculum vs. standard training in their controlled comparisons. Easy-to-hard substantially outperformed hard-to-easy, but neither consistently beat uniform sampling.
- Ensemble of orderings (easy-to-hard + hard-to-easy models combined) exceeded any single model.

### 1.6 "Curriculum Learning for LLM Pretraining: An Analysis of Learning Dynamics" (Jan 2026)
**Source:** https://arxiv.org/pdf/2601.21698 — MEDIUM confidence (very recent)

- Multi-stage pretraining (dominant web data → high-quality data) is validated as effective and used by OLMo 2, Phi-4, LongCat-Flash.
- **CL + cosine LR decay interact badly**: curriculum's advantage diminishes under standard cosine decay because LR is already low when hard examples appear. Fix: use constant LR + model averaging of final checkpoints (as used in Llama 3), or a more gradual decay schedule.
- Phase transitions (emergent abilities) appear during specific training windows, making curriculum order matter for capability acquisition.

### 1.7 DeepSpeed Curriculum Learning Tutorial
**Source:** https://www.deepspeed.ai/tutorials/curriculum-learning/ — HIGH confidence (official docs)

Production-validated CL implementation using **sequence length** as the primary difficulty proxy:

- Pacing options: fixed linear, fixed root (`(step/total)^(1/k)`), fixed discrete (manually specified breakpoints).
- Reported: 3.3x faster GPT-2 pre-training to the same perplexity with CL on sequence length.
- Learning rate warmup must be adjusted: divide warmup samples by `min_difficulty` to account for shorter early sequences.

### 1.8 SAFIM Benchmark — Syntax-Aware FIM Difficulty Taxonomy
**Source:** https://arxiv.org/html/2403.04814v1 — HIGH confidence (benchmark paper)

Defines three canonical difficulty levels for FIM based on syntactic role:
1. **Algorithmic Block** (hardest): innermost loop body or DP transition; test fails if replaced with no-op.
2. **Control-Flow Expression**: condition in `for`/`while`/`if`; test fails if replaced with trivially wrong value.
3. **API Function Call** (easiest): single function call requiring library knowledge.

Primary evaluation metric: **Pass@1** via execution against unit tests. This is the strongest possible signal for FIM quality.

---

## 2. Difficulty Definitions for Missing-Line Tasks

The project already uses "number of consecutive missing lines" as the primary curriculum axis. Here is how that maps to the research, and what additional axes to consider.

### 2.1 Number-of-Lines as Difficulty (the project's current plan)

**Justification:** Supported indirectly by multiple papers. Token count / sequence length is the dominant difficulty proxy in DeepSpeed CL, and the EMNLP 2025 FIM paper uses AST node count of the *middle segment*, which correlates with line count for Python.

**Practical thresholds for Python code:**

| Stage | Missing lines | Rationale |
|-------|--------------|-----------|
| 1 | 1 | Single assignment, return, or simple expression — model only needs to fill a leaf node |
| 2 | 2–3 | Typically one control-flow line + one body line; requires local coherence |
| 3 | 4–6 | A full if-block or short loop; requires structural reasoning |
| 4 | 7–10 | Multiple logical units; approaches function-body-level completion |

**Limitation:** Two 1-line gaps are not equal. A missing `return x` is far easier than a missing DP state-transition equation. Pure line count ignores semantic complexity.

### 2.2 Complementary Difficulty Axes (ranked by implementation cost)

**Low cost to add:**

- **Token length of the missing span** — normalize by model's tokenizer. More precise than line count. Already available during dataset preprocessing.
- **Cyclomatic complexity of the surrounding function** — measures control-flow density of context. Computable via `radon` (Python library). The ACL 2024 paper uses this as half of their OM metric.
- **Position in function** — missing lines at the beginning of a function body are harder (no local context established) than missing lines near the end (pattern largely set). Easy to compute.

**Medium cost:**

- **AST node type of the gap** — require AST parsing (`ast` module). Categorize by whether gap contains: expression statement, assignment, control-flow header, function call. Maps to the SAFIM taxonomy.
- **Halstead difficulty of the gap** — computable via `radon`. The ACL 2024 paper's second metric.

**High cost (not recommended for v1):**

- **Model-loss-based difficulty** — requires a reference model forward pass per sample. Computationally expensive at dataset scale.
- **Execution-based difficulty** — requires a Python sandbox per sample.

**Recommendation for this project:** Use line count as the primary stage axis (already planned). Add Cyclomatic Complexity of the enclosing function as a secondary sort within each stage. This is two lines of `radon` per sample and substantially improves intra-stage ordering.

---

## 3. Scheduling Strategies

### 3.1 Fixed Stages (the project's current plan)

Each stage trains on samples with a fixed max missing-line count. Stage boundary is defined by either:

- **Fixed epoch count per stage** — simple, predictable, but ignores whether the model has actually learned the current difficulty. Works well when stages are short.
- **Fixed step count per stage** — equivalent to epoch count for uniform datasets; allows easier comparison across dataset sizes. **Preferred over epochs** when dataset size may change between experiments.
- **Loss plateau trigger** — move to next stage when validation loss stops improving. More adaptive; requires eval loop to run per stage. The ACL 2024 paper used fixed step counts; adaptive triggers are theoretically better but need a stable eval signal.

**Practical recommendation:** Use fixed step counts per stage for v1 (predictable and reproducible). Log per-stage validation loss to ClearML. If a stage's loss has not dropped by at least 5% relative before the stage ends, flag it in logs for manual review.

### 3.2 Incremental vs. Disjoint Stages

**Disjoint (Sequential):** Each stage trains *only* on the current difficulty level.
- Risk: catastrophic forgetting of easier patterns when hard samples arrive.
- ACL 2024 result: worse than Hybrid on hard subsets.

**Incremental:** Each stage trains on all difficulties seen so far (cumulative pool).
- Avoids forgetting. Dataset grows each stage — later stages dominate compute.
- ACL 2024 result: comparable to Hybrid but with higher compute cost.

**Hybrid (recommended):** Each stage mixes current-difficulty samples with a replay fraction of the hardest examples from the previous stage.
- ACL 2024: best results (79.23% vs 74.58% baseline on execution tasks).
- Prevents forgetting without full dataset accumulation.
- Suggested replay ratio: 20–30% of current stage budget drawn from prior-stage hard examples.

### 3.3 Continuous Data Mixing (alternative to discrete stages)

Instead of stage transitions, fix a sampling distribution over difficulty buckets throughout training and gradually shift weights:

- The arxiv 2412.16589 paper does this: fixed distribution over AST node types for the entire fine-tuning run.
- Simpler to implement — no stage logic, no checkpoint management.
- Less interpretable — harder to know if any given difficulty level converged.
- **When to use this:** If the dataset is small enough that discrete stages would have too few steps each, a weighted distribution over all difficulty levels with a slow annealing from easy-heavy to hard-heavy is a reasonable alternative.

**Pacing functions for continuous mixing:**

| Name | Formula | Use when |
|------|---------|----------|
| Linear | `hard_fraction = min(1.0, start + (1-start) * step/total)` | Default, predictable |
| Root (concave) | `hard_fraction = (step/total)^0.5` | Spend more time on easy examples early |
| Step function | Fixed discrete transitions at step thresholds | When you want explicit phase control |

---

## 4. Single Continuous Training Run vs. Separate Checkpoints

### 4.1 Arguments for a Single Continuous Run

- Avoids repeated LR warmup overhead.
- Simpler ClearML logging (one task, multiple phase labels).
- No risk of checkpoint incompatibility across QLoRA adapter versions.
- The CMFT approach in arxiv 2412.16589 and the DeepSpeed CL system both use single continuous runs with shifting data distributions.

### 4.2 Arguments for Separate Checkpoints per Stage

- Each stage can be independently compared and rolled back.
- Enables ablation: "which stage contributed what?"
- Avoids a late-stage collapse destroying a good mid-stage checkpoint.
- The project spec mentions "saved model checkpoints per stage" as a success criterion — this suggests separate checkpoints are preferred here.
- In QLoRA, adapter weights are small (a few hundred MB for 7B model at rank 16), so saving a full adapter per stage is cheap.

### 4.3 Recommended Approach: Single Run with Per-Stage Checkpoint Saves

Train in a single Lightning `Trainer` run. After each stage's designated step count is completed:

1. Save the current adapter checkpoint (named `stage_N_checkpoint`).
2. Log a ClearML artifact for the checkpoint.
3. Run a validation pass and log stage-specific metrics.
4. Update the DataLoader to use the next stage's difficulty distribution.
5. Continue training without resetting the optimizer.

**Do NOT reset learning rate between stages** unless validation loss has plateaued for an entire stage. LR resets are expensive and can destabilize LoRA adapters. If resetting, use a short warmup (100–200 steps) at the start of each new stage.

**Exception:** If the stage transition moves to a substantially different data distribution (e.g., single-line to 5+ line completions), a mild LR bump back toward the peak (e.g., 1.5x current LR) with 100-step warmup can help the model adapt faster.

### 4.4 Learning Rate Interaction Warning

From arxiv 2601.21698: standard cosine LR decay reduces CL's advantage because the LR is already near-zero when hard examples arrive. For a multi-stage curriculum:

- Use **constant LR** during staging, then apply cosine decay only in the final stage.
- Or use a **cyclic LR** that resets mildly at each stage boundary.
- For QLoRA, 2e-4 with cosine decay to 1e-5 across the full run is a safe default (from lightning.ai LoRA insights).

---

## 5. Data Mixing Strategies

### 5.1 Strict Stage Progression (disjoint)

Only train on samples matching the current stage's difficulty window. Simple but risks forgetting. Not recommended without a replay buffer.

### 5.2 Weighted Pool with Scheduled Reweighting

All samples in the pool at all times; sampling weights shift over training:

```
easy_weight = max(0.1, 1.0 - step/total_steps)
hard_weight = min(0.9, step/total_steps)
```

Simple to implement in a PyTorch `WeightedRandomSampler`. The arxiv 2412.16589 paper uses a fixed (non-annealed) version of this.

### 5.3 Hybrid Replay (recommended)

Each stage trains on current-difficulty samples as the primary distribution, with a fixed fraction replayed from prior stages:

```
stage_N_samples = 0.75 * current_difficulty_pool + 0.25 * replay_from_prior_stages
```

Select replay samples by highest loss on the prior stage's final eval (hardest examples from easier difficulties). This maintains generalization without inflating dataset size.

### 5.4 Anti-Curriculum (hard-first)

From the "Does the Definition of Difficulty Matter?" paper: hard-to-easy performs **worse** than easy-to-hard but **better** than hard-to-easy for individual models. However, ensembling easy-to-hard and hard-to-easy models beats either alone. Not recommended for a single-model run.

---

## 6. Evaluation Metrics for Code Completion

### 6.1 Exact Match (EM)

- Compares predicted middle segment to ground truth character-by-character (after whitespace normalization).
- **Pro:** Zero ambiguity, fast to compute, standard in CCEval/RepoBench.
- **Con:** Any whitespace or naming variation scores zero, even for functionally correct code.
- **Use:** Primary offline metric for line completion. Report EM per stage.

### 6.2 Prefix Match / Edit Similarity

- Used in CCEval: "next-line accuracy" or "edit distance normalized by length."
- More forgiving than EM; rewards partial correctness.
- **Use:** Secondary offline metric. Useful for multi-line completions where exact match is too strict.

### 6.3 CodeBLEU

- Formula: weighted combination of BLEU, AST match, and data-flow match.
- **Source:** https://arxiv.org/abs/2009.10297
- **Pro:** Captures syntactic and semantic similarity beyond surface n-grams.
- **Con:** Correlates poorly with functional correctness. Can reward wrong-but-structurally-similar code. ICLR 2024 paper "Beyond Accuracy" shows CodeBLEU fails to distinguish functional quality.
- **Use:** Report as supplementary metric for interpretability; do not use as primary signal for curriculum stage promotion.

### 6.4 Pass@k (Execution-Based)

- **k=1:** Run the model's top completion; score 1 if code passes all unit tests.
- **k=5 or k=10:** Sample k completions; score 1 if any passes.
- **Pro:** The only metric that actually measures whether the completed code works. SAFIM, HumanEval-Infilling, and CRUXEval all use this.
- **Con:** Requires a Python execution sandbox per sample. Not feasible for every training step.
- **Practical approach:** Compute Pass@1 on a held-out test subset (100–500 examples) at the end of each curriculum stage. This is your primary quality gate.

### 6.5 HumanEval-Infilling

- **Source:** https://github.com/openai/human-eval-infilling
- Standard benchmark for FIM models. Reports Pass@1 on single-line and multi-line infilling tasks.
- Use for final model evaluation, not per-stage monitoring.

### 6.6 Recommended Metric Stack for This Project

| When | Metric | Why |
|------|--------|-----|
| Every eval step during training | Token-level loss (NLL) | Fast, stable training signal |
| Per-stage validation | Exact Match (EM) | Offline quality; quick |
| Per-stage validation | Edit Similarity | Forgiving EM for multi-line stages |
| End of each stage | Pass@1 on 200-sample subset | Functional correctness gate |
| Final model | HumanEval-Infilling Pass@1 | Standardized comparison |
| Final model | EM + CodeBLEU on test set | Completeness |

---

## 7. Practical Implementation Notes for This Project

### 7.1 Dataset Construction

The project already plans to mask N consecutive lines. Additional recommendations:

- **Stratify by position**: Track whether the gap is in the top, middle, or bottom third of the function body. Report EM by position — gaps at function start are typically harder.
- **Filter trivially easy gaps**: Single-line gaps that are just `pass`, blank lines, or single closing brackets should be excluded or placed in a "warmup" bin before Stage 1.
- **Filter trivially hard gaps**: Gaps that include function signature or class definition lines should be excluded from v1 (these require structural understanding beyond missing-line completion).

### 7.2 QLoRA Considerations

- LoRA rank 16, alpha 32, targeting q/k/v/o projections: standard effective setup confirmed by multiple sources for 6.7B models.
- DeepSeek-Coder already has FIM in its pretraining; fine-tuning should re-use the model's existing `<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>` tokens.
- With QLoRA + 4-bit quantization, gradient flow is through the LoRA adapters only. This means the model's base FIM knowledge is frozen; fine-tuning teaches the adapters to specialize for the project's Python distribution and gap patterns.
- **Batch size and gradient accumulation:** With 1 GPU and sequence length 1024, effective batch size of 8–16 (via gradient accumulation) is standard. For curriculum transitions, do not change batch size between stages.

### 7.3 ClearML Logging Strategy

- Log each curriculum stage as a labeled series (e.g., `stage_1/loss`, `stage_2/loss`) to enable per-stage loss curve comparison.
- Log per-stage EM and Pass@1 as scalar metrics at stage boundaries.
- Upload per-stage adapter checkpoints as artifacts.
- Log the curriculum configuration (N_missing_lines per stage, step counts, replay ratio) as hyperparameters.

### 7.4 Horizon-Length Prediction Add-on (optional)

The HLP technique (arxiv 2410.03103) adds a small linear head that predicts remaining middle tokens at each position. This directly addresses multi-line completion difficulty with minimal code change:

- Add a 1-layer linear regression head on transformer hidden states.
- Add L_HLP (L1 loss on normalized remaining-token count) to training loss with weight 0.1.
- Discard the HLP head at inference.
- Reported gains: up to 24% relative improvement on repository-level EM.
- This is compatible with QLoRA — the head can be trained as part of the LoRA adapter.

---

## 8. Pitfalls and Warnings

### 8.1 Curriculum Gains May Be Modest for Completion Tasks

The ACL 2024 paper found near-zero improvement in token-level code completion metrics from CL, while execution/reasoning tasks improved substantially. The primary mechanism: curriculum helps with generalization to hard examples, not with average-case accuracy. **Expect improvements on Stage 4 (7–10 line) completions, not Stage 1 (1 line) completions.**

### 8.2 Cosine LR Decay Undermines Multi-Stage Curriculum

Do not use cosine decay that reaches near-zero before the final curriculum stage. The model needs meaningful gradient signal when it first encounters hard examples. Use constant LR through stages, or a per-stage LR schedule.

### 8.3 Catastrophic Forgetting Without Replay

Training Stage 3 samples without replaying Stage 1 samples will degrade Stage 1 performance. Always include a replay buffer (20–30% of batch) from prior stages. The Efficient Rehearsal paper (arxiv 2402.08096) recommends replaying "collateral damage" samples — those the model previously got correct but now gets wrong.

### 8.4 Exact Match Underestimates Quality for Multi-Line Completions

A 4-line completion that differs only in variable naming from ground truth scores zero EM. Include Edit Similarity and Pass@1 to avoid prematurely concluding that curriculum is not working.

### 8.5 FIM Format Consistency

DeepSeek-Coder uses PSM (Prefix-Suffix-Middle) format at ~50% of training tokens. When constructing FIM training pairs, always use the model's native special tokens. Do not mix PSM and SPM formats within a training run unless the base model was pretrained with both.

### 8.6 Difficulty Metric Disagreement Is Low Risk

Per arxiv 2411.00973, different difficulty metrics agree >70% of the time. Line count, token count, and cyclomatic complexity will rank the same samples as hard most of the time. Don't over-engineer the difficulty metric — correctness of the curriculum ordering matters less than consistency.

---

## 9. Suggested Curriculum Schedule for This Project

Based on the research above, here is a concrete starting schedule:

```
Dataset: python-codes-25k.jsonl (~25,000 samples)
Split: 80% train, 10% val, 10% test
Gap creation: per sample, create all valid N-line gaps

Stage 1: N = 1 missing line
  - Training samples: all 1-line gaps (~25k samples * avg N per file)
  - Steps: 1,000
  - Replay: none (first stage)
  - LR: 2e-4 constant
  - Eval: EM + loss every 200 steps

Stage 2: N = 2–3 missing lines
  - Training samples: 2-line and 3-line gaps (primary: 75%) + Stage 1 hard examples (25%)
  - Steps: 1,000
  - LR: 2e-4 constant (no reset)
  - Eval: EM + loss every 200 steps; Pass@1 on 200-sample subset at end of stage

Stage 3: N = 4–6 missing lines
  - Training samples: 4–6 line gaps (75%) + replay from Stages 1–2 (25%)
  - Steps: 1,500 (harder task needs more steps)
  - LR: 2e-4, cosine decay to 5e-5 over this stage
  - Eval: EM + Edit Similarity + Pass@1 at end of stage

Final eval: HumanEval-Infilling Pass@1 on test set, EM per N-line bucket
```

Adjust step counts based on actual dataset size after gap expansion. Rule of thumb: each stage should see ~2–3 passes over its primary difficulty subset.

---

## 10. Sources

- [Improving FIM Code Completions via Context & Curriculum Based Learning (arxiv 2412.16589)](https://arxiv.org/abs/2412.16589)
- [Curriculum Learning for Small Code Language Models (ACL SRW 2024)](https://arxiv.org/abs/2407.10194)
- [Alignment with Fill-In-the-Middle for Enhancing Code Generation (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.419.pdf)
- [Horizon-Length Prediction: Advancing FIM Capabilities (arxiv 2410.03103)](https://arxiv.org/abs/2410.03103)
- [Does the Definition of Difficulty Matter? (arxiv 2411.00973)](https://arxiv.org/abs/2411.00973)
- [Curriculum Learning for LLM Pretraining: An Analysis of Learning Dynamics (arxiv 2601.21698)](https://arxiv.org/pdf/2601.21698)
- [DeepSpeed Curriculum Learning Tutorial](https://www.deepspeed.ai/tutorials/curriculum-learning/)
- [SAFIM: Syntax-Aware Fill-In-the-Middle Benchmark (arxiv 2403.04814)](https://arxiv.org/html/2403.04814v1)
- [Efficient Training of Language Models to Fill in the Middle - OpenAI (arxiv 2207.14255)](https://arxiv.org/abs/2207.14255)
- [CodeBLEU: a Method for Automatic Evaluation of Code Synthesis (arxiv 2009.10297)](https://arxiv.org/abs/2009.10297)
- [An Efficient Rehearsal Scheme for Catastrophic Forgetting Mitigation during Multi-stage Fine-tuning (arxiv 2402.08096)](https://arxiv.org/abs/2402.08096)
- [DeepSeek-Coder Technical Paper (arxiv 2401.14196)](https://arxiv.org/pdf/2401.14196)
- [HumanEval-Infilling Benchmark](https://github.com/openai/human-eval-infilling)
